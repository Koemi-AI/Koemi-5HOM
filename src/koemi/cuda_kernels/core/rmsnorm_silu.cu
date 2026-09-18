#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace koemi_cuda_core {
namespace {

constexpr int kWarpSize = 32;
constexpr int kMaximumWarps = 16;

template <typename scalar_t>
__device__ __forceinline__ float as_float(scalar_t value) {
    return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t as_scalar(float value) {
    return static_cast<scalar_t>(value);
}

__device__ __forceinline__ float silu_value(float value) {
    const float sigmoid = 1.0f / (1.0f + expf(-value));
    return value * sigmoid;
}

__device__ __forceinline__ float silu_derivative(float value) {
    const float sigmoid = 1.0f / (1.0f + expf(-value));
    return sigmoid * (1.0f + value * (1.0f - sigmoid));
}

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;
    value = threadIdx.x < warp_count ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) {
            warp_sums[0] = value;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

template <typename scalar_t>
__global__ void rmsnorm_silu_forward_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    float* inverse_rms,
    int64_t rows,
    int64_t hidden,
    float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    __shared__ float warp_sums[kMaximumWarps];
    const int64_t row_offset = row * hidden;
    float square_sum = 0.0f;
    for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        const float value = as_float(input[row_offset + column]);
        square_sum = fmaf(value, value, square_sum);
    }
    const float block_square_sum = block_sum(square_sum, warp_sums);
    const float normalized_inverse = rsqrtf(block_square_sum / static_cast<float>(hidden) + eps);
    if (threadIdx.x == 0) {
        inverse_rms[row] = normalized_inverse;
    }
    for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        const float normalized = as_float(input[row_offset + column]) * normalized_inverse * as_float(weight[column]);
        output[row_offset + column] = as_scalar<scalar_t>(silu_value(normalized));
    }
}

template <typename scalar_t>
__global__ void rmsnorm_silu_backward_kernel(
    const scalar_t* gradient_output,
    const scalar_t* input,
    const scalar_t* weight,
    const float* inverse_rms,
    scalar_t* gradient_input,
    float* gradient_weight,
    int64_t rows,
    int64_t hidden) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    __shared__ float warp_sums[kMaximumWarps];
    const int64_t row_offset = row * hidden;
    const float inverse = inverse_rms[row];
    float dot_product = 0.0f;
    for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        const float input_value = as_float(input[row_offset + column]);
        const float normalized = input_value * inverse;
        const float pre_activation = normalized * as_float(weight[column]);
        const float upstream = as_float(gradient_output[row_offset + column]);
        const float derivative = upstream * silu_derivative(pre_activation);
        dot_product = fmaf(derivative * as_float(weight[column]), input_value, dot_product);
    }
    const float reduced_dot_product = block_sum(dot_product, warp_sums);
    const float correction_scale = inverse * inverse * inverse / static_cast<float>(hidden);
    for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        const float input_value = as_float(input[row_offset + column]);
        const float normalized = input_value * inverse;
        const float pre_activation = normalized * as_float(weight[column]);
        const float upstream = as_float(gradient_output[row_offset + column]);
        const float derivative = upstream * silu_derivative(pre_activation);
        const float derivative_normalized = derivative * as_float(weight[column]);
        const float input_gradient = inverse * derivative_normalized - input_value * correction_scale * reduced_dot_product;
        gradient_input[row_offset + column] = as_scalar<scalar_t>(input_gradient);
        atomicAdd(&gradient_weight[column], derivative * normalized);
    }
}

int threads_for_hidden(int64_t hidden) {
    if (hidden <= 32) {
        return 32;
    }
    if (hidden <= 64) {
        return 64;
    }
    if (hidden <= 128) {
        return 128;
    }
    return 256;
}

template <typename scalar_t>
void launch_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& inverse_rms,
    float eps,
    cudaStream_t stream) {
    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    const int threads = threads_for_hidden(hidden);
    const dim3 grid(static_cast<unsigned int>(rows));
    const std::size_t shared_bytes = static_cast<std::size_t>(threads / kWarpSize) * sizeof(float);
    rmsnorm_silu_forward_kernel<scalar_t><<<grid, threads, shared_bytes, stream>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        inverse_rms.data_ptr<float>(),
        rows,
        hidden,
        eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward(
    const torch::Tensor& gradient_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& inverse_rms,
    torch::Tensor& gradient_input,
    torch::Tensor& gradient_weight,
    cudaStream_t stream) {
    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    const int threads = threads_for_hidden(hidden);
    const dim3 grid(static_cast<unsigned int>(rows));
    const std::size_t shared_bytes = static_cast<std::size_t>(threads / kWarpSize) * sizeof(float);
    rmsnorm_silu_backward_kernel<scalar_t><<<grid, threads, shared_bytes, stream>>>(
        gradient_output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        inverse_rms.data_ptr<float>(),
        gradient_input.data_ptr<scalar_t>(),
        gradient_weight.data_ptr<float>(),
        rows,
        hidden);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void validate_grid_size(int64_t rows) {
    TORCH_CHECK(rows <= std::numeric_limits<unsigned int>::max(), "input has too many rows for one CUDA launch");
}

}

std::vector<torch::Tensor> rmsnorm_silu_forward_cuda(torch::Tensor input, torch::Tensor weight, double eps) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto output = torch::empty_like(input);
    auto inverse_rms = torch::empty({input.size(0)}, input.options().dtype(torch::kFloat32));
    if (input.size(0) == 0) {
        return {output, inverse_rms};
    }
    validate_grid_size(input.size(0));
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(input.device().index());
    const float float_eps = static_cast<float>(eps);
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, input.scalar_type(), "rmsnorm_silu_forward_cuda", [&] {
        launch_forward<scalar_t>(input, weight, output, inverse_rms, float_eps, stream);
    });
    return {output, inverse_rms};
}

std::vector<torch::Tensor> rmsnorm_silu_backward_cuda(
    torch::Tensor gradient_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inverse_rms,
    double eps) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto gradient_input = torch::empty_like(input);
    auto gradient_weight_float = torch::zeros({weight.size(0)}, weight.options().dtype(torch::kFloat32));
    if (input.size(0) == 0) {
        return {gradient_input, torch::zeros_like(weight)};
    }
    validate_grid_size(input.size(0));
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(input.device().index());
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, input.scalar_type(), "rmsnorm_silu_backward_cuda", [&] {
        launch_backward<scalar_t>(
            gradient_output,
            input,
            weight,
            inverse_rms,
            gradient_input,
            gradient_weight_float,
            stream);
    });
    auto gradient_weight = gradient_weight_float.to(weight.scalar_type());
    return {gradient_input, gradient_weight};
}

}
