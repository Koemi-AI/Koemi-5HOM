#include <torch/extension.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/mm.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

template <typename scalar_t>
__device__ float as_float(const scalar_t value) {
    return static_cast<float>(value);
}

template <typename scalar_t>
__device__ scalar_t from_float(const float value) {
    return static_cast<scalar_t>(value);
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_max(float value, float* workspace) {
    const int thread_index = threadIdx.x;
    const int lane = thread_index & 31;
    const int warp = thread_index >> 5;
    value = warp_reduce_max(value);
    if (lane == 0) {
        workspace[warp] = value;
    }
    __syncthreads();
    value = thread_index < kWarps ? workspace[lane] : -FLT_MAX;
    if (warp == 0) {
        value = warp_reduce_max(value);
    }
    if (thread_index == 0) {
        workspace[0] = value;
    }
    __syncthreads();
    return workspace[0];
}

__device__ __forceinline__ float block_reduce_sum(float value, float* workspace) {
    const int thread_index = threadIdx.x;
    const int lane = thread_index & 31;
    const int warp = thread_index >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        workspace[warp] = value;
    }
    __syncthreads();
    value = thread_index < kWarps ? workspace[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
    }
    if (thread_index == 0) {
        workspace[0] = value;
    }
    __syncthreads();
    return workspace[0];
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void causal_surprise_logits_forward_kernel(
    const float* __restrict__ logits,
    const scalar_t* __restrict__ content_bias,
    const int64_t* __restrict__ target_ids,
    const bool* __restrict__ valid_mask,
    scalar_t* __restrict__ output,
    const int64_t row_count,
    const int64_t content_vocab,
    const float log_content_vocab) {
    const int64_t row_index = static_cast<int64_t>(blockIdx.x);
    const int thread_index = threadIdx.x;
    if (row_index >= row_count) {
        return;
    }
    if (!valid_mask[row_index]) {
        if (thread_index == 0) {
            output[row_index] = from_float<scalar_t>(0.0f);
        }
        return;
    }

    extern __shared__ float workspace[];
    const int64_t row_offset = row_index * content_vocab;
    float maximum = -FLT_MAX;
    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        maximum = fmaxf(
            maximum,
            logits[row_offset + vocabulary_index] + as_float(content_bias[vocabulary_index]));
    }
    const float row_maximum = block_reduce_max(maximum, workspace);

    float exponential_sum = 0.0f;
    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        const float logit = logits[row_offset + vocabulary_index] + as_float(content_bias[vocabulary_index]);
        exponential_sum += expf(logit - row_maximum);
    }
    const float partition = block_reduce_sum(exponential_sum, workspace);

    if (thread_index == 0) {
        const int64_t target = target_ids[row_index];
        const float target_logit = logits[row_offset + target] + as_float(content_bias[target]);
        const float negative_log_likelihood = logf(partition) + row_maximum - target_logit;
        const float surprise = 1.0f - expf(-negative_log_likelihood / log_content_vocab);
        output[row_index] = from_float<scalar_t>(surprise);
    }
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads) void causal_surprise_logits_backward_kernel(
    const float* __restrict__ logits,
    const scalar_t* __restrict__ content_bias,
    const int64_t* __restrict__ target_ids,
    const bool* __restrict__ valid_mask,
    const scalar_t* __restrict__ gradient,
    float* __restrict__ logit_gradient,
    const int64_t row_count,
    const int64_t content_vocab,
    const float log_content_vocab) {
    const int64_t row_index = static_cast<int64_t>(blockIdx.x);
    const int thread_index = threadIdx.x;
    if (row_index >= row_count) {
        return;
    }

    if (!valid_mask[row_index]) {
        return;
    }

    extern __shared__ float workspace[];
    const int64_t row_offset = row_index * content_vocab;
    float maximum = -FLT_MAX;
    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        maximum = fmaxf(
            maximum,
            logits[row_offset + vocabulary_index] + as_float(content_bias[vocabulary_index]));
    }
    const float row_maximum = block_reduce_max(maximum, workspace);

    float exponential_sum = 0.0f;
    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        const float logit = logits[row_offset + vocabulary_index] + as_float(content_bias[vocabulary_index]);
        exponential_sum += expf(logit - row_maximum);
    }
    const float partition = block_reduce_sum(exponential_sum, workspace);

    const int64_t target = target_ids[row_index];
    const float target_logit = logits[row_offset + target] + as_float(content_bias[target]);
    const float negative_log_likelihood = logf(partition) + row_maximum - target_logit;
    const float gradient_surprise = as_float(gradient[row_index]);
    const float gradient_nll = gradient_surprise * expf(-negative_log_likelihood / log_content_vocab) / log_content_vocab;

    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        const float logit = logits[row_offset + vocabulary_index] + as_float(content_bias[vocabulary_index]);
        const float probability = expf(logit - row_maximum) / partition;
        logit_gradient[row_offset + vocabulary_index] = gradient_nll *
            (probability - (vocabulary_index == target ? 1.0f : 0.0f));
    }
}

template <typename scalar_t>
void launch_forward(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask,
    torch::Tensor& output) {
    const auto row_count = prior_states.numel() / prior_states.size(2);
    const auto width = prior_states.size(2);
    const auto content_vocab = content_weight.size(0);
    const float log_content_vocab = std::log(static_cast<float>(content_vocab));
    const auto stream = at::cuda::getCurrentCUDAStream(prior_states.device().index()).stream();
    const auto prior_float = prior_states.view({row_count, width}).to(torch::kFloat32);
    const auto weight_float = content_weight.to(torch::kFloat32);
    const auto logits = at::mm(prior_float, weight_float.transpose(0, 1));
    causal_surprise_logits_forward_kernel<scalar_t><<<
        static_cast<unsigned int>(row_count),
        kThreads,
        kWarps * sizeof(float),
        stream>>>(
        logits.data_ptr<float>(),
        content_bias.data_ptr<scalar_t>(),
        target_ids.data_ptr<int64_t>(),
        valid_mask.data_ptr<bool>(),
        output.data_ptr<scalar_t>(),
        row_count,
        content_vocab,
        log_content_vocab);
}

template <typename scalar_t>
std::vector<torch::Tensor> launch_backward(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask,
    const torch::Tensor& gradient) {
    const auto row_count = prior_states.numel() / prior_states.size(2);
    const auto width = prior_states.size(2);
    const auto content_vocab = content_weight.size(0);
    const float log_content_vocab = std::log(static_cast<float>(content_vocab));
    const auto stream = at::cuda::getCurrentCUDAStream(prior_states.device().index()).stream();
    const auto prior_float = prior_states.view({row_count, width}).to(torch::kFloat32);
    const auto weight_float = content_weight.to(torch::kFloat32);
    const auto logits = at::mm(prior_float, weight_float.transpose(0, 1));
    auto logit_gradient = torch::zeros_like(logits);
    causal_surprise_logits_backward_kernel<scalar_t><<<
        static_cast<unsigned int>(row_count),
        kThreads,
        kWarps * sizeof(float),
        stream>>>(
        logits.data_ptr<float>(),
        content_bias.data_ptr<scalar_t>(),
        target_ids.data_ptr<int64_t>(),
        valid_mask.data_ptr<bool>(),
        gradient.data_ptr<scalar_t>(),
        logit_gradient.data_ptr<float>(),
        row_count,
        content_vocab,
        log_content_vocab);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gradient_prior = at::mm(logit_gradient, weight_float).view(prior_states.sizes());
    auto gradient_weight = at::mm(logit_gradient.transpose(0, 1), prior_float);
    auto gradient_bias = logit_gradient.sum(0);
    return {gradient_prior, gradient_weight, gradient_bias};
}

}

torch::Tensor causal_surprise_forward_cuda(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask) {
    const c10::cuda::CUDAGuard device_guard(prior_states.device());
    auto output = torch::zeros({prior_states.size(0), prior_states.size(1)}, prior_states.options());
    if (output.numel() == 0) {
        return output;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        prior_states.scalar_type(),
        "causal_surprise_forward_cuda",
        [&] {
            launch_forward<scalar_t>(
                prior_states,
                content_weight,
                content_bias,
                target_ids,
                valid_mask,
                output);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> causal_surprise_backward_cuda(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask,
    const torch::Tensor& gradient) {
    const c10::cuda::CUDAGuard device_guard(prior_states.device());
    if (prior_states.numel() == 0) {
        auto gradient_prior = torch::zeros(
            prior_states.sizes(), prior_states.options().dtype(torch::kFloat32));
        auto gradient_weight = torch::zeros(
            content_weight.sizes(), content_weight.options().dtype(torch::kFloat32));
        auto gradient_bias = torch::zeros(
            content_bias.sizes(), content_bias.options().dtype(torch::kFloat32));
        return {gradient_prior, gradient_weight, gradient_bias};
    }
    std::vector<torch::Tensor> gradients;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        prior_states.scalar_type(),
        "causal_surprise_backward_cuda",
        [&] {
            gradients = launch_backward<scalar_t>(
                prior_states,
                content_weight,
                content_bias,
                target_ids,
                valid_mask,
                gradient);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gradients;
}
