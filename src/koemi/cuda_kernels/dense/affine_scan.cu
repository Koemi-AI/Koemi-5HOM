#include <torch/extension.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

namespace {

constexpr int kThreadsPerWarp = 32;
constexpr int kWarpsPerBlock = 8;

template <typename scalar_t, typename accumulator_t>
__global__ void affine_scan_forward_kernel(
    const scalar_t* retention,
    const scalar_t* increment,
    const scalar_t* initial,
    scalar_t* states,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t feature_count,
    bool broadcast_retention) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp_in_block = threadIdx.x / kThreadsPerWarp;
    const int64_t work_item =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    const int64_t feature_tiles =
        (feature_count + kThreadsPerWarp - 1) / kThreadsPerWarp;
    const int64_t batch_index = work_item / feature_tiles;
    const int64_t feature_tile = work_item % feature_tiles;
    const int64_t feature_index = feature_tile * kThreadsPerWarp + lane;
    if (batch_index >= batch_size || feature_index >= feature_count) {
        return;
    }

    accumulator_t state = static_cast<accumulator_t>(
        initial[batch_index * feature_count + feature_index]);
    for (int64_t position = 0; position < sequence_length; ++position) {
        const int64_t state_offset =
            (batch_index * sequence_length + position) * feature_count +
            feature_index;
        const int64_t retention_offset = broadcast_retention
            ? batch_index * sequence_length + position
            : state_offset;
        const accumulator_t coefficient =
            static_cast<accumulator_t>(retention[retention_offset]);
        const accumulator_t increment_value =
            static_cast<accumulator_t>(increment[state_offset]);
        state = coefficient * state + increment_value;
        states[state_offset] = static_cast<scalar_t>(state);
    }
}

template <typename scalar_t, typename accumulator_t>
__global__ void affine_scan_backward_kernel(
    const scalar_t* retention,
    const scalar_t* initial,
    const scalar_t* states,
    const scalar_t* gradient_states,
    scalar_t* gradient_retention,
    scalar_t* gradient_increment,
    scalar_t* gradient_initial,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t feature_count,
    bool broadcast_retention) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp_in_block = threadIdx.x / kThreadsPerWarp;
    const int64_t work_item =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
    const int64_t feature_tiles =
        (feature_count + kThreadsPerWarp - 1) / kThreadsPerWarp;
    const int64_t batch_index = work_item / feature_tiles;
    const int64_t feature_tile = work_item % feature_tiles;
    const int64_t feature_index = feature_tile * kThreadsPerWarp + lane;
    if (batch_index >= batch_size || feature_index >= feature_count) {
        return;
    }

    accumulator_t adjoint = 0;
    for (int64_t position = sequence_length - 1; position >= 0; --position) {
        const int64_t state_offset =
            (batch_index * sequence_length + position) * feature_count +
            feature_index;
        const int64_t retention_offset = broadcast_retention
            ? batch_index * sequence_length + position
            : state_offset;
        const accumulator_t coefficient =
            static_cast<accumulator_t>(retention[retention_offset]);
        const accumulator_t previous_state = position == 0
            ? static_cast<accumulator_t>(
                  initial[batch_index * feature_count + feature_index])
            : static_cast<accumulator_t>(states[state_offset - feature_count]);
        adjoint += static_cast<accumulator_t>(gradient_states[state_offset]);
        gradient_increment[state_offset] = static_cast<scalar_t>(adjoint);
        gradient_retention[state_offset] =
            static_cast<scalar_t>(adjoint * previous_state);
        adjoint *= coefficient;
    }
    gradient_initial[batch_index * feature_count + feature_index] =
        static_cast<scalar_t>(adjoint);
}

int64_t feature_count(const at::Tensor& values) {
    TORCH_CHECK(values.dim() >= 2, "dense affine scan requires rank >= 2");
    TORCH_CHECK(values.size(0) > 0, "dense affine scan requires a positive batch size");
    TORCH_CHECK(values.size(1) > 0, "dense affine scan requires a positive sequence length");
    return values.numel() / (values.size(0) * values.size(1));
}

cudaStream_t current_stream(const at::Tensor& values) {
    return at::cuda::getCurrentCUDAStream(values.device().index()).stream();
}

at::Tensor affine_scan_forward(
    const at::Tensor& retention,
    const at::Tensor& increment,
    const at::Tensor& initial) {
    TORCH_CHECK(retention.is_cuda(), "retention must be a CUDA tensor");
    TORCH_CHECK(increment.is_cuda(), "increment must be a CUDA tensor");
    TORCH_CHECK(initial.is_cuda(), "initial must be a CUDA tensor");
    TORCH_CHECK(retention.is_contiguous(), "retention must be contiguous");
    TORCH_CHECK(increment.is_contiguous(), "increment must be contiguous");
    TORCH_CHECK(initial.is_contiguous(), "initial must be contiguous");
    TORCH_CHECK(retention.scalar_type() == increment.scalar_type(), "retention and increment dtypes differ");
    TORCH_CHECK(initial.scalar_type() == increment.scalar_type(), "initial and increment dtypes differ");

    const at::Tensor states = at::empty_like(increment);
    if (increment.numel() == 0) {
        return states;
    }
    const int64_t batch_size = increment.size(0);
    const int64_t sequence_length = increment.size(1);
    const int64_t features = feature_count(increment);
    const bool broadcast_retention =
        retention.numel() == batch_size * sequence_length;
    const int64_t work_items =
        batch_size * ((features + kThreadsPerWarp - 1) / kThreadsPerWarp);
    const int blocks = static_cast<int>(
        (work_items + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const auto stream = current_stream(increment);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        increment.scalar_type(),
        "koemi_dense_affine_scan_forward",
        [&] {
            using accumulator_t = at::acc_type<scalar_t, true>;
            affine_scan_forward_kernel<scalar_t, accumulator_t>
                <<<blocks, kWarpsPerBlock * kThreadsPerWarp, 0, stream>>>(
                    retention.data_ptr<scalar_t>(),
                    increment.data_ptr<scalar_t>(),
                    initial.data_ptr<scalar_t>(),
                    states.data_ptr<scalar_t>(),
                    batch_size,
                    sequence_length,
                    features,
                    broadcast_retention);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return states;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> affine_scan_backward(
    const at::Tensor& retention,
    const at::Tensor& initial,
    const at::Tensor& states,
    const at::Tensor& gradient_states) {
    TORCH_CHECK(retention.is_cuda(), "retention must be a CUDA tensor");
    TORCH_CHECK(initial.is_cuda(), "initial must be a CUDA tensor");
    TORCH_CHECK(states.is_cuda(), "states must be a CUDA tensor");
    TORCH_CHECK(gradient_states.is_cuda(), "gradient_states must be a CUDA tensor");
    TORCH_CHECK(retention.is_contiguous(), "retention must be contiguous");
    TORCH_CHECK(initial.is_contiguous(), "initial must be contiguous");
    TORCH_CHECK(states.is_contiguous(), "states must be contiguous");
    TORCH_CHECK(gradient_states.is_contiguous(), "gradient_states must be contiguous");
    TORCH_CHECK(retention.scalar_type() == states.scalar_type(), "retention and states dtypes differ");
    TORCH_CHECK(initial.scalar_type() == states.scalar_type(), "initial and states dtypes differ");
    TORCH_CHECK(gradient_states.scalar_type() == states.scalar_type(), "gradient_states and states dtypes differ");

    const at::Tensor gradient_retention = at::empty_like(states);
    const at::Tensor gradient_increment = at::empty_like(states);
    const at::Tensor gradient_initial = at::zeros_like(initial);
    if (states.numel() == 0) {
        return std::make_tuple(gradient_retention, gradient_increment, gradient_initial);
    }
    const int64_t batch_size = states.size(0);
    const int64_t sequence_length = states.size(1);
    const int64_t features = feature_count(states);
    const bool broadcast_retention =
        retention.numel() == batch_size * sequence_length;
    const int64_t work_items =
        batch_size * ((features + kThreadsPerWarp - 1) / kThreadsPerWarp);
    const int blocks = static_cast<int>(
        (work_items + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const auto stream = current_stream(states);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        states.scalar_type(),
        "koemi_dense_affine_scan_backward",
        [&] {
            using accumulator_t = at::acc_type<scalar_t, true>;
            affine_scan_backward_kernel<scalar_t, accumulator_t>
                <<<blocks, kWarpsPerBlock * kThreadsPerWarp, 0, stream>>>(
                    retention.data_ptr<scalar_t>(),
                    initial.data_ptr<scalar_t>(),
                    states.data_ptr<scalar_t>(),
                    gradient_states.data_ptr<scalar_t>(),
                    gradient_retention.data_ptr<scalar_t>(),
                    gradient_increment.data_ptr<scalar_t>(),
                    gradient_initial.data_ptr<scalar_t>(),
                    batch_size,
                    sequence_length,
                    features,
                    broadcast_retention);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(gradient_retention, gradient_increment, gradient_initial);
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &affine_scan_forward);
    module.def("backward", &affine_scan_backward);
}
