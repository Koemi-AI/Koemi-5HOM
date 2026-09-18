#include <torch/extension.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <vector>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__device__ float as_float(const scalar_t value) {
    return static_cast<float>(value);
}

template <typename scalar_t>
__device__ scalar_t from_float(const float value) {
    return static_cast<scalar_t>(value);
}

__device__ void reduce_max(float* values, const int thread_index) {
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            values[thread_index] = fmaxf(values[thread_index], values[thread_index + stride]);
        }
        __syncthreads();
    }
}

__device__ void reduce_sum(float* values, const int thread_index) {
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            values[thread_index] += values[thread_index + stride];
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__device__ float row_logit(
    const scalar_t* prior_row,
    const scalar_t* weight_row,
    const scalar_t* bias,
    const int64_t width) {
    float value = as_float(*bias);
    for (int64_t dimension = 0; dimension < width; ++dimension) {
        value += as_float(prior_row[dimension]) * as_float(weight_row[dimension]);
    }
    return value;
}

template <typename scalar_t>
__global__ void causal_surprise_forward_kernel(
    const scalar_t* __restrict__ prior_states,
    const scalar_t* __restrict__ content_weight,
    const scalar_t* __restrict__ content_bias,
    const int64_t* __restrict__ target_ids,
    const bool* __restrict__ valid_mask,
    scalar_t* __restrict__ output,
    const int64_t row_count,
    const int64_t width,
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

    extern __shared__ float reduction[];
    float* maximum_values = reduction;
    float* sum_values = reduction + kThreads;
    const scalar_t* prior_row = prior_states + row_index * width;

    float maximum = -FLT_MAX;
    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        maximum = fmaxf(
            maximum,
            row_logit(
                prior_row,
                content_weight + vocabulary_index * width,
                content_bias + vocabulary_index,
                width));
    }
    maximum_values[thread_index] = maximum;
    __syncthreads();
    reduce_max(maximum_values, thread_index);
    const float row_maximum = maximum_values[0];

    float exponential_sum = 0.0f;
    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        const float logit = row_logit(
            prior_row,
            content_weight + vocabulary_index * width,
            content_bias + vocabulary_index,
            width);
        exponential_sum += expf(logit - row_maximum);
    }
    sum_values[thread_index] = exponential_sum;
    __syncthreads();
    reduce_sum(sum_values, thread_index);

    if (thread_index == 0) {
        const int64_t target = target_ids[row_index];
        const float target_logit = row_logit(
            prior_row,
            content_weight + target * width,
            content_bias + target,
            width);
        const float negative_log_likelihood = logf(sum_values[0]) + row_maximum - target_logit;
        const float surprise = 1.0f - expf(-negative_log_likelihood / log_content_vocab);
        output[row_index] = from_float<scalar_t>(surprise);
    }
}

template <typename scalar_t>
__global__ void causal_surprise_backward_kernel(
    const scalar_t* __restrict__ prior_states,
    const scalar_t* __restrict__ content_weight,
    const scalar_t* __restrict__ content_bias,
    const int64_t* __restrict__ target_ids,
    const bool* __restrict__ valid_mask,
    const scalar_t* __restrict__ gradient,
    float* __restrict__ gradient_prior,
    float* __restrict__ gradient_weight,
    float* __restrict__ gradient_bias,
    const int64_t row_count,
    const int64_t width,
    const int64_t content_vocab,
    const float log_content_vocab) {
    const int64_t row_index = static_cast<int64_t>(blockIdx.x);
    const int thread_index = threadIdx.x;
    if (row_index >= row_count) {
        return;
    }

    extern __shared__ float reduction[];
    float* maximum_values = reduction;
    float* sum_values = reduction + kThreads;
    float* row_gradient = reduction + 2 * kThreads;
    for (int64_t dimension = thread_index; dimension < width; dimension += kThreads) {
        row_gradient[dimension] = 0.0f;
    }
    __syncthreads();
    if (!valid_mask[row_index]) {
        return;
    }

    const scalar_t* prior_row = prior_states + row_index * width;
    float maximum = -FLT_MAX;
    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        maximum = fmaxf(
            maximum,
            row_logit(
                prior_row,
                content_weight + vocabulary_index * width,
                content_bias + vocabulary_index,
                width));
    }
    maximum_values[thread_index] = maximum;
    __syncthreads();
    reduce_max(maximum_values, thread_index);
    const float row_maximum = maximum_values[0];

    float exponential_sum = 0.0f;
    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        const float logit = row_logit(
            prior_row,
            content_weight + vocabulary_index * width,
            content_bias + vocabulary_index,
            width);
        exponential_sum += expf(logit - row_maximum);
    }
    sum_values[thread_index] = exponential_sum;
    __syncthreads();
    reduce_sum(sum_values, thread_index);

    const int64_t target = target_ids[row_index];
    const float target_logit = row_logit(
        prior_row,
        content_weight + target * width,
        content_bias + target,
        width);
    const float negative_log_likelihood = logf(sum_values[0]) + row_maximum - target_logit;
    const float gradient_surprise = as_float(gradient[row_index]);
    const float gradient_nll = gradient_surprise * expf(-negative_log_likelihood / log_content_vocab) / log_content_vocab;

    for (int64_t vocabulary_index = thread_index;
         vocabulary_index < content_vocab;
         vocabulary_index += kThreads) {
        const float logit = row_logit(
            prior_row,
            content_weight + vocabulary_index * width,
            content_bias + vocabulary_index,
            width);
        const float probability = expf(logit - row_maximum) / sum_values[0];
        const float logit_gradient = gradient_nll * (probability - (vocabulary_index == target ? 1.0f : 0.0f));
        for (int64_t dimension = 0; dimension < width; ++dimension) {
            atomicAdd(
                row_gradient + dimension,
                logit_gradient * as_float(content_weight[vocabulary_index * width + dimension]));
            atomicAdd(
                gradient_weight + vocabulary_index * width + dimension,
                logit_gradient * as_float(prior_row[dimension]));
        }
        atomicAdd(gradient_bias + vocabulary_index, logit_gradient);
    }
    __syncthreads();
    for (int64_t dimension = thread_index; dimension < width; dimension += kThreads) {
        gradient_prior[row_index * width + dimension] = row_gradient[dimension];
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
    causal_surprise_forward_kernel<scalar_t><<<
        static_cast<unsigned int>(row_count),
        kThreads,
        2 * kThreads * sizeof(float),
        stream>>>(
        prior_states.data_ptr<scalar_t>(),
        content_weight.data_ptr<scalar_t>(),
        content_bias.data_ptr<scalar_t>(),
        target_ids.data_ptr<int64_t>(),
        valid_mask.data_ptr<bool>(),
        output.data_ptr<scalar_t>(),
        row_count,
        width,
        content_vocab,
        log_content_vocab);
}

template <typename scalar_t>
void launch_backward(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask,
    const torch::Tensor& gradient,
    torch::Tensor& gradient_prior,
    torch::Tensor& gradient_weight,
    torch::Tensor& gradient_bias) {
    const auto row_count = prior_states.numel() / prior_states.size(2);
    const auto width = prior_states.size(2);
    const auto content_vocab = content_weight.size(0);
    const float log_content_vocab = std::log(static_cast<float>(content_vocab));
    const auto stream = at::cuda::getCurrentCUDAStream(prior_states.device().index()).stream();
    causal_surprise_backward_kernel<scalar_t><<<
        static_cast<unsigned int>(row_count),
        kThreads,
        (2 * kThreads + width) * sizeof(float),
        stream>>>(
        prior_states.data_ptr<scalar_t>(),
        content_weight.data_ptr<scalar_t>(),
        content_bias.data_ptr<scalar_t>(),
        target_ids.data_ptr<int64_t>(),
        valid_mask.data_ptr<bool>(),
        gradient.data_ptr<scalar_t>(),
        gradient_prior.data_ptr<float>(),
        gradient_weight.data_ptr<float>(),
        gradient_bias.data_ptr<float>(),
        row_count,
        width,
        content_vocab,
        log_content_vocab);
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
    auto gradient_prior = torch::zeros(
        prior_states.sizes(), prior_states.options().dtype(torch::kFloat32));
    auto gradient_weight = torch::zeros(
        content_weight.sizes(), content_weight.options().dtype(torch::kFloat32));
    auto gradient_bias = torch::zeros(
        content_bias.sizes(), content_bias.options().dtype(torch::kFloat32));
    if (prior_states.numel() == 0) {
        return {gradient_prior, gradient_weight, gradient_bias};
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        prior_states.scalar_type(),
        "causal_surprise_backward_cuda",
        [&] {
            launch_backward<scalar_t>(
                prior_states,
                content_weight,
                content_bias,
                target_ids,
                valid_mask,
                gradient,
                gradient_prior,
                gradient_weight,
                gradient_bias);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gradient_prior, gradient_weight, gradient_bias};
}
