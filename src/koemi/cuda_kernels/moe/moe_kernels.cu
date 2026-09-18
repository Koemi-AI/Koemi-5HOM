#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace koemi_cuda_moe {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaximumSharedBytes = 96 * 1024;

template <typename scalar_t>
__device__ __forceinline__ float as_float(scalar_t value) {
    return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t as_scalar(float value) {
    return static_cast<scalar_t>(value);
}

__device__ __forceinline__ float silu_value(float value) {
    return value / (1.0f + expf(-value));
}

template <typename index_t>
__device__ __forceinline__ uint32_t content_hash(index_t token_id, index_t previous_token_id) {
    const uint64_t mixed = static_cast<uint64_t>(token_id) * 1000003ULL +
        static_cast<uint64_t>(previous_token_id) * 97409ULL;
    uint32_t value = static_cast<uint32_t>(mixed);
    value ^= value >> 16;
    value *= 0x45D9F3BU;
    value ^= value >> 16;
    value *= 0x45D9F3BU;
    return value ^ (value >> 16);
}

template <typename index_t>
__global__ void route_topk_kernel(
    const index_t* token_ids,
    const index_t* previous_token_ids,
    const bool* valid_mask,
    int64_t* assignments,
    int64_t token_count,
    int64_t expert_count,
    int64_t top_k) {
    const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= token_count) {
        return;
    }
    const bool valid = valid_mask[token];
    const uint32_t hash = content_hash(token_ids[token], previous_token_ids[token]);
    for (int64_t slot = 0; slot < top_k; ++slot) {
        const uint32_t offset = static_cast<uint32_t>(slot) * 0x9E3779B9U;
        assignments[token * top_k + slot] = valid
            ? static_cast<int64_t>((hash + offset) % static_cast<uint32_t>(expert_count))
            : -1;
    }
}

template <typename index_t>
__global__ void build_pair_keys_kernel(
    const index_t* assignments,
    const bool* valid_mask,
    int64_t* pair_keys,
    int64_t* pair_values,
    int64_t token_count,
    int64_t top_k,
    int64_t expert_count) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t pair_count = token_count * top_k;
    if (pair >= pair_count) {
        return;
    }
    const int64_t token = pair / top_k;
    const int64_t slot = pair - token * top_k;
    const index_t expert = assignments[pair];
    bool valid = valid_mask[token] && expert >= 0 && expert < expert_count;
    for (int64_t earlier_slot = 0; earlier_slot < slot && valid; ++earlier_slot) {
        valid = assignments[token * top_k + earlier_slot] != expert;
    }
    pair_keys[pair] = valid ? static_cast<int64_t>(expert) : expert_count;
    pair_values[pair] = pair;
}

__global__ void histogram_kernel(
    const int64_t* sorted_experts,
    int64_t* counts,
    int64_t pair_count,
    int64_t expert_count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pair_count) {
        return;
    }
    const int64_t expert = sorted_experts[index];
    if (expert >= 0 && expert <= expert_count) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(counts + expert),
            static_cast<unsigned long long>(1));
    }
}

__global__ void gather_sorted_rows_kernel(
    const int64_t* sorted_pairs,
    int64_t* sorted_rows,
    int64_t pair_count,
    int64_t top_k) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pair_count) {
        return;
    }
    sorted_rows[index] = sorted_pairs[index] / top_k;
}

template <typename scalar_t>
__global__ void gather_sorted_context_kernel(
    const scalar_t* context,
    const int64_t* sorted_pairs,
    const int64_t* valid_pair_count,
    scalar_t* sorted_context,
    int64_t pair_count,
    int64_t width,
    int64_t top_k) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = pair_count * width;
    if (linear >= total) {
        return;
    }
    const int64_t sorted_pair = linear / width;
    const int64_t column = linear - sorted_pair * width;
    if (sorted_pair >= *valid_pair_count) {
        sorted_context[linear] = as_scalar<scalar_t>(0.0f);
        return;
    }
    const int64_t source_pair = sorted_pairs[sorted_pair];
    const int64_t source_token = source_pair / top_k;
    sorted_context[linear] = context[source_token * width + column];
}

template <typename scalar_t>
__global__ void grouped_expert_mlp_kernel(
    const scalar_t* sorted_context,
    const int64_t* sorted_experts,
    const int64_t* segment_offsets,
    const scalar_t* gate_weights,
    const scalar_t* gate_biases,
    const scalar_t* value_weights,
    const scalar_t* value_biases,
    const scalar_t* output_weights,
    const scalar_t* output_biases,
    scalar_t* output,
    int64_t pair_count,
    int64_t input_width,
    int64_t hidden_width,
    int64_t expert_count) {
    extern __shared__ float hidden_values[];
    float* gated_values = hidden_values;
    float* value_values = hidden_values + hidden_width;
    const int64_t pair = static_cast<int64_t>(blockIdx.x);
    if (pair >= pair_count || pair >= segment_offsets[expert_count]) {
        return;
    }
    const int64_t expert = sorted_experts[pair];
    if (expert < 0 || expert >= expert_count) {
        return;
    }
    const int64_t input_offset = pair * input_width;
    const int64_t gate_offset = expert * hidden_width * input_width;
    const int64_t output_offset = expert * input_width * hidden_width;
    for (int64_t hidden = threadIdx.x; hidden < hidden_width; hidden += blockDim.x) {
        float gate_value = as_float(gate_biases[expert * hidden_width + hidden]);
        float value_value = as_float(value_biases[expert * hidden_width + hidden]);
        for (int64_t column = 0; column < input_width; ++column) {
            const float input_value = as_float(sorted_context[input_offset + column]);
            gate_value = fmaf(input_value, as_float(gate_weights[gate_offset + hidden * input_width + column]), gate_value);
            value_value = fmaf(input_value, as_float(value_weights[gate_offset + hidden * input_width + column]), value_value);
        }
        gated_values[hidden] = silu_value(gate_value);
        value_values[hidden] = value_value;
    }
    __syncthreads();
    for (int64_t column = threadIdx.x; column < input_width; column += blockDim.x) {
        float result = as_float(output_biases[expert * input_width + column]);
        for (int64_t hidden = 0; hidden < hidden_width; ++hidden) {
            result = fmaf(
                gated_values[hidden] * value_values[hidden],
                as_float(output_weights[output_offset + column * hidden_width + hidden]),
                result);
        }
        output[input_offset + column] = as_scalar<scalar_t>(result);
    }
}

template <typename scalar_t>
__global__ void combine_topk_kernel(
    const scalar_t* expert_outputs,
    const int64_t* sorted_rows,
    const int64_t* segment_offsets,
    float* combined,
    int64_t pair_count,
    int64_t width,
    int64_t expert_count,
    int64_t token_count,
    int64_t top_k) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = pair_count * width;
    if (linear >= total) {
        return;
    }
    const int64_t pair = linear / width;
    const int64_t column = linear - pair * width;
    if (pair >= segment_offsets[expert_count]) {
        return;
    }
    const int64_t row = sorted_rows[pair];
    if (row < 0 || row >= token_count) {
        return;
    }
    atomicAdd(
        combined + row * width + column,
        as_float(expert_outputs[linear]) / static_cast<float>(top_k));
}

int threads_for_width(int64_t width) {
    if (width <= 32) {
        return 32;
    }
    if (width <= 64) {
        return 64;
    }
    if (width <= 128) {
        return 128;
    }
    return kThreads;
}

unsigned int blocks_for(int64_t items) {
    const int64_t blocks = (items + kThreads - 1) / kThreads;
    TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(), "CUDA launch has too many blocks");
    return static_cast<unsigned int>(blocks);
}

template <typename scalar_t>
void launch_gather_sorted_context(
    const torch::Tensor& context,
    const torch::Tensor& sorted_pairs,
    const torch::Tensor& segment_offsets,
    torch::Tensor& sorted_context,
    int64_t top_k,
    cudaStream_t stream) {
    const int64_t pair_count = sorted_pairs.size(0);
    const int64_t width = context.size(1);
    const int64_t total = pair_count * width;
    if (total == 0) {
        return;
    }
    gather_sorted_context_kernel<scalar_t><<<blocks_for(total), kThreads, 0, stream>>>(
        context.data_ptr<scalar_t>(),
        sorted_pairs.data_ptr<int64_t>(),
        segment_offsets.data_ptr<int64_t>() + segment_offsets.size(0) - 1,
        sorted_context.data_ptr<scalar_t>(),
        pair_count,
        width,
        top_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_grouped_expert_mlp(
    const torch::Tensor& sorted_context,
    const torch::Tensor& sorted_experts,
    const torch::Tensor& segment_offsets,
    const torch::Tensor& gate_weights,
    const torch::Tensor& gate_biases,
    const torch::Tensor& value_weights,
    const torch::Tensor& value_biases,
    const torch::Tensor& output_weights,
    const torch::Tensor& output_biases,
    torch::Tensor& output,
    cudaStream_t stream) {
    const int64_t pair_count = sorted_context.size(0);
    const int64_t input_width = sorted_context.size(1);
    const int64_t hidden_width = gate_weights.size(1);
    const int64_t expert_count = gate_weights.size(0);
    const std::size_t shared_bytes = static_cast<std::size_t>(hidden_width) * 2 * sizeof(float);
    TORCH_CHECK(shared_bytes <= kMaximumSharedBytes, "hidden width requires more than 96 KiB of shared memory");
    if (pair_count == 0) {
        return;
    }
    if (shared_bytes > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            grouped_expert_mlp_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }
    const int threads = threads_for_width(std::max(input_width, hidden_width));
    grouped_expert_mlp_kernel<scalar_t><<<static_cast<unsigned int>(pair_count), threads, shared_bytes, stream>>>(
        sorted_context.data_ptr<scalar_t>(),
        sorted_experts.data_ptr<int64_t>(),
        segment_offsets.data_ptr<int64_t>(),
        gate_weights.data_ptr<scalar_t>(),
        gate_biases.data_ptr<scalar_t>(),
        value_weights.data_ptr<scalar_t>(),
        value_biases.data_ptr<scalar_t>(),
        output_weights.data_ptr<scalar_t>(),
        output_biases.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        pair_count,
        input_width,
        hidden_width,
        expert_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_combine_topk(
    const torch::Tensor& expert_outputs,
    const torch::Tensor& sorted_rows,
    const torch::Tensor& segment_offsets,
    torch::Tensor& combined,
    int64_t token_count,
    int64_t top_k,
    cudaStream_t stream) {
    const int64_t pair_count = expert_outputs.size(0);
    const int64_t width = expert_outputs.size(1);
    const int64_t total = pair_count * width;
    if (total == 0) {
        return;
    }
    combine_topk_kernel<scalar_t><<<blocks_for(total), kThreads, 0, stream>>>(
        expert_outputs.data_ptr<scalar_t>(),
        sorted_rows.data_ptr<int64_t>(),
        segment_offsets.data_ptr<int64_t>(),
        combined.data_ptr<float>(),
        pair_count,
        width,
        segment_offsets.size(0) - 1,
        token_count,
        top_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}

torch::Tensor route_topk_cuda(
    const torch::Tensor& token_ids,
    const torch::Tensor& previous_token_ids,
    const torch::Tensor& valid_mask,
    int64_t expert_count,
    int64_t top_k) {
    c10::cuda::CUDAGuard device_guard(token_ids.device());
    auto assignments = torch::empty(
        {token_ids.numel(), top_k},
        token_ids.options().dtype(torch::kInt64));
    if (token_ids.numel() == 0) {
        return assignments;
    }
    const auto stream = at::cuda::getDefaultCUDAStream(token_ids.device().index());
    const unsigned int blocks = static_cast<unsigned int>((token_ids.numel() + kThreads - 1) / kThreads);
    if (token_ids.scalar_type() == torch::kInt) {
        route_topk_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
            token_ids.data_ptr<int32_t>(),
            previous_token_ids.data_ptr<int32_t>(),
            valid_mask.data_ptr<bool>(),
            assignments.data_ptr<int64_t>(),
            token_ids.numel(),
            expert_count,
            top_k);
    } else {
        route_topk_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
            token_ids.data_ptr<int64_t>(),
            previous_token_ids.data_ptr<int64_t>(),
            valid_mask.data_ptr<bool>(),
            assignments.data_ptr<int64_t>(),
            token_ids.numel(),
            expert_count,
            top_k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return assignments;
}

std::vector<torch::Tensor> permute_topk_cuda(
    const torch::Tensor& context,
    const torch::Tensor& assignments,
    const torch::Tensor& valid_mask,
    int64_t expert_count) {
    c10::cuda::CUDAGuard device_guard(context.device());
    const int64_t token_count = context.size(0);
    const int64_t top_k = assignments.size(1);
    const int64_t pair_count = token_count * top_k;
    auto index_options = context.options().dtype(torch::kInt64);
    auto pair_keys = torch::empty({pair_count}, index_options);
    auto pair_values = torch::empty({pair_count}, index_options);
    auto sorted_keys = torch::empty_like(pair_keys);
    auto sorted_values = torch::empty_like(pair_values);
    auto counts = torch::zeros({expert_count + 1}, index_options);
    auto segment_offsets = torch::zeros({expert_count + 1}, index_options);
    auto sorted_rows = torch::empty_like(pair_values);
    auto sorted_context = torch::empty({pair_count, context.size(1)}, context.options());
    const auto stream = at::cuda::getDefaultCUDAStream(context.device().index());

    const unsigned int pair_blocks = static_cast<unsigned int>((pair_count + kThreads - 1) / kThreads);
    if (assignments.scalar_type() == torch::kInt) {
        build_pair_keys_kernel<int32_t><<<pair_blocks, kThreads, 0, stream>>>(
            assignments.data_ptr<int32_t>(),
            valid_mask.data_ptr<bool>(),
            pair_keys.data_ptr<int64_t>(),
            pair_values.data_ptr<int64_t>(),
            token_count,
            top_k,
            expert_count);
    } else {
        build_pair_keys_kernel<int64_t><<<pair_blocks, kThreads, 0, stream>>>(
            assignments.data_ptr<int64_t>(),
            valid_mask.data_ptr<bool>(),
            pair_keys.data_ptr<int64_t>(),
            pair_values.data_ptr<int64_t>(),
            token_count,
            top_k,
            expert_count);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    std::size_t sort_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_bytes,
        pair_keys.data_ptr<int64_t>(),
        sorted_keys.data_ptr<int64_t>(),
        pair_values.data_ptr<int64_t>(),
        sorted_values.data_ptr<int64_t>(),
        static_cast<int>(pair_count),
        0,
        64,
        stream));
    auto sort_workspace = torch::empty(
        {static_cast<int64_t>(sort_bytes)},
        context.options().dtype(torch::kUInt8));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_workspace.data_ptr(),
        sort_bytes,
        pair_keys.data_ptr<int64_t>(),
        sorted_keys.data_ptr<int64_t>(),
        pair_values.data_ptr<int64_t>(),
        sorted_values.data_ptr<int64_t>(),
        static_cast<int>(pair_count),
        0,
        64,
        stream));

    histogram_kernel<<<pair_blocks, kThreads, 0, stream>>>(
        sorted_keys.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(),
        pair_count,
        expert_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    std::size_t scan_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr,
        scan_bytes,
        counts.data_ptr<int64_t>(),
        segment_offsets.data_ptr<int64_t>() + 1,
        static_cast<int>(expert_count),
        stream));
    auto scan_workspace = torch::empty(
        {static_cast<int64_t>(scan_bytes)},
        context.options().dtype(torch::kUInt8));
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scan_workspace.data_ptr(),
        scan_bytes,
        counts.data_ptr<int64_t>(),
        segment_offsets.data_ptr<int64_t>() + 1,
        static_cast<int>(expert_count),
        stream));

    gather_sorted_rows_kernel<<<pair_blocks, kThreads, 0, stream>>>(
        sorted_values.data_ptr<int64_t>(),
        sorted_rows.data_ptr<int64_t>(),
        pair_count,
        top_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (context.scalar_type() == torch::kFloat32) {
        launch_gather_sorted_context<float>(context, sorted_values, segment_offsets, sorted_context, top_k, stream);
    } else if (context.scalar_type() == torch::kFloat16) {
        launch_gather_sorted_context<at::Half>(context, sorted_values, segment_offsets, sorted_context, top_k, stream);
    } else {
        launch_gather_sorted_context<at::BFloat16>(context, sorted_values, segment_offsets, sorted_context, top_k, stream);
    }
    return {sorted_context, sorted_rows, sorted_keys, segment_offsets};
}

torch::Tensor grouped_expert_mlp_cuda(
    const torch::Tensor& sorted_context,
    const torch::Tensor& sorted_experts,
    const torch::Tensor& segment_offsets,
    const torch::Tensor& gate_weights,
    const torch::Tensor& gate_biases,
    const torch::Tensor& value_weights,
    const torch::Tensor& value_biases,
    const torch::Tensor& output_weights,
    const torch::Tensor& output_biases) {
    c10::cuda::CUDAGuard device_guard(sorted_context.device());
    auto output = torch::zeros_like(sorted_context);
    if (sorted_context.size(0) == 0) {
        return output;
    }
    const auto stream = at::cuda::getDefaultCUDAStream(sorted_context.device().index());
    if (sorted_context.scalar_type() == torch::kFloat32) {
        launch_grouped_expert_mlp<float>(
            sorted_context,
            sorted_experts,
            segment_offsets,
            gate_weights,
            gate_biases,
            value_weights,
            value_biases,
            output_weights,
            output_biases,
            output,
            stream);
    } else if (sorted_context.scalar_type() == torch::kFloat16) {
        launch_grouped_expert_mlp<at::Half>(
            sorted_context,
            sorted_experts,
            segment_offsets,
            gate_weights,
            gate_biases,
            value_weights,
            value_biases,
            output_weights,
            output_biases,
            output,
            stream);
    } else {
        launch_grouped_expert_mlp<at::BFloat16>(
            sorted_context,
            sorted_experts,
            segment_offsets,
            gate_weights,
            gate_biases,
            value_weights,
            value_biases,
            output_weights,
            output_biases,
            output,
            stream);
    }
    return output;
}

torch::Tensor combine_topk_cuda(
    const torch::Tensor& expert_outputs,
    const torch::Tensor& sorted_rows,
    const torch::Tensor& segment_offsets,
    int64_t token_count,
    int64_t top_k) {
    c10::cuda::CUDAGuard device_guard(expert_outputs.device());
    auto combined = torch::zeros(
        {token_count, expert_outputs.size(1)},
        expert_outputs.options().dtype(torch::kFloat32));
    if (expert_outputs.size(0) == 0) {
        return combined.to(expert_outputs.scalar_type());
    }
    const auto stream = at::cuda::getDefaultCUDAStream(expert_outputs.device().index());
    if (expert_outputs.scalar_type() == torch::kFloat32) {
        launch_combine_topk<float>(expert_outputs, sorted_rows, segment_offsets, combined, token_count, top_k, stream);
    } else if (expert_outputs.scalar_type() == torch::kFloat16) {
        launch_combine_topk<at::Half>(expert_outputs, sorted_rows, segment_offsets, combined, token_count, top_k, stream);
    } else {
        launch_combine_topk<at::BFloat16>(expert_outputs, sorted_rows, segment_offsets, combined, token_count, top_k, stream);
    }
    return combined.to(expert_outputs.scalar_type());
}

}
