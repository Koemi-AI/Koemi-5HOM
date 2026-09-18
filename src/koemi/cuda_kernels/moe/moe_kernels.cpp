#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace koemi_cuda_moe {

torch::Tensor route_topk_cuda(
    const torch::Tensor& token_ids,
    const torch::Tensor& previous_token_ids,
    const torch::Tensor& valid_mask,
    int64_t expert_count,
    int64_t top_k);

std::vector<torch::Tensor> permute_topk_cuda(
    const torch::Tensor& context,
    const torch::Tensor& assignments,
    const torch::Tensor& valid_mask,
    int64_t expert_count);

torch::Tensor grouped_expert_mlp_cuda(
    const torch::Tensor& sorted_context,
    const torch::Tensor& sorted_experts,
    const torch::Tensor& segment_offsets,
    const torch::Tensor& gate_weights,
    const torch::Tensor& gate_biases,
    const torch::Tensor& value_weights,
    const torch::Tensor& value_biases,
    const torch::Tensor& output_weights,
    const torch::Tensor& output_biases);

torch::Tensor combine_topk_cuda(
    const torch::Tensor& expert_outputs,
    const torch::Tensor& sorted_rows,
    const torch::Tensor& segment_offsets,
    int64_t token_count,
    int64_t top_k);

namespace {

constexpr int64_t kMaximumExpertCount = 1 << 20;
constexpr int64_t kMaximumTopK = 256;
constexpr int64_t kMaximumLaunchItems = std::numeric_limits<unsigned int>::max();

void validate_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void validate_same_device(const std::vector<std::pair<const torch::Tensor*, const char*>>& tensors) {
    const auto device = tensors.front().first->device();
    for (const auto& [tensor, name] : tensors) {
        TORCH_CHECK(tensor->device() == device, name, " must use ", device);
    }
}

void validate_positive_bound(int64_t value, const char* name, int64_t limit) {
    TORCH_CHECK(value > 0, name, " must be positive");
    TORCH_CHECK(value <= limit, name, " exceeds the CUDA MoE safety limit of ", limit);
}

void validate_launch_items(int64_t items, const char* name) {
    TORCH_CHECK(items >= 0 && items <= kMaximumLaunchItems, name, " exceeds one CUDA launch");
}

void validate_float_dtype(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat16 ||
            tensor.scalar_type() == torch::kBFloat16,
        name,
        " must use float32, float16, or bfloat16");
}

void validate_route_inputs(
    const torch::Tensor& token_ids,
    const torch::Tensor& previous_token_ids,
    const torch::Tensor& valid_mask,
    int64_t expert_count,
    int64_t top_k) {
    validate_cuda_tensor(token_ids, "token_ids");
    validate_cuda_tensor(previous_token_ids, "previous_token_ids");
    validate_cuda_tensor(valid_mask, "valid_mask");
    validate_same_device({{&token_ids, "token_ids"}, {&previous_token_ids, "previous_token_ids"}, {&valid_mask, "valid_mask"}});
    TORCH_CHECK(
        token_ids.scalar_type() == torch::kInt || token_ids.scalar_type() == torch::kLong,
        "token_ids must use int32 or int64");
    TORCH_CHECK(previous_token_ids.scalar_type() == token_ids.scalar_type(), "token_ids and previous_token_ids must use one dtype");
    TORCH_CHECK(valid_mask.scalar_type() == torch::kBool, "valid_mask must use boolean dtype");
    TORCH_CHECK(token_ids.sizes() == previous_token_ids.sizes(), "routing tensors must have equal shapes");
    TORCH_CHECK(token_ids.sizes() == valid_mask.sizes(), "routing tensors must have equal shapes");
    validate_positive_bound(expert_count, "expert_count", kMaximumExpertCount);
    validate_positive_bound(top_k, "top_k", kMaximumTopK);
    TORCH_CHECK(top_k <= expert_count, "top_k must not exceed expert_count");
    validate_launch_items(token_ids.numel(), "token_ids");
}

void validate_permute_inputs(
    const torch::Tensor& context,
    const torch::Tensor& assignments,
    const torch::Tensor& valid_mask,
    int64_t expert_count) {
    validate_cuda_tensor(context, "context");
    validate_cuda_tensor(assignments, "assignments");
    validate_cuda_tensor(valid_mask, "valid_mask");
    validate_same_device({{&context, "context"}, {&assignments, "assignments"}, {&valid_mask, "valid_mask"}});
    validate_float_dtype(context, "context");
    TORCH_CHECK(context.dim() == 2, "context must have shape [token_count, width]");
    TORCH_CHECK(context.size(0) > 0 && context.size(1) > 0, "context dimensions must be positive");
    TORCH_CHECK(assignments.dim() == 2, "assignments must have shape [token_count, top_k]");
    TORCH_CHECK(assignments.scalar_type() == torch::kInt || assignments.scalar_type() == torch::kLong, "assignments must use int32 or int64");
    TORCH_CHECK(assignments.size(0) == context.size(0), "assignments token count must match context");
    TORCH_CHECK(assignments.size(1) > 0 && assignments.size(1) <= kMaximumTopK, "assignments top_k is outside the CUDA MoE limit");
    TORCH_CHECK(valid_mask.dim() == 1 && valid_mask.scalar_type() == torch::kBool, "valid_mask must be a one-dimensional boolean tensor");
    TORCH_CHECK(valid_mask.size(0) == context.size(0), "valid_mask token count must match context");
    validate_positive_bound(expert_count, "expert_count", kMaximumExpertCount);
    TORCH_CHECK(assignments.size(1) <= expert_count, "top_k must not exceed expert_count");
    validate_launch_items(assignments.numel(), "assignments");
}

void validate_grouped_inputs(
    const torch::Tensor& sorted_context,
    const torch::Tensor& sorted_experts,
    const torch::Tensor& segment_offsets,
    const torch::Tensor& gate_weights,
    const torch::Tensor& gate_biases,
    const torch::Tensor& value_weights,
    const torch::Tensor& value_biases,
    const torch::Tensor& output_weights,
    const torch::Tensor& output_biases) {
    const std::vector<std::pair<const torch::Tensor*, const char*>> tensors = {
        {&sorted_context, "sorted_context"},
        {&sorted_experts, "sorted_experts"},
        {&segment_offsets, "segment_offsets"},
        {&gate_weights, "gate_weights"},
        {&gate_biases, "gate_biases"},
        {&value_weights, "value_weights"},
        {&value_biases, "value_biases"},
        {&output_weights, "output_weights"},
        {&output_biases, "output_biases"},
    };
    for (const auto& [tensor, name] : tensors) {
        validate_cuda_tensor(*tensor, name);
    }
    validate_same_device(tensors);
    validate_float_dtype(sorted_context, "sorted_context");
    TORCH_CHECK(sorted_context.dim() == 2, "sorted_context must have shape [pair_count, width]");
    TORCH_CHECK(sorted_context.size(1) > 0, "sorted_context width must be positive");
    TORCH_CHECK(sorted_experts.dim() == 1 && sorted_experts.scalar_type() == torch::kLong, "sorted_experts must be one-dimensional int64");
    TORCH_CHECK(sorted_experts.size(0) == sorted_context.size(0), "sorted_experts must match pair_count");
    TORCH_CHECK(segment_offsets.dim() == 1 && segment_offsets.scalar_type() == torch::kLong, "segment_offsets must be one-dimensional int64");
    TORCH_CHECK(gate_weights.dim() == 3 && value_weights.dim() == 3 && output_weights.dim() == 3, "expert weights must be three-dimensional");
    TORCH_CHECK(gate_weights.size(0) > 0 && gate_weights.size(1) > 0, "expert and hidden dimensions must be positive");
    TORCH_CHECK(gate_weights.size(2) == sorted_context.size(1), "gate weight width must match sorted_context");
    TORCH_CHECK(value_weights.sizes() == gate_weights.sizes(), "value weights must match gate weights");
    TORCH_CHECK(output_weights.size(0) == gate_weights.size(0), "output weights expert count must match gate weights");
    TORCH_CHECK(output_weights.size(1) == sorted_context.size(1), "output weights width must match sorted_context");
    TORCH_CHECK(output_weights.size(2) == gate_weights.size(1), "output weights hidden width must match gate weights");
    TORCH_CHECK(gate_biases.sizes() == torch::IntArrayRef({gate_weights.size(0), gate_weights.size(1)}), "gate biases do not match gate weights");
    TORCH_CHECK(value_biases.sizes() == gate_biases.sizes(), "value biases do not match gate biases");
    TORCH_CHECK(output_biases.sizes() == torch::IntArrayRef({gate_weights.size(0), sorted_context.size(1)}), "output biases do not match output weights");
    TORCH_CHECK(segment_offsets.size(0) == gate_weights.size(0) + 1, "segment_offsets must have expert_count + 1 entries");
    for (const auto& [tensor, name] : tensors) {
        if (tensor != &sorted_experts && tensor != &segment_offsets) {
            TORCH_CHECK(tensor->scalar_type() == sorted_context.scalar_type(), name, " must match sorted_context dtype");
        }
    }
    validate_launch_items(sorted_context.size(0), "sorted_context");
}

void validate_combine_inputs(
    const torch::Tensor& expert_outputs,
    const torch::Tensor& sorted_rows,
    const torch::Tensor& segment_offsets,
    int64_t token_count,
    int64_t top_k) {
    validate_cuda_tensor(expert_outputs, "expert_outputs");
    validate_cuda_tensor(sorted_rows, "sorted_rows");
    validate_cuda_tensor(segment_offsets, "segment_offsets");
    validate_same_device({{&expert_outputs, "expert_outputs"}, {&sorted_rows, "sorted_rows"}, {&segment_offsets, "segment_offsets"}});
    validate_float_dtype(expert_outputs, "expert_outputs");
    TORCH_CHECK(expert_outputs.dim() == 2 && expert_outputs.size(1) > 0, "expert_outputs must have shape [pair_count, width]");
    TORCH_CHECK(sorted_rows.dim() == 1 && sorted_rows.scalar_type() == torch::kLong, "sorted_rows must be one-dimensional int64");
    TORCH_CHECK(sorted_rows.size(0) == expert_outputs.size(0), "sorted_rows must match pair_count");
    TORCH_CHECK(segment_offsets.dim() == 1 && segment_offsets.scalar_type() == torch::kLong && segment_offsets.size(0) >= 2, "segment_offsets must have at least two int64 entries");
    validate_positive_bound(token_count, "token_count", kMaximumLaunchItems);
    validate_positive_bound(top_k, "top_k", kMaximumTopK);
    validate_launch_items(expert_outputs.size(0), "expert_outputs");
}

}

torch::Tensor route_topk(
    const torch::Tensor& token_ids,
    const torch::Tensor& previous_token_ids,
    const torch::Tensor& valid_mask,
    int64_t expert_count,
    int64_t top_k) {
    validate_route_inputs(token_ids, previous_token_ids, valid_mask, expert_count, top_k);
    return route_topk_cuda(token_ids, previous_token_ids, valid_mask, expert_count, top_k);
}

std::vector<torch::Tensor> permute_topk(
    const torch::Tensor& context,
    const torch::Tensor& assignments,
    const torch::Tensor& valid_mask,
    int64_t expert_count) {
    validate_permute_inputs(context, assignments, valid_mask, expert_count);
    return permute_topk_cuda(context, assignments, valid_mask, expert_count);
}

torch::Tensor grouped_expert_mlp(
    const torch::Tensor& sorted_context,
    const torch::Tensor& sorted_experts,
    const torch::Tensor& segment_offsets,
    const torch::Tensor& gate_weights,
    const torch::Tensor& gate_biases,
    const torch::Tensor& value_weights,
    const torch::Tensor& value_biases,
    const torch::Tensor& output_weights,
    const torch::Tensor& output_biases) {
    validate_grouped_inputs(
        sorted_context,
        sorted_experts,
        segment_offsets,
        gate_weights,
        gate_biases,
        value_weights,
        value_biases,
        output_weights,
        output_biases);
    return grouped_expert_mlp_cuda(
        sorted_context,
        sorted_experts,
        segment_offsets,
        gate_weights,
        gate_biases,
        value_weights,
        value_biases,
        output_weights,
        output_biases);
}

torch::Tensor combine_topk(
    const torch::Tensor& expert_outputs,
    const torch::Tensor& sorted_rows,
    const torch::Tensor& segment_offsets,
    int64_t token_count,
    int64_t top_k) {
    validate_combine_inputs(expert_outputs, sorted_rows, segment_offsets, token_count, top_k);
    return combine_topk_cuda(expert_outputs, sorted_rows, segment_offsets, token_count, top_k);
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("KOEMI_CUDA_ABI_VERSION") = 1;
    module.def("route_topk", &koemi_cuda_moe::route_topk, "HERM content route top-k CUDA");
    module.def("permute_topk", &koemi_cuda_moe::permute_topk, "Stable MoE token permutation CUDA");
    module.def("grouped_expert_mlp", &koemi_cuda_moe::grouped_expert_mlp, "Inference-only grouped expert MLP CUDA");
    module.def("combine_topk", &koemi_cuda_moe::combine_topk, "MoE top-k combine CUDA");
}
