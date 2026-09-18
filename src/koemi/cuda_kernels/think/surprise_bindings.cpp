#include <torch/extension.h>

#include <vector>

torch::Tensor causal_surprise_forward_cuda(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask);

std::vector<torch::Tensor> causal_surprise_backward_cuda(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask,
    const torch::Tensor& gradient);

void validate_common_inputs(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask) {
    TORCH_CHECK(prior_states.is_cuda(), "prior_states must be a CUDA tensor");
    TORCH_CHECK(content_weight.is_cuda(), "content_weight must be a CUDA tensor");
    TORCH_CHECK(content_bias.is_cuda(), "content_bias must be a CUDA tensor");
    TORCH_CHECK(target_ids.is_cuda(), "target_ids must be a CUDA tensor");
    TORCH_CHECK(valid_mask.is_cuda(), "valid_mask must be a CUDA tensor");
    TORCH_CHECK(prior_states.dim() == 3, "prior_states must have rank 3");
    TORCH_CHECK(content_weight.dim() == 2, "content_weight must have rank 2");
    TORCH_CHECK(content_bias.dim() == 1, "content_bias must have rank 1");
    TORCH_CHECK(target_ids.dim() == 2, "target_ids must have rank 2");
    TORCH_CHECK(valid_mask.dim() == 2, "valid_mask must have rank 2");
    TORCH_CHECK(prior_states.is_contiguous(), "prior_states must be contiguous");
    TORCH_CHECK(content_weight.is_contiguous(), "content_weight must be contiguous");
    TORCH_CHECK(content_bias.is_contiguous(), "content_bias must be contiguous");
    TORCH_CHECK(target_ids.is_contiguous(), "target_ids must be contiguous");
    TORCH_CHECK(valid_mask.is_contiguous(), "valid_mask must be contiguous");
    TORCH_CHECK(prior_states.device() == content_weight.device(), "all tensors must use one CUDA device");
    TORCH_CHECK(prior_states.device() == content_bias.device(), "all tensors must use one CUDA device");
    TORCH_CHECK(prior_states.device() == target_ids.device(), "all tensors must use one CUDA device");
    TORCH_CHECK(prior_states.device() == valid_mask.device(), "all tensors must use one CUDA device");
    TORCH_CHECK(prior_states.scalar_type() == content_weight.scalar_type(), "head and state dtypes must match");
    TORCH_CHECK(prior_states.scalar_type() == content_bias.scalar_type(), "head and state dtypes must match");
    TORCH_CHECK(target_ids.scalar_type() == torch::kLong, "target_ids must use torch.int64");
    TORCH_CHECK(valid_mask.scalar_type() == torch::kBool, "valid_mask must use torch.bool");
    TORCH_CHECK(target_ids.size(0) == prior_states.size(0), "target_ids batch must match prior_states");
    TORCH_CHECK(target_ids.size(1) == prior_states.size(1), "target_ids sequence must match prior_states");
    TORCH_CHECK(valid_mask.size(0) == prior_states.size(0), "valid_mask batch must match prior_states");
    TORCH_CHECK(valid_mask.size(1) == prior_states.size(1), "valid_mask sequence must match prior_states");
    TORCH_CHECK(content_weight.size(1) == prior_states.size(2), "head width must match state width");
    TORCH_CHECK(content_bias.size(0) == content_weight.size(0), "bias size must match vocabulary size");
    TORCH_CHECK(prior_states.size(2) <= 4096, "state width must be at most 4096");
}

torch::Tensor forward(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask) {
    validate_common_inputs(prior_states, content_weight, content_bias, target_ids, valid_mask);
    return causal_surprise_forward_cuda(
        prior_states, content_weight, content_bias, target_ids, valid_mask);
}

std::vector<torch::Tensor> backward(
    const torch::Tensor& prior_states,
    const torch::Tensor& content_weight,
    const torch::Tensor& content_bias,
    const torch::Tensor& target_ids,
    const torch::Tensor& valid_mask,
    const torch::Tensor& gradient) {
    validate_common_inputs(prior_states, content_weight, content_bias, target_ids, valid_mask);
    TORCH_CHECK(gradient.is_cuda(), "gradient must be a CUDA tensor");
    TORCH_CHECK(gradient.is_contiguous(), "gradient must be contiguous");
    TORCH_CHECK(gradient.scalar_type() == prior_states.scalar_type(), "gradient dtype must match output dtype");
    TORCH_CHECK(gradient.sizes() == target_ids.sizes(), "gradient shape must match target_ids");
    TORCH_CHECK(gradient.device() == prior_states.device(), "gradient must use the input CUDA device");
    return causal_surprise_backward_cuda(
        prior_states, content_weight, content_bias, target_ids, valid_mask, gradient);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &forward, "Causal surprise forward");
    module.def("backward", &backward, "Causal surprise backward");
}
