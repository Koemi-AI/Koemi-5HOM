#include <torch/extension.h>

#include <cmath>
#include <vector>

namespace koemi_cuda_core {

std::vector<torch::Tensor> rmsnorm_silu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps);

std::vector<torch::Tensor> rmsnorm_silu_backward_cuda(
    torch::Tensor gradient_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inverse_rms,
    double eps);

void validate_forward_inputs(const torch::Tensor& input, const torch::Tensor& weight, double eps) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must have shape [rows, hidden]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden]");
    TORCH_CHECK(input.size(1) > 0, "input hidden dimension must be positive");
    TORCH_CHECK(weight.size(0) == input.size(1), "weight hidden dimension must match input");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16 ||
                    input.scalar_type() == torch::kFloat32,
                "input must use float16, bfloat16 or float32");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "input and weight must have the same dtype");
    TORCH_CHECK(input.device() == weight.device(), "input and weight must use the same CUDA device");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
}

void validate_backward_inputs(
    const torch::Tensor& gradient_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& inverse_rms,
    double eps) {
    validate_forward_inputs(input, weight, eps);
    TORCH_CHECK(gradient_output.is_cuda(), "gradient_output must be a CUDA tensor");
    TORCH_CHECK(gradient_output.is_contiguous(), "gradient_output must be contiguous");
    TORCH_CHECK(gradient_output.sizes() == input.sizes(), "gradient_output must match input shape");
    TORCH_CHECK(gradient_output.scalar_type() == input.scalar_type(), "gradient_output must match input dtype");
    TORCH_CHECK(gradient_output.device() == input.device(), "gradient_output must use the input CUDA device");
    TORCH_CHECK(inverse_rms.is_cuda(), "inverse_rms must be a CUDA tensor");
    TORCH_CHECK(inverse_rms.is_contiguous(), "inverse_rms must be contiguous");
    TORCH_CHECK(inverse_rms.scalar_type() == torch::kFloat32, "inverse_rms must use float32");
    TORCH_CHECK(inverse_rms.dim() == 1 && inverse_rms.size(0) == input.size(0), "inverse_rms must have shape [rows]");
    TORCH_CHECK(inverse_rms.device() == input.device(), "inverse_rms must use the input CUDA device");
}

std::vector<torch::Tensor> rmsnorm_silu_forward(torch::Tensor input, torch::Tensor weight, double eps) {
    validate_forward_inputs(input, weight, eps);
    return rmsnorm_silu_forward_cuda(input, weight, eps);
}

std::vector<torch::Tensor> rmsnorm_silu_backward(
    torch::Tensor gradient_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inverse_rms,
    double eps) {
    validate_backward_inputs(gradient_output, input, weight, inverse_rms, eps);
    return rmsnorm_silu_backward_cuda(gradient_output, input, weight, inverse_rms, eps);
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("KOEMI_CUDA_ABI_VERSION") = 1;
    module.def("rmsnorm_silu_forward", &koemi_cuda_core::rmsnorm_silu_forward, "RMSNorm+SiLU CUDA forward");
    module.def("rmsnorm_silu_backward", &koemi_cuda_core::rmsnorm_silu_backward, "RMSNorm+SiLU CUDA backward");
}
