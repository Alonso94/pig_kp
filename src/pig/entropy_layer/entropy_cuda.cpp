#include <torch/extension.h>

torch::Tensor entropy_cuda_forward(torch::Tensor x, int region_size, float bandwidth);

torch::Tensor entropy_cuda_backward(torch::Tensor x, torch::Tensor d_entropy, int region_size, float bandwidth);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x);

torch::Tensor entropy_forward(torch::Tensor x, int region_size, float bandwidth) {
  CHECK_INPUT(x);
  return entropy_cuda_forward(x, region_size, bandwidth);
}

torch::Tensor entropy_backward(torch::Tensor x, torch::Tensor d_entropy, int region_size, float bandwidth) {
  CHECK_INPUT(x);
  CHECK_INPUT(d_entropy);
  return entropy_cuda_backward(x, d_entropy, region_size, bandwidth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &entropy_forward, "The forward pass of entropy layer");
  m.def("backward", &entropy_backward, "The backward pass of entropy layer");
}