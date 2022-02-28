#include <torch/extension.h>

torch::Tensor joint_entropy_cuda_forward(torch::Tensor x, float bandwidth);

torch::Tensor joint_entropy_cuda_backward(torch::Tensor x, torch::Tensor d_joint_entropy, float bandwidth);


torch::Tensor joint_entropy_forward(torch::Tensor x, float bandwidth) {
  return joint_entropy_cuda_forward(x, bandwidth);
}

torch::Tensor joint_entropy_backward(torch::Tensor x, torch::Tensor d_joint_entropy, float bandwidth) {
  return joint_entropy_cuda_backward(x, d_joint_entropy, bandwidth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &joint_entropy_forward, "The forward pass of joint_entropy layer");
  m.def("backward", &joint_entropy_backward, "The backward pass of joint_entropy layer");
}