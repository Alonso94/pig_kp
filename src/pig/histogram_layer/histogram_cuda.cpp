#include <torch/extension.h>

torch::Tensor histogram_cuda_forward(torch::Tensor x, float bandwidth);

torch::Tensor histogram_cuda_backward(torch::Tensor x, torch::Tensor d_histogram, float bandwidth);


torch::Tensor histogram_forward(torch::Tensor x, float bandwidth) {
  return histogram_cuda_forward(x, bandwidth);
}

torch::Tensor histogram_backward(torch::Tensor x, torch::Tensor d_histogram, float bandwidth) {
  return histogram_cuda_backward(x, d_histogram, bandwidth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &histogram_forward, "The forward pass of histogram layer");
  m.def("backward", &histogram_backward, "The backward pass of histogram layer");
}