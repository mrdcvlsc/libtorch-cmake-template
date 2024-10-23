#include "model.hpp"

MultiClassClassification::MultiClassClassification() : 
  conv_relu_stack(register_module("conv_relu_stack", torch::nn::Sequential(
    torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 4, /*kernel_size=*/3).stride(1).padding(1)),
    torch::nn::BatchNorm2d(4),
    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
    torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 4, /*kernel_size=*/3).stride(1).padding(1)),
    torch::nn::BatchNorm2d(4),
    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
    torch::nn::MaxPool2d(2)
  ))),
  
  flatten(register_module("flatten", torch::nn::Flatten())),

  linear_relu_stack(register_module("linear_relu_stack", torch::nn::Sequential(
    torch::nn::Linear(4 * 14 * 14, 128),
    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
    torch::nn::Linear(128, 128),
    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
    torch::nn::Linear(128, 10),
    torch::nn::Softmax(1)
  )))
{ }

torch::Tensor MultiClassClassification::forward(torch::Tensor x) {
  x = conv_relu_stack->forward(x);
  x = flatten->forward(x);
  x = linear_relu_stack->forward(x);
  return x;
}
