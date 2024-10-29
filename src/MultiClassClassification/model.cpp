#include "model.hpp"

using Conv2d = torch::nn::Conv2d;
using Conv2dOptions = torch::nn::Conv2dOptions;

using BatchNorm2d = torch::nn::BatchNorm2d;
using BatchNorm2dOptions = torch::nn::BatchNorm2dOptions;

using MaxPool2d = torch::nn::MaxPool2d;
using MaxPool2dOptions = torch::nn::MaxPool2dOptions;

MultiClassClassification::MultiClassClassification() :
  conv_relu_stack(register_module("conv_relu_stack", torch::nn::Sequential(
    Conv2d(Conv2dOptions(1, 4, {3, 3}).stride({1, 1}).padding({1, 1}).bias(true)),
    BatchNorm2d(BatchNorm2dOptions(4).eps(1e-05).momentum(0.1).affine(true).track_running_stats(true)),
    torch::nn::ReLU(torch::nn::ReLUOptions(true)),

    Conv2d(Conv2dOptions(4, 4, {3, 3}).stride({1, 1}).padding({1, 1}).bias(true)),
    BatchNorm2d(BatchNorm2dOptions(4).eps(1e-05).momentum(0.1).affine(true).track_running_stats(true)),
    torch::nn::ReLU(torch::nn::ReLUOptions(true)),
    
    MaxPool2d(MaxPool2dOptions({2, 2}).stride({2, 2}).padding({0, 0}).dilation(1).ceil_mode(false))
  ))),
  
  flatten(register_module("flatten", torch::nn::Flatten())),

  linear_relu_stack(register_module("linear_relu_stack", torch::nn::Sequential(
    torch::nn::Linear(4 * 14 * 14, 128),
    torch::nn::ReLU(torch::nn::ReLUOptions(true)),
    torch::nn::Linear(128, 128),
    torch::nn::ReLU(torch::nn::ReLUOptions(true)),
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
