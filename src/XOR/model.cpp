#include "model.hpp"
#include "torch/nn/init.h"

XorNet::XorNet() {
  this->linear_stack = register_module("linear_stack",
    torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(2, 2).bias(true)),
      torch::nn::Sigmoid(),
      torch::nn::Linear(torch::nn::LinearOptions(2, 1).bias(true)),
      torch::nn::Sigmoid()
    )
  );

  initialize_parameters();
}

torch::Tensor XorNet::forward(torch::Tensor x) {
  x = this->linear_stack->forward(x);
  return x;
}

void XorNet::initialize_parameters() {
  torch::NoGradGuard noGrad;
  for (auto& module : this->linear_stack->children()) {
    if (auto* linear = module->as<torch::nn::Linear>()) {
      torch::nn::init::normal_(linear->weight, 0.5f, 1.f);
      torch::nn::init::constant_(linear->bias, 0.5f);
    }
  }
}