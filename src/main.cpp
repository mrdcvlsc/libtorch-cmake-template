#include <ATen/ops/sigmoid.h>
#include <iostream>
#include <memory>

#include <torch/torch.h>

struct XorNet: torch::nn::Module {
  XorNet() {
    fc1 = register_module("fc1", torch::nn::Linear(2, 2));
    fc2 = register_module("fc2", torch::nn::Linear(2, 1));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(fc1->forward(x));
    x = torch::sigmoid(fc2->forward(x));
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
  auto xor_net = std::make_shared<XorNet>();

  std::cout << "xor_net : \n" << xor_net << "\n\n";
}