#ifndef XOR_MODEL_HPP
#define XOR_MODEL_HPP

#include <torch/torch.h>

struct XorNet: torch::nn::Module {
  XorNet();
  torch::Tensor forward(torch::Tensor x);
  void initialize_parameters();

  torch::nn::Sequential linear_stack{nullptr};
};

#endif // XOR_MODEL_HPP