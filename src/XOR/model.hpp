#ifndef XOR_MODEL_HPP
#define XOR_MODEL_HPP

#include "torch/nn/modules/activation.h"
#include <memory>
#include <torch/torch.h>

struct ActivatedLinearReLU: torch::nn::Module {
  ActivatedLinearReLU();
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Sequential layer{nullptr};
};

struct ActivatedLinearSigmoid: torch::nn::Module {
  ActivatedLinearSigmoid();
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Linear layer{nullptr};
  torch::nn::Sigmoid activation{nullptr};
};

struct XorNet: torch::nn::Module {
  XorNet();
  torch::Tensor forward(torch::Tensor x);

  torch::nn::Sequential layer0{nullptr};
  std::shared_ptr<ActivatedLinearReLU> layer1{nullptr};
  std::shared_ptr<ActivatedLinearSigmoid> layer2{nullptr};
};

#endif // XOR_MODEL_HPP