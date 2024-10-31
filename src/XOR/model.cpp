#include "model.hpp"
#include <memory>

ActivatedLinearReLU::ActivatedLinearReLU() {
  this->layer = register_module("layer",
    torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(2, 2).bias(true)),
      torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(2)),
      torch::nn::ReLU(torch::nn::ReLUOptions(true))
    )
  );
}

torch::Tensor ActivatedLinearReLU::forward(torch::Tensor x) {
  x = this->layer->forward(x);
  return x;
}

ActivatedLinearSigmoid::ActivatedLinearSigmoid() {
  this->layer = register_module("layer",
    torch::nn::Linear(torch::nn::LinearOptions(2, 1).bias(true))
  );

  this->activation = register_module("activation", torch::nn::Sigmoid());
}

torch::Tensor ActivatedLinearSigmoid::forward(torch::Tensor x) {
  x = this->layer->forward(x);
  x = this->activation->forward(x);
  return x;
}

XorNet::XorNet() {
  this->layer0 = register_module("layer0",
    torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(2, 2).bias(true)),
      torch::nn::Sigmoid()
    )
  );
  this->layer1 = register_module("layer1", std::make_shared<ActivatedLinearReLU>());
  this->layer2 = register_module("layer2", std::make_shared<ActivatedLinearSigmoid>());
}

torch::Tensor XorNet::forward(torch::Tensor x) {
  x = this->layer0->forward(x);
  x = this->layer1->forward(x);
  x = this->layer2->forward(x);
  return x;
}