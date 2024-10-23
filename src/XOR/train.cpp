#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "model.hpp"

int main() {
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

  XorNet model;
  model.to(device);
  
  std::cout << "Weights BEFORE training:\n";
  for (const auto& pair : model.named_parameters()) {
    std::cout << pair.key() << ":\n" << pair.value() << "\n\n";
  }

  torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.55f));
  torch::nn::MSELoss loss_fn;
  
  torch::Tensor inputs = torch::tensor({
    {0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}
  });

  torch::Tensor labels = torch::tensor({{0.f}, {1.f}, {1.f}, {0.f}});

  inputs = inputs.to(device).to(torch::kFloat32);
  labels = labels.to(device).to(torch::kFloat32);

  std::cout << "inputs  : \n" << inputs << "\n";
  std::cout << "labels  : \n" << labels << "\n";

  auto initial_output = model.forward(inputs);
  auto initial_loss = loss_fn(initial_output, labels);

  std::cout << "\nBefore Training:\n";
  std::cout << "output  : \n" << initial_output << "\n";
  std::cout << "loss    : \n" << initial_loss   << "\n";

  model.train();

  const size_t MAX_EPOCH = 50'000;
  size_t epoch = 0;

  while (epoch < MAX_EPOCH) {
    optimizer.zero_grad();

    auto training_output = model.forward(inputs);
    auto training_loss = loss_fn(training_output, labels);

    training_loss.backward();
    optimizer.step();

    // log loss every 500 epochs
    if (epoch % 500 == 500 - 1) {
      std::cout << "training loss (epoch[" << epoch + 1 << "]) : " << training_loss.item().toFloat() << "\n";
    }

    if (training_loss.item().toFloat() < 0.005f) {
      std::cout << "TRAINING DONE LOSS BELOW < 0.005f | epochs: " << epoch + 1 << "\n";
      break;
    }

    epoch++;
  }

  if (epoch == MAX_EPOCH) {
    std::cout << "MAX EPOCH ACHIEVED\n";
  }

  model.eval();

  std::cout << "\n\nWeights AFTER training:\n";
  for (const auto& pair : model.named_parameters()) {
    std::cout << pair.key() << ":\n" << pair.value() << "\n\n";
  }

  auto trained_output = model.forward(inputs);
  auto trained_loss = loss_fn(trained_output, labels);

  std::cout << "\nAfter Training:\n";
  std::cout << "output    : \n" << trained_output << "\n";
  std::cout << "loss start: " << initial_loss.item().toFloat() << "\n";
  std::cout << "loss end  : " << trained_loss.item().toFloat() << "\n";

  std::cout << "Total Training Epochs Done : " << epoch << "\n";

  return 0;
} 