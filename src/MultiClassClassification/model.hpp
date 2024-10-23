#ifndef MULTI_CLASS_CLASSIFICATION_MODEL_HPP
#define MULTI_CLASS_CLASSIFICATION_MODEL_HPP

#include <torch/torch.h>

struct MultiClassClassification : torch::nn::Module {
    MultiClassClassification();

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential conv_relu_stack{nullptr};
    torch::nn::Flatten flatten{nullptr};
    torch::nn::Sequential linear_relu_stack{nullptr};
};

#endif // MULTI_CLASS_CLASSIFICATION_MODEL_HPP
