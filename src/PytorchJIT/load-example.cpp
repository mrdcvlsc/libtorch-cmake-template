#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module model;

    try {
        model = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    for (const auto& param : model.named_parameters()) {
        std::cout << "Parameter Name: " << param.name << std::endl;
        std::cout << "Parameter Value: " << param.value << std::endl;
    }

    for (const auto& param : model.named_buffers()) {
        std::cout << "Buffers Name: " << param.name << std::endl;
        std::cout << "Buffers Value: " << param.value << std::endl;
    }

    for (const auto& param : model.named_modules()) {
        std::cout << "Modules Name: " << param.name << std::endl;
    }

    for (const auto& param : model.named_children()) {
        std::cout << "Children Name: " << param.name << std::endl;
    }

    return 0;
}