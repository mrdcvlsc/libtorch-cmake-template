#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "model.hpp"

int main(int argc, const char* argv[]) {
    auto model = std::make_shared<XorNet>();

    std::cout << "\n========================================================\n\n";

    for (const auto& param : model->named_parameters()) {
        std::cout << "Parameter Name: " << param.key() << std::endl;

        if (argc == 2 && std::string(argv[1]) == "show") {
            std::cout << "Parameter Value: \n" << param.value() << std::endl;
        }
    }

    std::cout << "\n========================================================\n\n";

    for (const auto& buffs : model->named_buffers()) {
        std::cout << "Buffers Name: " << buffs.key() << std::endl;
        
        if (argc == 2 && std::string(argv[1]) == "show") {
            std::cout << "Buffers Value: \n" << buffs.value() << std::endl;
        }
    }

    std::cout << "\n========================================================\n\n";

    for (const auto& param : model->named_modules()) {
        std::cout << "Modules Name: " << param.key() << std::endl;
    }

    std::cout << "\n========================================================\n\n";

    for (const auto& param : model->named_children()) {
        std::cout << "Children Name: " << param.key() << std::endl;
    }

    std::cout << "\n========================================================\n\n";

    return 0;
}