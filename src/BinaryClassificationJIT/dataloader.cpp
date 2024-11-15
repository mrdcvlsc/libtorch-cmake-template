#include <filesystem>
#include <iostream>
#include <torch/torch.h>
#include "../ImageReader/image_reader.hpp"

int main() {
    std::cout << "Reading Image...\n";
    
    tj::JPEG img("test.jpg", tj::TJPF_RGB);

    if (!img.isGood()) {
        std::cout << "Error in reading the image\n";
        return 1;
    } else {
        std::cout << "Reading Image Successful\n";
    }

    std::cout << "JPEG : \n";
    std::cout << "filename : " << img.getFilename() << "\n";
    std::cout << "width    : " << img.getWidth() << "\n";
    std::cout << "height   : " << img.getHeight() << "\n";
    std::cout << "channels : " << tj::tjPixelSize[img.getPixelFormat()] << "\n";

    std::cout << "Converting Image Pixel Values To Tensor...\n";
    auto pixel = img.to_tensor();
    std::cout << "Successful In Converting Image Pixel Values To Tensor\n";

    std::cout << "pixel value:\n" << pixel << "\n\n";

    return 0;
}