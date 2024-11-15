#include "image_reader.hpp"
#include "torch/types.h"
#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

namespace tj {

torch::Tensor JPEG::to_tensor() {

    auto options = torch::TensorOptions().dtype(torch::kUInt8);

    auto pixel_tensor = torch::from_blob(
        img_pixels.data(),
        {width, height, tjPixelSize[pixel_format]},
        options
    ).permute({2, 0, 1});

    return pixel_tensor;
}

} // namespace tj