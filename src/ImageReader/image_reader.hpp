#ifndef IMAGE_READER_HPP
#define IMAGE_READER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <torch/torch.h>

/** libjpeg-turbo - namespace that contains C++ wrapper functions for basic jpeg read and write opreations */
namespace tj {

extern "C"
{
#include "../turbojpeg/include/turbojpeg.h"
}

class JPEG {

    private:

    int width;
    int height;
    int sub_sampling;
    int pixel_format;
    int good;
    std::vector<unsigned char> img_pixels;
    std::string filename;

    public:

    std::string getFilename() const;
    int getWidth() const;
    int getHeight() const;
    int getSubSampling() const;
    int getPixelFormat() const;
    int isGood() const;

    JPEG(const std::string &filename, const int pixel_format);
    torch::Tensor to_tensor();
};

} // namespace tj

#endif // IMAGE_READER_HPP