#include "image_reader.hpp"

namespace tj {

/// pixel format conversion can only happen here not in the write function.
JPEG::JPEG(const std::string &filename, const int pixel_format) {
    
    tjhandle jpeg_decompressor = tjInitDecompress();

    if (!jpeg_decompressor) {
        std::cerr << "Failed to initialize JPEG decompressor." << '\n';
        good = false;
        return;
    }

    // Read the JPEG file into a buffer
    std::ifstream jpeg_file(filename, std::ios::binary);
    
    if (!jpeg_file) {
        std::cerr << "Could not open file: " << filename << '\n';
        tjDestroy(jpeg_decompressor);
        good = false;
        return;
    }

    jpeg_file.seekg(0, std::ios::end);
    long jpeg_byte_size = jpeg_file.tellg();
    jpeg_file.seekg(0, std::ios::beg);

    std::vector<unsigned char> jpeg_read_buffer(jpeg_byte_size);
    jpeg_file.read(reinterpret_cast<char*>(jpeg_read_buffer.data()), jpeg_byte_size);
    jpeg_file.close();

    // Get image dimensions and subsampling type
    bool decompress_header_err = tjDecompressHeader2(
        jpeg_decompressor,
        jpeg_read_buffer.data(), jpeg_read_buffer.size(),
        &width, &height, &sub_sampling
    );
    
    if (decompress_header_err) {
        std::cerr << "Error decompressing JPEG header: " << tjGetErrorStr() << '\n';
        tjDestroy(jpeg_decompressor);
        good = false;
        return;
    }

    // Allocate buffer for decompressed image
    this->pixel_format = pixel_format;
    img_pixels.resize(width * height * tjPixelSize[pixel_format]);

    // is pixel format an output variable or we just need to pass its addresss for some fancy reasons?

    bool decompress_img_err = tjDecompress2(
        jpeg_decompressor,
        jpeg_read_buffer.data(), jpeg_read_buffer.size(),
        img_pixels.data(), width, /* pitch = */ 0, height, pixel_format,
        TJFLAG_FASTDCT
    );

    if (decompress_img_err) {
        std::cerr << "Error decompressing JPEG image: " << tjGetErrorStr() << '\n';
        tjDestroy(jpeg_decompressor);
        good = false;
        return;
    }

    tjDestroy(jpeg_decompressor);
    good = true;
}

std::string JPEG::getFilename() const {
    return this->filename;
}

int JPEG::getWidth() const {
    return this->width;
}

int JPEG::getHeight() const {
    return this->height;
}

int JPEG::getSubSampling() const {
    return this->sub_sampling;
}

int JPEG::getPixelFormat() const {
    return this->pixel_format;
}

int JPEG::isGood() const {
    return this->good;
}


} // namespace tj