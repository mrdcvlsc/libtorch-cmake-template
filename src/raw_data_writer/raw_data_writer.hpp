#ifndef LIBTORCH_CMAKE_TEMPLATE_UTILS_HPP
#define LIBTORCH_CMAKE_TEMPLATE_UTILS_HPP

#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

struct RawDataWriter {
    std::string directory;

    RawDataWriter();

    bool addScalar(const std::string& tag, double scalar_value, int global_step);
};

#endif