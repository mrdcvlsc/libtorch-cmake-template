#ifndef LIBTORCH_CMAKE_TEMPLATE_UTILS_HPP
#define LIBTORCH_CMAKE_TEMPLATE_UTILS_HPP

#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <map>
#include <string>

struct RawDataWriter {
    std::string directory;

    RawDataWriter();

    bool add_scalar(const std::string& tag, double scalar_value, int global_step);
    bool add_scalars(const std::string& tag, const std::map<std::string, double>& scalars, int global_step);
};

#endif