#ifndef LIBTORCH_CMAKE_TEMPLATE_UTILS_HPP
#define LIBTORCH_CMAKE_TEMPLATE_UTILS_HPP

#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

struct RawDataWriter {
    std::string directory;

    RawDataWriter() {
        // Check and create "raw_data_runs" directory
        fs::path base_dir = "raw_data_runs";
        if (!fs::exists(base_dir)) {
            fs::create_directory(base_dir);
        }

        // Create a timestamped folder inside "raw_data_runs"
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm buf;
        std::stringstream timestamp;

        // Use std::localtime for cross-platform compatibility
        if (std::tm* t = std::localtime(&in_time_t)) {
            buf = *t;
            timestamp << std::put_time(&buf, "%Y_%m_%d_%H_%M");
        } else {
            std::cerr << "Error: Failed to retrieve time.\n";
            return;
        }

        directory = (base_dir / timestamp.str()).string();
        fs::create_directory(directory);
    }

    bool addScalar(const std::string& tag, double scalar_value, int global_step) {
        auto now = std::chrono::system_clock::now();

        // Check if `tag` string is valid length
        if (tag.size() < 1 || tag.size() > 30) {
            std::cerr << "Error: Tag must be between 1 and 30 characters.\n";
            return false;
        }

        // Ensure valid filename for tag and check/create binary file
        fs::path file_path = fs::path(directory) / (tag + ".bin");
        bool is_new_file = !fs::exists(file_path);

        // Open file for read/write in binary mode, creating if necessary
        std::fstream file(file_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::app);
        if (!file) {
            std::cerr << "Error: Failed to open or create file.\n";
            return false;
        }

        // If file is new or empty, write the tag string in the first 30 bytes
        if (is_new_file || fs::file_size(file_path) == 0) {
            file.seekp(0);
            file.write(tag.c_str(), tag.size());
            file.write(std::string(30, '0').c_str(), 30 - tag.size()); // Pad to 30 bytes
        }

        // Append scalar_value and global_step to file
        file.seekp(0, std::ios::end);
        file.write(reinterpret_cast<const char*>(&scalar_value), sizeof(scalar_value));
        file.write(reinterpret_cast<const char*>(&global_step), sizeof(global_step));

        // Append current time in seconds as a floating-point value
        auto time_since_epoch_ns = now.time_since_epoch().count(); // nanoseconds
        double current_time_in_seconds = static_cast<double>(time_since_epoch_ns) / static_cast<double>(1e9);
        file.write(reinterpret_cast<const char*>(&current_time_in_seconds), sizeof(current_time_in_seconds));

        return true;
    }
};


#endif