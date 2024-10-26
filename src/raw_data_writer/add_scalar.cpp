#include "raw_data_writer.hpp"

bool RawDataWriter::add_scalar(const std::string& tag, double scalar_value, int global_step) {

    auto now = std::chrono::system_clock::now();

    // Ensure valid filename for tag and check/create binary file
    std::filesystem::path file_path = std::filesystem::path(directory) / (tag + ".scalar");
    bool is_new_file = !std::filesystem::exists(file_path);

    // Open file for read/write in binary mode, creating if necessary
    std::fstream file(file_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::app);
    if (!file) {
        std::cerr << "Error: Failed to open or create file.\n";
        return false;
    }

    // If file is new or empty
    if (is_new_file || std::filesystem::file_size(file_path) == 0) {
        file.seekp(0);

        // write the byte indicator that the file is for addScalar
        file.write("s", 1);

        // write the tag string in the binary file
        int tag_len = tag.size();
        file.write(reinterpret_cast<const char*>(&tag_len), sizeof(tag_len));
        file.write(tag.c_str(), tag.size());
    }

    // Append scalar_value and global_step to file
    file.seekp(0, std::ios::end);
    file.write(reinterpret_cast<const char*>(&scalar_value), sizeof(scalar_value));
    file.write(reinterpret_cast<const char*>(&global_step), sizeof(global_step));

    // Append current time in seconds as a floating-point value
    auto time_since_epoch_ns = now.time_since_epoch().count();
    double current_time_in_seconds = static_cast<double>(time_since_epoch_ns) / static_cast<double>(1e9);
    file.write(reinterpret_cast<const char*>(&current_time_in_seconds), sizeof(current_time_in_seconds));

    return true;
}