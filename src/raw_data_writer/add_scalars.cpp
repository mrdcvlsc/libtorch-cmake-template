#include "raw_data_writer.hpp"

bool RawDataWriter::add_scalars(const std::string& tag, const std::map<std::string, double>& scalars, int global_step) {

    auto now = std::chrono::system_clock::now();

    // Ensure valid filename for tag and check/create binary file
    std::filesystem::path file_path = std::filesystem::path(directory) / (tag + ".scalars");
    bool is_new_file = !std::filesystem::exists(file_path);

    // Open file for read/write in binary mode, creating if necessary
    std::fstream file(file_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::app);
    if (!file) {
        std::cerr << "Error: Failed to open or create file.\n";
        return false;
    }

    if (is_new_file || std::filesystem::file_size(file_path) == 0) {
        file.seekp(0);

        // Write the byte indicator for addScalars
        file.write("m", 1);

        // Write the tag length and tag
        int tag_len = tag.size();
        file.write(reinterpret_cast<const char*>(&tag_len), sizeof(tag_len));
        file.write(tag.c_str(), tag.size());

        // Write the number of scalar tags
        int num_scalars = scalars.size();
        file.write(reinterpret_cast<const char*>(&num_scalars), sizeof(num_scalars));

        // Write each scalar tag length and tag
        for (const auto& [scalar_tag, _] : scalars) {
            int scalar_tag_len = scalar_tag.size();
            file.write(reinterpret_cast<const char*>(&scalar_tag_len), sizeof(scalar_tag_len));
            file.write(scalar_tag.c_str(), scalar_tag_len);
        }
    } else {
        // Read and validate the number of scalars in the existing file

        // Move past the 1-byte indicator
        file.seekg(1, std::ios::beg);

        // Read the stored tag length and skip the tag
        int stored_tag_len;
        file.read(reinterpret_cast<char*>(&stored_tag_len), sizeof(stored_tag_len));
        file.seekg(stored_tag_len, std::ios::cur);

        // Read the number of scalar tags
        int existing_num_scalars;
        file.read(reinterpret_cast<char*>(&existing_num_scalars), sizeof(existing_num_scalars));

        // Check if the number of scalars matches
        if (existing_num_scalars != scalars.size()) {
            std::cerr << "Error: Number of scalar tags does not match the existing data.\n";
            return false;
        }
    }

    // Append each scalar value to the file, along with global_step and timestamp
    file.seekp(0, std::ios::end);
    for (const auto& [_, scalar_value] : scalars) {
        file.write(reinterpret_cast<const char*>(&scalar_value), sizeof(scalar_value));
    }

    // Write global_step
    file.write(reinterpret_cast<const char*>(&global_step), sizeof(global_step));

    // Append current time in seconds as a floating-point value
    auto time_since_epoch_ns = now.time_since_epoch().count();
    double current_time_in_seconds = static_cast<double>(time_since_epoch_ns) / static_cast<double>(1e9);
    file.write(reinterpret_cast<const char*>(&current_time_in_seconds), sizeof(current_time_in_seconds));

    return true;
}
