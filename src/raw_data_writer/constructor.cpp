#include "raw_data_writer.hpp"

RawDataWriter::RawDataWriter() {
    // Check and create "raw_data_runs" directory
    std::filesystem::path base_dir = "raw_data_runs";
    if (!std::filesystem::exists(base_dir)) {
        std::filesystem::create_directory(base_dir);
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
    std::filesystem::create_directory(directory);
}