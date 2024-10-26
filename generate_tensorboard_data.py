import os
import struct
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Function to parse binary files and extract stored data
def parse_binary_file(filepath):
    data_entries = []

    with open(filepath, "rb") as file:
        # Read the first 30 bytes (for the tag, if needed)
        tag = file.read(30).decode("utf-8").rstrip("\x00")

        # Continue reading entries, each entry is 20 bytes
        while True:
            entry = file.read(20)
            if len(entry) < 20:
                break

            scalar_value, global_step, timestamp = struct.unpack("<did", entry)
            data_entries.append((scalar_value, global_step, timestamp))

    return tag, data_entries

# Locate the latest directory inside raw_data_runs
def get_run_directory(base_dir="raw_data_runs"):
    base_path = Path(base_dir)
    directories = sorted(base_path.iterdir(), key=os.path.getmtime, reverse=True)
    return directories

# Write parsed data to TensorBoard
def write_to_tensorboard(run_dir, log_dir="runs"):
    timestamp = run_dir.name  # Use the folder name as timestamp
    run_folder_name = os.path.join(log_dir, f"training_{timestamp}")

    if os.path.exists(run_folder_name):
        print(f'Skipping {run_folder_name}, tensorboard data is already generated')
        return

    writer = SummaryWriter(f"{log_dir}/training_{timestamp}")

    for filepath in Path(run_dir).glob("*.bin"):
        tag, data_entries = parse_binary_file(filepath)
        for scalar_value, global_step, timestamp in data_entries:
            # Log each scalar value with the global_step and timestamp
            writer.add_scalar(tag.replace('0', ''), scalar_value, global_step, walltime=timestamp)
    
    writer.close()

if __name__ == "__main__":
    directories = get_run_directory()

    if directories:
        print(f"Processing data from: {directories}\n")
    else:
        print("No data available to process.\n")

    for directory in directories:
        if directory:
            print(f"Processing data from: {directory}")
            write_to_tensorboard(directory)
        else:
            print("No data available to process.")
