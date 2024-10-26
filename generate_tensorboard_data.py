import os
import struct
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Function to parse binary files and extract stored data
def parse_add_scalar_binary_file(filepath):
    data_entries = []

    with open(filepath, "rb") as file:
        # skip the byte indicator
        _ = file.read(1)

        # get the tag length
        tag_len = file.read(4)
        tag_len, = struct.unpack("<i", tag_len)

        # Read the tag string
        tag = file.read(tag_len).decode("utf-8")

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
    else:
        print(f'Creating {run_folder_name}...')

    writer = SummaryWriter(f"{log_dir}/training_{timestamp}")

    for filepath in Path(run_dir).glob("*.scalar"):
        print(f'Reading {filepath}...')

        # get the byte indicator
        with open(filepath, "rb") as file:
            byte_indicator = file.read(1).decode("utf-8")

        print(f"Byte Indicator: {byte_indicator} : {filepath}\n")
            
        if byte_indicator == "s":
            print(f"Extracting {filepath} as addScalar\n")

            tag, data_entries = parse_add_scalar_binary_file(filepath)
            for scalar_value, global_step, timestamp in data_entries:
                # Log each scalar value with the global_step and timestamp
                writer.add_scalar(tag.replace('0', ''), scalar_value, global_step, walltime=timestamp)
        else:
            print("Invalid Byte Indicator For SummaryWriter Binary Saved Data")

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
