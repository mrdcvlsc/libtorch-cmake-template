import os
import struct
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter

# Function to parse binary files and extract stored data
def parse_add_scalar_binary_file(filepath):
    data_entries = []

    with open(filepath, "rb") as file:

        # Read the 1-byte indicator
        indicator = file.read(1).decode("utf-8")
        if indicator != "s":
            raise ValueError("File does not support addScalar data.")

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

import struct

# Function to parse binary files created by addScalars
def parse_add_scalars_binary_file(filepath):
    data_entries = []

    with open(filepath, "rb") as file:

        # Read the 1-byte indicator
        indicator = file.read(1).decode("utf-8")
        if indicator != "m":
            raise ValueError("File does not support addScalars data.")

        # Read the tag length and tag
        tag_len = struct.unpack("<i", file.read(4))[0]
        tag = file.read(tag_len).decode("utf-8")

        # Read the number of scalar tags
        num_scalars = struct.unpack("<i", file.read(4))[0]

        # Read each scalar tag length and tag
        scalar_tags = []
        for _ in range(num_scalars):
            scalar_tag_len = struct.unpack("<i", file.read(4))[0]
            scalar_tag = file.read(scalar_tag_len).decode("utf-8")
            scalar_tags.append(scalar_tag)

        # Read the scalar values, global_step, and timestamp
        entry_size = (8 * num_scalars) + 12  # size of each entry: num_scalars * double + int + double
        while True:
            entry = file.read(entry_size)
            if len(entry) < entry_size:
                break

            # Extract scalar values, global_step, and timestamp from the entry
            scalar_values = struct.unpack(f"<{'d' * num_scalars}", entry[:8 * num_scalars])
            global_step = struct.unpack("<i", entry[8 * num_scalars:8 * num_scalars + 4])[0]
            timestamp = struct.unpack("<d", entry[8 * num_scalars + 4:])[0]
            data_entries.append((dict(zip(scalar_tags, scalar_values)), global_step, timestamp))

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

            print(f"Extracting {filepath} as addScalar (s)\n")

            tag, data_entries = parse_add_scalar_binary_file(filepath)
            for scalar_value, global_step, timestamp in data_entries:
                writer.add_scalar(tag, scalar_value, global_step, walltime=timestamp)        
        else:
            print("Invalid Byte Indicator For SummaryWriter.add_scalar (s) Binary Saved Data")

    for filepath in Path(run_dir).glob("*.scalars"):
        print(f'Reading {filepath}...')

        # get the byte indicator
        with open(filepath, "rb") as file:
            byte_indicator = file.read(1).decode("utf-8")

        print(f"Byte Indicator: {byte_indicator} : {filepath}\n")
            
        if byte_indicator == "m":

            print(f"Extracting {filepath} as addScalars (m)\n")

            tag, data_entries = parse_add_scalars_binary_file(filepath)
            for scalar_values, global_step, timestamp in data_entries:
                writer.add_scalars(tag, scalar_values, global_step, walltime=timestamp)
        
        else:
            print("Invalid Byte Indicator For SummaryWriter.add_scalars (m) Binary Saved Data")

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
