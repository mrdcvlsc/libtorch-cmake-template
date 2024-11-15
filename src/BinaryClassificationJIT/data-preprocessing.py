import math
from typing import List, TypedDict

import os
import shutil
import random

# Define the TypedDict
class DatasetPartitionConfig(TypedDict):
    folder_name: str
    distribution: float

# Function to split and move images
def split_images(main_class_folder, partition_configs: List[DatasetPartitionConfig]):
    images = [img for img in os.listdir(main_class_folder) if os.path.isfile(os.path.join(main_class_folder, img))]
    class_name = os.path.split(main_class_folder)[-1]
    print(f'split_images: total images.len = {len(images)}\n')

    # Shuffle the images randomly for random selection
    for i in range(3):
        random.shuffle(images)
    
    # Determine the split index
    partition_indices = [0]
    for parition_config in partition_configs:
        partition_indices.append(partition_indices[-1] + int(math.floor(len(images) * parition_config['distribution'])))
    
    # check if indices are correct

    partition_indices[-1] = len(images) - 1

    for i in range(len(partition_indices) - 1):
        if partition_indices[i] >= partition_indices[i + 1]:
            print(f'Error indices generation: overlaping index detected at partition_indices[{i} - {i + 1}]')
            print(f'Error indices generation: partition_indices[{i}] = {partition_indices[i]}')
            print(f'Error indices generation: partition_indices[{i + 1}] = {partition_indices[i + 1]}')
            print(partition_indices)
            exit(-1)

    # make the partitions of images for the current class
    partitioned_image_sets = []
    for i in range(len(partition_configs)):
        partitioned_image_sets.append(images[partition_indices[i]:partition_indices[i + 1]])

    # Split the images into train and validation sets

    for i, image_partition in enumerate(partitioned_image_sets):
        # Move the images to the appropriate folders
        print(f'Moving images from "{main_class_folder}" to "{os.path.join(partition_configs[i]["folder_name"], class_name)}"')
        for img in image_partition:
            shutil.copy(
                os.path.join(main_class_folder, img),
                os.path.join(partition_configs[i]['folder_name'], class_name, img)
            )
        print(f'Moved images from "{main_class_folder}" to "{partition_configs[i]["folder_name"]}"\n')

# Function to count the number of images in a folder
def count_images(folder):
    return len([file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))])

if __name__ == "__main__":
    ############# COUNT NUMBER OF CATS AND DOG IMAGES #############

    # Define path to the folder of the image dataset which should
    # contain the class folders inside, where each class folders
    # contains their respective images
    DatasetDir = os.path.join('data', 'microsoft-catsvsdogs-dataset', 'PetImages')

    partition_folder = 'partitioned-dataset'

    if not os.path.exists(os.path.join(DatasetDir, partition_folder)):

        # Get all the name of the folders which is also the class names of images in the dataset
        ClassNames = [folder for folder in os.listdir(DatasetDir) if os.path.isdir(os.path.join(DatasetDir, folder))]
        print(f'class_names : {ClassNames}\n')

        # get the path of all class folders
        OriginalCalssImgDirs = []
        for class_name in ClassNames:
            OriginalCalssImgDirs.append(os.path.join(DatasetDir, class_name))

        # Count images in each class folder
        for i, class_dir in enumerate(OriginalCalssImgDirs):
            img_count = count_images(class_dir)
            print(f'Number of {ClassNames[i]} images: {img_count}')
        print()

        ############ CREATE TRAINING AND VALIDATION DATASET FOLDERS ##############

        # Define the new class folders

        # Annotate the list of TypedDicts
        DatasetFolderPartitionConfig: List[DatasetPartitionConfig] = [
            { 'folder_name': os.path.join(DatasetDir, partition_folder, 'training_dataset'), 'distribution': 17/20 },
            { 'folder_name': os.path.join(DatasetDir, partition_folder, 'testing_dataset'), 'distribution': 2/20 },
            { 'folder_name': os.path.join(DatasetDir, partition_folder, 'validation_dataset'), 'distribution': 1/20 }
        ]

        partitioned_dataset_dirs = []
        for config in DatasetFolderPartitionConfig:
            partitioned_dataset_dirs.append(config['folder_name'])

        # Create partitioned class directories if they don't exist
        print('Creating Dataset Folder Partitions')
        for partition_dir in partitioned_dataset_dirs:
            for class_name in ClassNames:
                os.makedirs(os.path.join(partition_dir, class_name), exist_ok=True)
                print(f'Created "{class_name}" folder inside "{partition_dir}"')
        print()

        print("Partitioning Image Dataset Folder")
        for class_dir in OriginalCalssImgDirs:
            split_images(class_dir, DatasetFolderPartitionConfig)

        print("Images have been split and moved successfully.")
    else:
        print('Dataset Partition Folder Already Found: Pre-processing of dataset was skipped')