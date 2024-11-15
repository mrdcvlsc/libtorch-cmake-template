import os

import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

import matplotlib.pyplot as plt

if __name__ == "__main__":

    ################################### PREPARE TRAINING DATASET ###########

    training_data_path = os.path.join(
        "data", "microsoft-catsvsdogs-dataset", "PetImages",
        "partitioned-dataset", "training_dataset"
    )

    train_data = torchvision.datasets.ImageFolder(
        training_data_path,
        transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

    train_inputs, train_labels = next(iter(train_dataloader))
    train_inputs : torch.Tensor = train_inputs
    train_labels : torch.Tensor = train_labels

    print("train_inputs = ", train_inputs)
    print("train_labels = ", train_labels)
    print("train_inputs.shape = ", train_inputs.shape)
    print("train_labels.shape = ", train_labels.shape)
    print("train_inputs.dtype = ", train_inputs.dtype)
    print("train_labels.dtype = ", train_labels.dtype)

    # Create a figure with 1 row and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(10,5))

    idx = 0
    while idx < 4:
        axes[0][idx].imshow(train_inputs[idx].clone().permute(1, 2, 0), cmap='brg')
        axes[1][idx].imshow(train_inputs[idx + 4].clone().permute(1, 2, 0), cmap='brg')
        idx += 1

    # Display the plot
    plt.show()
