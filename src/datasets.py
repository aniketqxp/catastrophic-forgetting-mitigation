"""
Dataset classes for continual learning experiments.
Includes both permuted MNIST and class-incremental MNIST datasets.
"""

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


class PermutedMNIST(Dataset):
    """
    Permuted MNIST dataset for task-incremental learning.
    Each task applies a fixed permutation to the pixel space.
    """
    
    def __init__(self, root: str, train: bool = True, transform=None, permutations=None):
        """
        Initialize PermutedMNIST dataset.
        
        Args:
            root: Path to store MNIST data
            train: Whether to load training or test set
            transform: Optional transforms to apply to images
            permutations: List of pixel permutation indices for each task
        """
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=root, train=train, transform=transforms.ToTensor(), download=True
        )
        self.transform = transform
        self.permutations = permutations
        self.train = train

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        if self.permutations is not None:
            image = image.view(-1)[self.permutations].view(image.shape)
        if self.transform:
            image = self.transform(image)
        return image, label


class ClassIncrementalMNIST(Dataset):
    """
    Class-incremental MNIST dataset where each task introduces new classes.
    """
    
    def __init__(self, root: str, train: bool = True, transform=None, classes=None):
        """
        Initialize ClassIncrementalMNIST dataset.
        
        Args:
            root: Path to store MNIST data
            train: Whether to load training or test set
            transform: Optional transforms to apply to images
            classes: List of class indices to include in this task
        """
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=root, train=train, transform=transforms.ToTensor(), download=True
        )
        self.transform = transform
        self.classes = classes
        self.train = train
        
        # Filter data to include only specified classes
        self.data = []
        self.targets = []
        for image, label in self.mnist_dataset:
            if label in self.classes:
                self.data.append(image)
                self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
