"""
Neural network models for continual learning experiments.
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """
    Simple two-layer fully connected neural network.
    Used as the baseline model for catastrophic forgetting experiments.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize SimpleNN.
        
        Args:
            input_size: Size of input features (e.g., 784 for flattened MNIST)
            hidden_size: Number of hidden units in first layer
            output_size: Number of output classes
        """
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
