"""
Experience replay buffer for continual learning.
Stores samples from previous tasks to mitigate catastrophic forgetting.
"""

import torch
import random


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling from previous task data.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize ReplayBuffer.
        
        Args:
            capacity: Maximum number of samples to store in buffer
        """
        self.capacity = capacity
        self.data = []
        self.labels = []

    def add(self, images, labels):
        """
        Add samples to the buffer.
        
        Args:
            images: Tensor of image data
            labels: Tensor of corresponding labels
        """
        if isinstance(images, torch.Tensor):
            self.data.extend([images[i] for i in range(images.size(0))])
        else:
            self.data.extend(images)
        
        if isinstance(labels, torch.Tensor):
            self.labels.extend(labels.tolist())
        else:
            self.labels.extend(labels)
        
        # Keep buffer size within capacity
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]
            self.labels = self.labels[-self.capacity:]

    def sample(self, batch_size: int):
        """
        Sample a batch from the buffer.
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            Tuple of (images, labels) tensors
        """
        if len(self.data) == 0:
            return None, None
        
        indices = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        images = torch.stack([self.data[i] for i in indices])
        labels = torch.tensor([self.labels[i] for i in indices], dtype=torch.long)
        return images, labels
    
    def is_empty(self):
        """Check if buffer is empty."""
        return len(self.data) == 0
