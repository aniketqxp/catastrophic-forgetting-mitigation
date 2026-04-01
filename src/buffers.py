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
        Add samples to the buffer using random replacement if capacity is reached.
        """
        if isinstance(images, torch.Tensor):
            new_images = [images[i] for i in range(images.size(0))]
        else:
            new_images = images
        
        if isinstance(labels, torch.Tensor):
            new_labels = labels.tolist()
        else:
            new_labels = labels

        for img, lbl in zip(new_images, new_labels):
            if len(self.data) < self.capacity:
                self.data.append(img)
                self.labels.append(lbl)
            else:
                # Randomly replace an existing sample (Reservoir-style)
                idx = random.randint(0, self.capacity - 1)
                self.data[idx] = img
                self.labels[idx] = lbl

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
