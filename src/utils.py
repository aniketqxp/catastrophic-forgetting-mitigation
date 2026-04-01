"""
Utility functions for catastrophic forgetting experiments.
Includes seed management, logging, and common helper functions.
"""

import torch
import numpy as np
import random


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Additional PyTorch settings for reproducibility
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior (may be slower but ensures reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the appropriate device (GPU if available, otherwise CPU).
    
    Returns:
        torch.device: The device to use for training
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
