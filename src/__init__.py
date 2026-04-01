"""
Catastrophic Forgetting Mitigation Package

This package provides datasets, models, and training utilities for continual learning experiments.
"""

from .datasets import PermutedMNIST, ClassIncrementalMNIST
from .models import SimpleNN
from .buffers import ReplayBuffer
from .training import (
    train_task,
    evaluate_all_tasks,
    evaluate_cumulative_tasks,
    store_initial_params,
    train_task_with_l2_regularization,
    train_task_with_replay
)
from .utils import set_seed, get_device

__all__ = [
    "PermutedMNIST",
    "ClassIncrementalMNIST",
    "SimpleNN",
    "ReplayBuffer",
    "train_task",
    "evaluate_all_tasks",
    "evaluate_cumulative_tasks",
    "store_initial_params",
    "train_task_with_l2_regularization",
    "train_task_with_replay",
    "set_seed",
    "get_device",
]
