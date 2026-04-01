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
    train_task_with_replay,
    train_task_with_ewc,
    compute_fisher_matrix
)
from .utils import set_seed, get_device
from .metrics import calculate_forgetting_metrics
from .plotting import plot_all_results, display_accuracy_table, plot_forgetting_rates, plot_task_accuracies, plot_learning_curves

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
    "train_task_with_ewc",
    "compute_fisher_matrix",
    "set_seed",
    "get_device",
    "calculate_forgetting_metrics",
    "plot_all_results",
    "display_accuracy_table",
    "plot_forgetting_rates",
    "plot_task_accuracies",
    "plot_learning_curves",
]
