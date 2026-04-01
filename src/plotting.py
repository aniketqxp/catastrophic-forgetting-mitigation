"""
Plotting utilities for catastrophic forgetting experiments.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

def plot_learning_curves(training_losses: List[float], title="Loss Learning Curve"):
    """Plot the training loss across all epochs."""
    plt.figure(figsize=(12, 5))
    plt.plot(training_losses, label='Training Loss', color='blue', marker='o')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_task_accuracies(task_accuracies_history: List[List[float]], num_tasks: int, title="Task Accuracies During Sequential Training"):
    """Plot the accuracy of each task as new tasks are learned sequentially."""
    plt.figure(figsize=(12, 5))
    for task_idx in range(num_tasks):
        epochs = []
        values = []
        for eval_idx, accs in enumerate(task_accuracies_history):
            if task_idx < len(accs):
                epochs.append(eval_idx + 1)
                values.append(accs[task_idx])
        if values:
            plt.plot(epochs, values, marker='o', label=f'Task {task_idx+1}')
    
    plt.title(title)
    plt.xlabel('Training Progress (Tasks Seen)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def display_accuracy_table(task_accuracies_history: List[List[float]]):
    """Print a textual table of task accuracies at different points in time."""
    accuracy_rows = []
    for accs in task_accuracies_history:
        row = {f'Task {i+1}': accs[i] for i in range(len(accs))}
        accuracy_rows.append(row)
    accuracy_table = pd.DataFrame(accuracy_rows)
    accuracy_table.index = [f'After Task {i+1}' for i in range(len(task_accuracies_history))]
    print('\nTask Accuracies Table:\n')
    print(accuracy_table.to_string())

def plot_forgetting_rates(forgetting_rate: Dict[str, List[float]], title="Forgetting Rates"):
    """Plot how much each task's accuracy degraded over time."""
    if not forgetting_rate:
        print("No forgetting data available.")
        return
        
    plt.figure(figsize=(12, 6))
    for task_name, forgetting in forgetting_rate.items():
        if forgetting:
            plt.plot(range(1, len(forgetting) + 1), forgetting, marker='o', label=f'{task_name} Forgetting')
    plt.title(title)
    plt.xlabel('Subsequent Tasks')
    plt.ylabel('Forgetting (Accuracy Drop %)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_all_results(training_losses: List[float], task_accuracies_history: List[List[float]], forgetting_rate: Dict[str, List[float]], num_tasks: int, method_name="Sequential Training"):
    """Helper to plot all metrics consecutively."""
    plot_learning_curves(training_losses, title=f'Loss Learning Curve - {method_name}')
    plot_task_accuracies(task_accuracies_history, num_tasks, title=f'Task Accuracies During {method_name}')
    display_accuracy_table(task_accuracies_history)
    plot_forgetting_rates(forgetting_rate, title=f'Forgetting Rates ({method_name})')
