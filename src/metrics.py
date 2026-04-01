"""
Metrics calculation for catastrophic forgetting experiments.
"""

from typing import List, Dict, Any

def calculate_forgetting_metrics(task_accuracies_history: List[List[float]], initial_accuracies: List[float]) -> Dict[str, List[float]]:
    """
    Calculate forgetting rates for each task after learning subsequent tasks.
    
    Args:
        task_accuracies_history: A list of lists where task_accuracies_history[i] 
                                 contains the accuracy of all seen tasks after evaluating after task i.
        initial_accuracies: Accuracy of each task right after it was first learned.
        
    Returns:
        A dictionary mapping task labels (e.g., "Task 1") to a list of accuracy drops 
        at each subsequent evaluation point.
    """
    forgetting_rate = {}
    
    # For each task except the very last one (as it has no subsequent tasks to forget it)
    for task_idx in range(len(initial_accuracies) - 1):
        forgetting = []
        
        # Calculate forgetting for the task at each subsequent evaluation point
        for eval_idx, accuracies in enumerate(task_accuracies_history):
            # Only consider evaluation points AFTER the task was first trained
            if eval_idx > task_idx and task_idx < len(accuracies):
                drop = initial_accuracies[task_idx] - accuracies[task_idx]
                forgetting.append(drop)
                
        if forgetting:
            forgetting_rate[f"Task {task_idx + 1}"] = forgetting
            
    return forgetting_rate
