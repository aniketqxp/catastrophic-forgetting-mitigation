import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from src import (
    set_seed, PermutedMNIST, SimpleNN, 
    train_task, train_task_with_ewc, train_task_with_replay,
    compute_fisher_matrix, evaluate_all_tasks,
    calculate_forgetting_metrics, plot_all_results
)

def run_experiment(method='baseline', num_tasks=5, epochs=5, lr=0.01, batch_size=64, **kwargs):
    set_seed(42)
    input_size = 28 * 28
    hidden_size = 256
    output_size = 10
    
    # Setup data
    permutations = [torch.randperm(input_size) for _ in range(num_tasks)]
    train_tasks = [PermutedMNIST(root="./data", train=True, permutations=permutations[i]) for i in range(num_tasks)]
    test_tasks = [PermutedMNIST(root="./data", train=False, permutations=permutations[i]) for i in range(num_tasks)]
    
    train_loaders = [DataLoader(train_tasks[i], batch_size=batch_size, shuffle=True) for i in range(num_tasks)]
    test_loaders = [DataLoader(test_tasks[i], batch_size=batch_size, shuffle=False) for i in range(num_tasks)]
    
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    training_losses = []
    task_accuracies_history = []
    initial_accuracies = []
    
    fisher_matrices = []
    optimal_params = []
    
    print(f"Starting {method} experiment with {num_tasks} tasks...")
    
    for task_idx in range(num_tasks):
        print(f"\n{'='*50}\nTraining on Task {task_idx+1}\n{'='*50}")
        
        if method == 'ewc':
            ewc_lambda = kwargs.get('ewc_lambda', 40.0)
            task_loss, _ = train_task_with_ewc(
                model, task_idx, train_loaders[task_idx], criterion, optimizer,
                fisher_matrices if task_idx > 0 else None,
                optimal_params if task_idx > 0 else None,
                ewc_lambda=ewc_lambda, epochs=epochs
            )
            # Update Fisher and optimal params for EWC
            fisher = compute_fisher_matrix(model, train_loaders[task_idx], criterion)
            fisher_matrices.append(fisher)
            optimal_params.append({name: param.data.clone() for name, param in model.named_parameters()})
            
        elif method == 'replay':
            from src import ReplayBuffer
            buffer_size = kwargs.get('buffer_size', 200)
            replay_buffer = ReplayBuffer(buffer_size)
            # Replay training logic would need more complexity here or in src
            # For simplicity in main.py, we call the src function
            task_loss, _ = train_task_with_replay(
                model, task_idx, train_loaders[task_idx], criterion, optimizer,
                replay_buffer, epochs=epochs
            )
        else: # baseline
            task_loss, _ = train_task(model, task_idx, train_loaders[task_idx], criterion, optimizer, epochs=epochs)
            
        training_losses.extend(task_loss)
        
        print("\nEvaluating on all tasks seen so far:")
        task_accs = evaluate_all_tasks(model, test_loaders[:task_idx+1])
        task_accuracies_history.append(task_accs)
        initial_accuracies.append(task_accs[0] if task_idx == 0 else task_accs[-1])
        
    forgetting_rate = calculate_forgetting_metrics(task_accuracies_history, initial_accuracies)
    plot_all_results(training_losses, task_accuracies_history, forgetting_rate, num_tasks, method_name=method.upper())
    print("\nExperiment completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Continual Learning Experiments')
    parser.add_argument('--method', type=str, default='baseline', choices=['baseline', 'ewc', 'replay'], help='CL method')
    parser.add_argument('--tasks', type=int, default=5, help='Number of tasks')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per task')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    args = parser.parse_args()
    run_experiment(method=args.method, num_tasks=args.tasks, epochs=args.epochs, lr=args.lr)
