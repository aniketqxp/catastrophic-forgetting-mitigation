"""
Training and evaluation functions for continual learning experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset


def train_task(model, task_idx: int, train_loader, criterion, optimizer, epochs: int = 5):
    """
    Train model on a single task.
    
    Args:
        model: Neural network model to train
        task_idx: Index of the current task (for logging)
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for training
        epochs: Number of epochs to train
        
    Returns:
        Tuple of (losses, accuracies) for each epoch
    """
    task_train_loss = []
    task_train_acc = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        task_train_loss.append(epoch_loss)
        task_train_acc.append(epoch_acc)
        
        print(f'Task {task_idx+1}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return task_train_loss, task_train_acc


def evaluate_all_tasks(model, test_loaders):
    """
    Evaluate model on all tasks.
    
    Args:
        model: Neural network model to evaluate
        test_loaders: List of DataLoaders for test data of each task
        
    Returns:
        List of accuracies for each task
    """
    accuracies = []
    
    for i, test_loader in enumerate(test_loaders):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Task {i+1} Accuracy: {accuracy:.2f}%')
    
    return accuracies


def evaluate_cumulative_tasks(model, test_datasets):
    """
    Evaluate model on cumulative set of all seen tasks (for class-incremental setup).
    
    Args:
        model: Neural network model to evaluate
        test_datasets: List of datasets for each task (will be concatenated)
        
    Returns:
        Overall accuracy on combined test set
    """
    combined_test_set = ConcatDataset(test_datasets)
    test_loader = DataLoader(combined_test_set, batch_size=64, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def store_initial_params(model):
    """
    Store initial model parameters (used for regularization-based methods).
    
    Args:
        model: Neural network model
        
    Returns:
        Dictionary mapping parameter names to their values
    """
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()
    return initial_params


def train_task_with_l2_regularization(
    model, task_idx: int, train_loader, criterion, optimizer, 
    initial_params_list=None, l2_lambda: float = 0.01, epochs: int = 5
):
    """
    Train model on a task with L2 regularization to previous parameters.
    
    Args:
        model: Neural network model to train
        task_idx: Index of current task
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for training
        initial_params_list: List of initial parameters for each previous task
        l2_lambda: Regularization strength
        epochs: Number of epochs to train
        
    Returns:
        Tuple of (losses, accuracies) for each epoch
    """
    task_train_loss = []
    task_train_acc = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Add L2 regularization to parameters from all previous tasks
            if initial_params_list:
                l2_reg = 0.0
                for initial_params in initial_params_list:
                    for name, param in model.named_parameters():
                        if name in initial_params:
                            l2_reg += torch.sum((param - initial_params[name]) ** 2)
                loss = loss + (l2_lambda / 2) * l2_reg
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        task_train_loss.append(epoch_loss)
        task_train_acc.append(epoch_acc)
        
        print(f'Task {task_idx+1}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return task_train_loss, task_train_acc


def train_task_with_replay(
    model, task_idx: int, train_loader, criterion, optimizer, 
    replay_buffer, epochs: int = 5, batch_size: int = 64
):
    """
    Train model on a task using experience replay.
    
    Args:
        model: Neural network model to train
        task_idx: Index of current task
        train_loader: DataLoader for current task training data
        criterion: Loss function
        optimizer: Optimizer for training
        replay_buffer: ReplayBuffer instance for sampling from previous tasks
        epochs: Number of epochs to train
        batch_size: Batch size for training
        
    Returns:
        Tuple of (losses, accuracies) for each epoch
    """
    task_train_loss = []
    task_train_acc = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # Sample from the replay buffer
            if not replay_buffer.is_empty():
                replay_inputs, replay_labels = replay_buffer.sample(batch_size // 2)
                if replay_inputs is not None:
                    inputs = torch.cat((inputs, replay_inputs))
                    labels = torch.cat((labels, replay_labels))

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        task_train_loss.append(epoch_loss)
        task_train_acc.append(epoch_acc)
        
        print(f'Task {task_idx+1}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return task_train_loss, task_train_acc
