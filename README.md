# Catastrophic Forgetting Mitigation

A comprehensive study of mitigating **catastrophic forgetting** in neural networks through continual learning. This project implements and compares multiple state-of-the-art techniques to enable models to learn sequentially from new tasks without degrading performance on previously learned tasks.

## Problem Statement

Catastrophic forgetting occurs when neural networks trained sequentially on new tasks rapidly lose performance on old tasks. This is a fundamental challenge for **continual learning** and **lifelong learning** systems. Traditional neural networks suffer from this phenomenon because they lack mechanisms to preserve knowledge from previous tasks.

## Methodology

This project evaluates three categories of mitigation strategies on **Split-MNIST** datasets with both task-incremental and class-incremental learning scenarios:

### Learning Scenarios
- **Task-Incremental Learning (Permuted MNIST)**: Each task is a permutation of pixel space; model sees all 10 MNIST classes but in different orders
- **Class-Incremental Learning (Split-MNIST)**: Each task introduces new digit classes; later tasks combine with earlier ones

### Methods Evaluated

1. **Baseline (No Mitigation)**: Standard sequential training on tasks
2. **L2 Regularization**: Constrains parameter changes to stay close to initial values from previous tasks
3. **Elastic Weight Consolidation (EWC)**: Uses the Fisher Information Matrix to selectively protect parameters crucial for past tasks
4. **Synaptic Intelligence (SI)**: Online tracking of parameter importance during training to penalize changes to important weights
5. **PackNet**: Prunes unimportant weights and freezes important task-specific subnetworks to prevent interference
6. **Experience Replay**: Stores and replays samples from previous tasks using reservoir sampling and 1:1 balanced training batches
7. **Naive Rehearsal**: Combines current task with stored replay data using a simple FIFO buffer

### Metrics

- **Accuracy**: Performance on each task after sequential training
- **Forgetting Rate**: Accuracy drop on previous tasks after learning new ones
- **Training Time**: Computational efficiency per task
- **Model Capacity**: Parameter efficiency across tasks

## Project Structure

```
catastrophic-forgetting-mitigation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── docs/
│   └── METHODOLOGY.md                # Detailed methodology and results
│
├── src/                               # Modular source code
│   ├── __init__.py                   # Package exports
│   ├── datasets.py                   # Dataset classes (PermutedMNIST, ClassIncrementalMNIST)
│   ├── models.py                     # Neural network architectures (SimpleNN)
│   ├── training.py                   # Training and evaluation functions
│   ├── buffers.py                    # Experience replay buffer
│   └── utils.py                      # Utilities (seed setup, device detection)
│
├── Code/                              # Executable notebooks
│   ├── Baseline_task.ipynb           # Baseline on permuted MNIST
│   ├── Baseline_class.ipynb          # Baseline on class-incremental MNIST
│   ├── L2_permutated.ipynb           # L2 regularization on permuted MNIST
│   ├── EWC_permutated.ipynb          # Elastic Weight Consolidation on permuted MNIST
│   ├── SI_permutated.ipynb           # Synaptic Intelligence on permuted MNIST
│   ├── packnet.ipynb                 # PackNet on permuted MNIST
│   ├── experience_replay.ipynb       # Experience replay (reservoir + balanced) on class-incremental MNIST
│   └── naive_rehearsal.ipynb         # Naive rehearsal on class-incremental MNIST
│
├── Final Review Presentation.pptx    # Research findings presentation
└── Project Report.pdf                # Complete analysis report
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/aniketqxp/catastrophic-forgetting-mitigation.git
   cd catastrophic-forgetting-mitigation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run experiments** (using Jupyter)
   ```bash
   jupyter notebook Code/
   ```
   - Start with `Baseline_task.ipynb` to understand the core concept
   - Compare results across different mitigation methods

## Key Findings

- **Trade-offs Observed**: Different methods excel at different aspects:
  - L2 regularization provides good forgetting mitigation with minimal overhead
  - EWC offers superior task-aware protection compared to naive L2 by leveraging the Fisher Information Matrix
  - Experience replay offers the best accuracy but requires significant memory; balanced reservoir sampling is key
  - PackNet achieves excellent zero-forgetting performance by permanently freezing critical weights
  - Synaptic Intelligence provides interpretable online importance tracking without needing post-task Fisher computation

- **No Silver Bullet**: No single method dominates all scenarios; the choice depends on specific application requirements

- **Compute vs. Memory**: Regularization-based methods trade memory efficiency for slightly higher forgetting, while replay-based methods require more storage

## Reproducibility

All experiments use a fixed random seed (`seed=42`) for reproducibility. 

**Note**: Results may vary slightly depending on hardware (CPU vs. GPU) and software versions. The latest run uses PyTorch 2.0.0 and torchvision 0.15.0.

## Code Quality

- **Modular Design**: Common logic extracted into `src/` modules to avoid duplication
- **Type Hints**: Functions include parameter and return type annotations
- **Docstrings**: Comprehensive documentation for all modules and functions
- **Reproducibility**: Deterministic seeding and configurable random state management

## Reports & Presentations

- **Project Report** (`Project Report.pdf`): Detailed methodology, experimental results, tables, and analysis
- **Final Presentation** (`Final Review Presentation.pptx`): Executive summary and key visualizations

*Note: These documents are from the initial project run. Results may differ when re-running with the refactored code.*

## Author

**Aniket Shinde**

## References

- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.
- Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through Synaptic Intelligence. ICML.
- Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning. CVPR.
- Rusu, A. A., et al. (2016). Progressive Neural Networks. arXiv preprint arXiv:1606.04671.
- Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental Class Learning and Representation Learning. CVPR.
- Rusu, A. A., et al. (2020). Passive Open-World Object Detection. arXiv preprint arXiv:1904.04998.
