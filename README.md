# Catastrophic Forgetting Mitigation

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#)

A production-grade implementation and comparative study of techniques to mitigate catastrophic forgetting in neural networks. This project enables deep learning models to learn sequentially from new tasks without overwriting previously acquired knowledge, a core requirement for lifelong learning systems.

## The Value Proposition

Traditional neural networks suffer from catastrophic forgetting: when trained on a new task, they abruptly lose performance on previous ones. This repository provides a unified framework to evaluate and deploy state-of-the-art Continual Learning (CL) strategies, including regularization, architectural, and rehearsal-based methods.

## System Architecture

The following diagram illustrates the data flow and mitigation pipeline:

```mermaid
graph TD
    %% Node Definitions
    A[/Data Stream/] --> B{Task Type}
    B -->|Permuted MNIST| C[Task-Incremental]
    B -->|Split MNIST| D[Class-Incremental]
    
    C --> E([L2/EWC/SI: Regularization])
    C --> F([PackNet: Architectural])
    D --> H([Replay/Rehearsal: Memory])
    
    E --> G[(Optimized Model)]
    F --> G
    H --> G
    
    G --> I>Evaluation Metrics]
    I --> J[/Accuracy & Forgetting Results/]

    %% Styling
    style A fill:#2d3436,stroke:#000,color:#fff
    style B fill:#6c5ce7,stroke:#000,color:#fff
    style C fill:#0984e3,stroke:#000,color:#fff
    style D fill:#0984e3,stroke:#000,color:#fff
    style E fill:#00b894,stroke:#000,color:#fff
    style F fill:#00b894,stroke:#000,color:#fff
    style H fill:#00b894,stroke:#000,color:#fff
    style G fill:#fdcb6e,stroke:#000,color:#000
    style I fill:#d63031,stroke:#000,color:#fff
    style J fill:#2d3436,stroke:#000,color:#fff
```

## Key Features

- **Comprehensive Library**: Implementations of EWC, SI, PackNet, and Experience Replay.
- **Dual Scenarios**: Support for both Task-Incremental (Permuted MNIST) and Class-Incremental (Split-MNIST) learning.
- **Modular Design**: Core logic decoupled from execution, allowing easy extension of new models or datasets.
- **Unified CLI**: Single entry point for running large-scale experiments.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+

### Installation
```bash
# Clone the repository
git clone https://github.com/aniketqxp/catastrophic-forgetting-mitigation.git
cd catastrophic-forgetting-mitigation

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
Run a baseline experiment using the unified CLI:
```bash
python main.py --method baseline --tasks 5 --epochs 5
```

Or run Elastic Weight Consolidation (EWC) to see mitigation in action:
```bash
python main.py --method ewc --tasks 5 --epochs 5
```

## Results Visualization

Detailed metrics and learning curves are generated for every run.

### Training Accuracy Curves
![Accuracy Curves](assets/Baseline_task_plot.png)
*Figure 1: Comparison of accuracy preservation across sequential tasks.*

### Knowledge Retention Analysis
![Forgetting Analysis](assets/SI_permutated_plot.png)
*Figure 2: Visualization of knowledge retention using Synaptic Intelligence (SI).*

## Project Structure

```text
├── Code/                   # Original research notebooks
├── docs/                   # Extended methodology and reports
├── src/                    # Core library (Datasets, Models, Buffers)
├── assets/                 # Generated visualizations and plots
├── main.py                 # Unified execution entry point
├── pyproject.toml          # Project configuration
└── requirements.txt        # Dependency manifest
```

## License
Distributed under the MIT License. See `LICENSE` for more information.
