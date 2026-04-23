# Catastrophic Forgetting Mitigation

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#)

A production-grade implementation and comparative study of techniques to mitigate **catastrophic forgetting** in neural networks. This project enables deep learning models to learn sequentially from new tasks without overwriting previously acquired knowledge, a core requirement for lifelong learning systems.

## 🚀 The Value Proposition

Traditional neural networks suffer from catastrophic forgetting: when trained on a new task, they abruptly lose performance on previous ones. This repository provides a unified framework to evaluate and deploy state-of-the-art **Continual Learning (CL)** strategies, including regularization, architectural, and rehearsal-based methods.

## 🏗️ System Architecture

The following diagram illustrates the data flow and mitigation pipeline:

```mermaid
graph TD
    A[Data Stream] --> B{Task Type}
    B -- Permuted MNIST --> C[Task-Incremental]
    B -- Split MNIST --> D[Class-Incremental]
    
    subgraph Mitigation Strategies
        E[L2/EWC/SI] --> |Regularization| G[Optimized Model]
        F[PackNet] --> |Architectural| G
        H[Replay/Rehearsal] --> |Memory-based| G
    end
    
    C --> E
    C --> F
    D --> H
    
    G --> I[Evaluation Metrics]
    I --> J[Accuracy/Forgetting Rate]
```

## 🛠️ Key Features

- **Comprehensive Library**: Implementations of EWC, SI, PackNet, and Experience Replay.
- **Dual Scenarios**: Support for both Task-Incremental (Permuted MNIST) and Class-Incremental (Split-MNIST) learning.
- **Modular Design**: Core logic decoupled from execution, allowing easy extension of new models or datasets.
- **Unified CLI**: Single entry point for running large-scale experiments.

## 🚦 Getting Started

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

## 📊 Results Visualization

Detailed metrics and learning curves are generated for every run.

<!-- [INSERT_SCREENSHOT: Training Accuracy Curves] -->
![Accuracy Table](docs/results_table_placeholder.png)
*Example: Comparison of accuracy preservation across tasks.*

<!-- [INSERT_SCREENSHOT: Forgetting Rate Comparison] -->
![Forgetting Rates](docs/forgetting_rates_placeholder.png)
*Example: Visualization of knowledge retention using different CL methods.*

## 📂 Project Structure

```text
├── Code/                   # Original research notebooks
├── docs/                   # Extended methodology and reports
├── src/                    # Core library (Datasets, Models, Buffers)
├── main.py                 # Unified execution entry point
├── pyproject.toml          # Project metadata
└── requirements.txt        # Dependency manifest
```

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 👤 Author
**Aniket Shinde** - [GitHub](https://github.com/aniketqxp)
