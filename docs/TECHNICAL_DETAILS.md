# Technical Details: Continual Learning Mitigation Framework

This document provides exhaustive engineering documentation for the `catastrophic-forgetting-mitigation` repository. It is intended for granular technical audits and architectural reference.

---

## 1. Logic & Algorithms

### Elastic Weight Consolidation (EWC)
EWC implements a Bayesian approach to parameter preservation.
- **Mathematical Transformation**: The penalty term uses the diagonal approximation of the Fisher Information Matrix (FIM).
  - $L(\theta) = L_T(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{A,i}^*)^2$
- **Fisher Estimation**: FIM is estimated by accumulating squared gradients over the training distribution of task $T-1$ after its completion. 
- **Implementation Detail**: The diagonal elements are stored in a dictionary mapping parameter names to tensors, which is efficient for $O(N)$ penalty calculation during the backward pass of subsequent tasks.

### Synaptic Intelligence (SI)
SI tracks parameter importance online without requiring a separate post-task estimation phase.
- **Algorithm**: It computes "surrogate loss" based on the trajectory length of gradients in parameter space.
- **Math**: $\omega_k^\mu = \int_{t^{\mu-1}}^{t^\mu} g(t) \cdot d\theta(t)$ where $g(t)$ is the gradient and $\theta(t)$ is the parameter trajectory.
- **Why**: This approach is computationally cheaper than EWC and better captures the importance of weights that were critical during the early stages of optimization.

### PackNet (Architectural Pruning)
PackNet prevents forgetting by strictly isolating task-specific subnetworks.
- **Logic**: Iterative magnitude-based pruning. 
- **Workflow**: 
  1. Train Task A.
  2. Prune $X\%$ of weights with smallest absolute magnitude.
  3. Freeze the remaining important weights for Task A.
  4. Train Task B using only the previously pruned (free) weights.
- **Implementation Hack**: Gradient masking is used to ensure that optimizer steps do not modify frozen weights.

### Experience Replay & Reservoir Sampling
- **Buffer Management**: Uses **Reservoir Sampling** to maintain a fixed-size buffer $N$.
  - Probability of inclusion for sample $t$: $P(t) = \frac{N}{t}$.
- **Batch Balancing**: Training batches are constrained to a 1:1 ratio between current task data and replayed data.
- **Why**: This eliminates the "recency bias" where the model forgets older tasks because they are not represented in the current gradient update.

---

## 2. Engineering "Fires" & Hacks

### Path Import Resolution
- **Issue**: Standard Python module resolution fails when running notebooks in a sub-directory (`Code/`) that need to access a sibling source directory (`src/`).
- **Workaround**: Manual injection into `sys.path` within every notebook: `sys.path.insert(0, '..')`. This bypasses the need for complex `PYTHONPATH` environment variables or local `pip install -e .` during rapid prototyping.

### Fisher Matrix Accumulation
- **Challenge**: Computing the full Fisher Information Matrix is $O(N^2)$ in space, which is impossible for modern NNs.
- **Resolution**: Used a diagonal approximation, storing only the variance of the gradients. This reduces space complexity to $O(N)$ while preserving $90\%+$ of the regularization effectiveness.

### Circular Import Handling
- **Issue**: High coupling between `training.py` (which needs models) and `models.py`.
- **Hack**: All dependency orchestration is handled at the `src/__init__.py` level, allowing modules to import from the package namespace rather than each other directly.

---

## 3. Performance & Scaling

### Memory Optimization
- **Tensor Casting**: Explicit use of `.data.clone()` when storing optimal parameters for EWC. This ensures that the stored weights are detached from the computation graph, preventing massive memory leaks from accumulated Autograd nodes.
- **Buffer Capping**: Replay buffers use a hard limit on samples (e.g., 200 samples) to ensure the memory footprint is constant even if the project scales to hundreds of tasks.

### Device Orchestration
- **Logic**: `get_device()` abstraction layer.
- **Backend Support**: Automatically detects and prioritizes `cuda` (NVIDIA), `mps` (Apple Silicon), and falls back to `cpu`. This ensures zero-configuration portability between development and training environments.

---

## 4. Architecture & Domain Nuance

### State Management
- **Persistence**: Fisher matrices and optimal parameters are handled as distinct state dictionaries.
- **Why**: This allows for "additive" regularization where the model can be updated with $N$ different Fisher matrices representing $N$ different tasks, enabling true lifelong learning.

### Decoupling Logic from Execution
- **Architecture**: Core implementation resides in `src/` (the "Library"), while `Code/` (the "Experiment") and `main.py` (the "CLI") handle the execution flow.
- **Advantage**: This ensures that changes to the training algorithm (e.g., tweaking the EWC lambda formula) automatically propagate to all notebooks and the CLI without code duplication.

### Domain Nuance: Class-Incremental Split
- **Rule**: In Class-Incremental scenarios, the output layer must account for all classes seen to date. 
- **Method**: The `evaluate_cumulative_tasks` function concatenates the test datasets of all tasks to perform a global accuracy check, which is the "Gold Standard" for measuring true catastrophic forgetting.
