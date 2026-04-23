# Methodology & Technical Details

## Continual Learning Framework

### Setup

All experiments follow a **sequential task-learning paradigm**:

1. **Initialize** a neural network
2. **For each task** (in sequence):
   - Train on task data
   - Evaluate on all tasks seen so far
   - Track metrics (accuracy, forgetting, time)
3. **Analyze** forgetting rates and trade-offs

### Network Architecture

All experiments use a simple **2-layer fully connected network**:

```
Input (784) → Linear(784 → 256) → ReLU → Linear(256 → 10) → Output
```

**Rationale**: Simplicity enables focus on continual learning algorithms rather than architectural complexity.

## Dataset Details

### 1. Permuted MNIST (Task-Incremental)

- **Number of Tasks**: 5
- **Task Design**: Each task applies a fixed pixel permutation to MNIST
  - Train each task on all 10 digit classes with permuted pixel layout
  - Model must adapt to different input distributions without forgetting

**Formula**: For task $t$, apply permutation $\pi_t$ to input $x$:
$$\tilde{x}_t = \text{permute}(x, \pi_t)$$

**Why this setup?**
- Tasks are truly disjoint in input space
- All tasks share the same label space (0-9)
- Difficulty increases as more tasks are learned

### 2. Class-Incremental MNIST (Class-Incremental)

- **Number of Tasks**: 5
- **Classes per Task**: 2
- **Task Structure**:
  - Task 1: Digits {0, 1}
  - Task 2: Digits {2, 3}
  - Task 3: Digits {4, 5}
  - Task 4: Digits {6, 7}
  - Task 5: Digits {8, 9}

**Why this setup?**
- Realistic scenario where new classes arrive over time
- Model must expand output capacity conceptually
- Evaluation includes cumulative classes from all tasks

---

## Mitigation Methods

### Method 1: Baseline (No Mitigation)

**Algorithm**: Standard SGD training without regularization

```python
for task in tasks:
    model.train(task)
    model.evaluate(all_tasks)
```

**Expected Behavior**: Largest forgetting rates as model overwrites previous knowledge

---

### Method 2: L2 Regularization

**Key Idea**: Constrain parameters to stay close to their initial values from previous tasks

**Loss Function**:
$$\mathcal{L}_{\text{task}} = \mathcal{L}_{\text{CE}} + \frac{\lambda}{2} \sum_{i=1}^{t-1} \|w - w^*_i\|^2$$

where:
- $\mathcal{L}_{\text{CE}}$ = Cross-entropy loss on new task
- $w^*_i$ = Stored parameters after task $i$
- $\lambda$ = Regularization strength

**Hyperparameter**: $\lambda$ (default: 0.01)

**Pros**:
- Simple to implement
- Minimal computational overhead
- Effective for small task numbers

**Cons**:
- Treats all parameters equally
- Scales poorly with many tasks
- Can over-constrain parameters

---

### Method 3: Elastic Weight Consolidation (EWC)

**Key Idea**: Protect parameters that are statistically important to previous tasks according to the Fisher Information Matrix.

**Algorithm**:
1. **Train** on task $t$
2. **Compute Fisher Information Matrix** $F$ for the parameters using the task $t$ training data
3. **Apply regularization** in task $t+1$:
   $$\mathcal{L}_{\text{task}} = \mathcal{L}_{\text{CE}} + \frac{\lambda}{2} \sum_j F_j (w_j - w^*_j)^2$$

**Hyperparameter**: $\lambda$ (default: 40.0)

**Pros**:
- Specifically identifies which parameters matter for past tasks
- High theoretical grounding
- Protects critical paths while freeing unimportant weights

**Cons**:
- Requires a post-task pass to compute the Fisher matrix
- Diagonal Fisher approximation can sometimes be insufficient

---

### Method 4: Synaptic Intelligence (SI)

**Key Idea**: Track parameter importance and penalize changes to important weights

**Algorithm**:

1. **Track parameter changes** during each task:
   $$\theta_j^{\text{path}} = \sum_\ell \frac{\partial \mathcal{L}}{\partial w_j} \Delta w_j$$

2. **Compute importance** (omega) after each task:
   $$\omega_j = \frac{(\theta_j^{\text{path}})^2}{|\Delta w_j| + \xi}$$

3. **Apply regularization** in subsequent tasks:
   $$\mathcal{L}_{\text{SI}} = \frac{\lambda}{2} \sum_j \omega_j (w_j - w^*_j)^2$$

**Hyperparameter**: $\lambda$ (default: 0.1)

**Pros**:
- Adaptive importance weighting
- Better retention than L2 regularization
- Interpretable parameter importance

**Cons**:
- More complex implementation
- Additional tracking overhead
- Sensitivity to importance threshold

---

### Method 5: PackNet (Progressive Pruning)

**Key Idea**: Learn task-specific subnetworks by pruning and freezing weights

**Algorithm**:

1. **Train** on task $t$ with standard SGD
2. **Prune** unimportant free weights (70% by default)
3. **Freeze** the currently important weights (30%) for all previous tasks
4. **Allocate** the pruned (now free) weights to new tasks
5. **Mask** application during inference to use task-specific subnetwork

**Hyperparameter**: `keep_ratio` (default: 0.30 - keeps 30% per task)

**Pros**:
- Prevents catastrophic forgetting by freezing
- Parameter sharing across tasks
- Clear task-specific boundaries

**Cons**:
- Reduced model capacity per task
- Requires task identity at inference
- May not be suitable for fine-tuning scenarios

---

### Method 6: Experience Replay (ER)

**Key Idea**: Store and replay samples from previous tasks during training

**Algorithm**:

```
Buffer = empty
for task in tasks:
    for epoch in epochs:
        for (x, y) in current_task:
            # Sample from buffer
            (x_replay, y_replay) = buffer.sample(batch_size // 2)
            # Combine and train
            model.train(cat(x, x_replay), cat(y, y_replay))
    # Add current task data to buffer
    buffer.add(current_task_data, reservoir_sampling=True)
```

**Hyperparameter**: `buffer_capacity` (default: 2000), `batch_balancing` (1:1 ratio)

**Pros**:
- Reservoir sampling ensures equal representation of all past tasks
- Balanced 1:1 batch ratios prevent recency bias
- Can achieve near-optimal performance
- Simple to implement

**Cons**:
- High memory requirements
- Computational cost of replaying
- May not scale to large datasets

---

### Method 7: Naive Rehearsal

**Key Idea**: Simple FIFO buffer without reservoir sampling or batch balancing.

**Algorithm**:

Similar to experience replay but with:
- Simpler replacement strategy (first-in-first-out)
- No importance-based sampling
- Fixed buffer size

**Pros**:
- Extremely simple
- Low memory overhead compared to full storing
- Provides baseline for replay methods

**Cons**:
- Uniform preservation (no bias toward important samples)
- Still requires memory
- May not emphasize critical samples

---

## Evaluation Metrics

### 1. Accuracy on Task $t$ after Task $T$

$$A_{t,T} = \frac{\text{correct predictions on task } t}{\text{total samples in task } t}$$

Reported as: Accuracy matrix $A$ where $A[t,T]$ = accuracy on task $t$ after learning task $T$.

### 2. Backward Transfer (Forgetting)

$$\text{Forgetting}_t = A_{t,t} - A_{t,T}$$

Measures accuracy drop on task $t$ after learning subsequent tasks.

Average forgetting:
$$\bar{F} = \frac{1}{|T|-1} \sum_{t=1}^{|T|-1} \text{Forgetting}_t$$

### 3. Forward Transfer

$$\text{Forward Transfer}_T = A_{T, T} - A_{T, T-1}$$

Measures whether learning task $T$ benefits from knowledge of task $T-1$ (usually negative).

### 4. Training Time

$$\text{Time}_t = \text{elapsed time for task } t \text{ training}$$

Tracks computational overhead of each method.

---

## Experimental Setup

### Hyperparameters (Fixed across all methods)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Random Seed | 42 | Reproducibility |
| Learning Rate | 0.01 | Standard SGD |
| Optimizer | SGD (no momentum) | Simple, stable |
| Batch Size | 64 | Balanced |
| Epochs per Task | 5 | Sufficient convergence |
| Loss Function | Cross-Entropy | Standard classification |
| Input Size | 784 (28×28) | Flattened MNIST |
| Hidden Size | 256 | Mid-layer capacity |
| Output Size | 10 | MNIST digit classes |

### Method-Specific Parameters

| Method | Parameter | Value | Rationale |
|--------|-----------|-------|-----------|
| L2 Reg | $\lambda$ | 0.01 | Balance regularization |
| EWC | $\lambda$ | 40.0 | From Bayesian optimization |
| SI | $\lambda$ | 0.1 | Stronger (weighted) regularization |
| PackNet | Keep Ratio | 30% | Leaves capacity for later tasks |
| ER | Buffer Capacity | 2000 | Reservoir sampled, balanced training |
| Naive Rehearsal | Buffer Size | 2000 | Same limit as ER but simple FIFO |

---

## Implementation Details

### Reproducibility Guarantees

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- **Same seed** across all runs ensures deterministic results
- **GPU determinism** enabled for reproducibility on CUDA
- **Benchmark disabled** to prioritize determinism over performance

### Data Pipeline

1. **Download**: MNIST automatically downloaded on first run (stored in `./data/`)
2. **Normalization**: ToTensor() normalizes to [0, 1]
3. **No augmentation**: Clean, deterministic setup
4. **Batch sampling**: Shuffled training, sequential testing

---

## Expected Results Summary

Based on the literature and our implementation:

| Method | Avg Forgetting | Avg Accuracy | Time/Task | Notes |
|--------|---|---|---|---|
| Baseline | ~40-50% | 70-80% | Fastest | Clear forgetting |
| L2 Reg | ~15-25% | 80-85% | +5% | Good balance |
| EWC | ~5-15% | 85-90% | +20% | Fisher matrix computation overhead |
| SI | ~10-20% | 82-88% | +10% | Better than L2 |
| PackNet | ~0-5% | 80-87% | +15% | Near-zero forgetting on frozen weights |
| ER | ~2-5% | 88-93% | +25% | Best accuracy, balanced reservoir memory |
| Naive Rehearsal | ~10-20% | 75-80% | +20% | Suffers from FIFO unbalancing |

*Exact values depend on dataset, hyperparameters, and randomness. Re-runs may show minor variation.*

---

## References & Related Work

1. **Elastic Weight Consolidation**: Kirkpatrick, J., et al. (2017). "Overcoming catastrophic forgetting in neural networks." PNAS.

2. **Synaptic Intelligence**: Zenke, F., Poole, B., & Ganguli, S. (2017). "Continual Learning Through Synaptic Intelligence." ICML.

3. **PackNet**: Mallya, A., & Lazebnik, S. (2018). "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning." CVPR.

4. **Experience Replay**: Lin, L. J. (1992). "Self-improving reactive agents based on reinforcement learning, planning and teaching." Machine Learning. Rebuffi, S. A. et al. (2017) "iCaRL".

5. **Continual Learning**: Rusu, A. A., et al. (2016). "Progressive Neural Networks." NIPS. Serra, J., et al. (2018). "Continual Learning with Deep Generative Replay." NIPS.

6. **Catastrophic Forgetting**: McCloskey, M., & Cohen, N. J. (1989). "Catastrophic Interference in Connectionist Networks: The Temporal Instability of Task Learning." Psychological Review.

---

## Running Individual Experiments

### Quick Start (Single Experiment)

```bash
# Start Jupyter
jupyter notebook notebooks/

# Open and run Baseline_task.ipynb to see:
# - How catastrophic forgetting develops
# - Forgetting metrics visualization
# - Performance drop across tasks
```

### Comparison Study

Run these in sequence to compare methods:

1. `Baseline_task.ipynb` (no protection)
2. `L2_permutated.ipynb` (light regularization)
3. `EWC_permutated.ipynb` (Fisher-based regularization)
4. `SI_permutated.ipynb` (adaptive online regularization)
5. `packnet.ipynb` (architectural protection via pruning)
6. `experience_replay.ipynb` & `naive_rehearsal.ipynb` (memory-based)

Then compare results across final accuracy and forgetting tables.

---

## Troubleshooting

### Results differ on re-run?
- Normal with GPU (determinism is CPU-default)
- Check `set_seed(42)` is called at notebook start
- Ensure PyTorch version matches requirements.txt

### Out of memory?
- Reduce `batch_size` (currently 64)
- Reduce `buffer_capacity` for replay methods
- Use CPU instead of GPU

### Permutation different each time?
- Permutations are generated randomly **after** seed is set (by design for variation)
- Use fixed permutations if strict reproducibility is required

---

**Last Updated**: April 2026  
**Python Version**: 3.10+  
**PyTorch Version**: 2.0.0+
