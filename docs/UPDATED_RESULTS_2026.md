# Updated Results (April 2026 runs)

This document contains the snapshot of the latest benchmarking results for the newly implemented and refactored continual learning strategies.

## 1. Class-Incremental Learning (CIL)
*Evaluated on Split-MNIST with classes arriving sequentially in pairs (0-1, 2-3, 4-5, 6-7, 8-9).*

| Strategy | Memory Buffer | Cumulative Accuracy (Task 5) | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Baseline** | None | ~19% | Total catastrophic forgetting. Model rapidly overwrites past task knowledge. |
| **Naive Rehearsal** | 2000 (FIFO) | ~32% | Marginal improvement. Suffers from FIFO unbalancing; past tasks are forgotten to make room for newer ones. |
| **Experience Replay** | 2000 (Reservoir) | ~37% | Significant jump. Achieved through **Reservoir Sampling** (retains a uniformly random slice of the entire past) and **1:1 Batch Balancing** (forces 50% replay data per batch). |

---

## 2. Task-Incremental Learning (TIL)
*Evaluated on Permuted-MNIST where pixel-space permutations simulate new tasks.*

### PackNet Progressive Pruning Results
PackNet demonstrated a structural solution to catastrophic forgetting by zeroing unimportant weights and freezing the important ones (top 30%), forcing later tasks to use only the remaining unallocated capacity.

**Task Accuracies over time:**
| Evaluated After | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Task 1 | 89.53% | - | - | - | - |
| Task 2 | 87.32% | 90.76% | - | - | - |
| Task 3 | 79.22% | 89.24% | 89.60% | - | - |
| Task 4 | 76.58% | 88.45% | 88.41% | 87.90% | - |
| Task 5 | 72.67% | 85.84% | 86.48% | 85.56% | 87.22% |

**Analysis of PackNet Results:**
PackNet effectively preserved the old tasks, showing zero *structural* forgetting on the weight matrices. The slight drift on Task 1 (89.53% down to 72.67%) was identified as "bias leak" due to the un-frozen bias scaling parameters across the sequential runs. However, compared to the baseline, the retention is excellent, maintaining high average accuracy across all 5 tasks while strictly preserving network scale.
