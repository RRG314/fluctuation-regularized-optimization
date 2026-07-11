# Public API

The public import package is `fluctuation_opt`.

```python
from fluctuation_opt import (
    LifshitzSwarm,
    ZeroPointOptimizer,
    GradientPressureBalancer,
)
```

## `LifshitzSwarm`

Derivative-free minimizer with Lifshitz-style pair attraction between
fitness-graded candidate solutions.

Use when:

- the objective is expensive or non-differentiable;
- derivative-free search is appropriate;
- robustness matters more than machine-precision final convergence.

Minimal example:

```python
import torch
from fluctuation_opt import LifshitzSwarm

def sphere(x):
    return (x ** 2).sum(dim=-1)

swarm = LifshitzSwarm([(-5.0, 5.0)] * 3, n_particles=24, seed=0)
result = swarm.minimize(sphere, max_iter=120)
print(result["fun"], result["x"])
```

## `ZeroPointOptimizer`

PyTorch optimizer that applies zero-point-smoothed gradients through
antithetic parameter perturbations and an Adam-style update.

Use when:

- differentiable training is available;
- flatness or perturbation robustness matters;
- matched gradient-evaluation budgets are acceptable.

Minimal example:

```python
import torch
from fluctuation_opt import ZeroPointOptimizer

theta = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
opt = ZeroPointOptimizer([theta], lr=0.05, sigma=0.01, seed=0)

for _ in range(250):
    opt.step(lambda: (theta ** 2).sum())
```

## `GradientPressureBalancer`

Adaptive loss weighting for multi-term losses. It measures per-term gradient
pressure and reweights terms so one term does not dominate training.

Use when:

- losses combine residual, boundary, initial-condition, or other competing
  objectives;
- term gradients differ by orders of magnitude;
- PINN training is unstable with fixed weights.

## Core modules

`fluctuation_opt.core` contains:

- `lifshitz`: reflectivity maps and pairwise Lifshitz-style forces;
- `matsubara`: quantum annealing schedules and thermal variance;
- `spectral`: Hessian-vector-product spectral estimators and ZPE diagnostics.
