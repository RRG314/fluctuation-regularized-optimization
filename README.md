# casimir-opt

Casimir-effect-inspired optimization experiments and utilities.

This repository contains a research prototype, not a published PyPI package.
Install it from source when you want to reproduce the tests, run the
benchmarks, or inspect the implementation.

## What this is

`casimir-opt` explores whether mathematical structures from Casimir/Lifshitz
physics can be used as useful optimization mechanisms:

| Component | Use it for | Main idea |
| --- | --- | --- |
| `CasimirSwarm` | Derivative-free black-box search | Candidate solutions act like partially reflective mirrors coupled by short-range Lifshitz-style attraction. |
| `CasimirOptimizer` | PyTorch training | Optimizes a zero-point-smoothed effective loss, biasing training toward flatter minima. |
| `CasimirPressureBalancer` | PINNs and multi-term losses | Reweights loss terms by measured gradient pressure so one term does not dominate the others. |
| `casimir_opt.core` | Physics/numerics utilities | Matsubara schedules, Lifshitz couplings, stochastic Lanczos quadrature, and zero-point-energy diagnostics. |

The strongest current evidence is for robustness and stiff/rugged problems,
especially PINNs. This is not claimed to beat tuned standard optimizers on
every benchmark.

## Current validation status

The repository includes:

| Artifact | Location |
| --- | --- |
| Validation report | `REPORT.md` |
| Paper source and compiled PDF | `paper/paper.tex`, `paper/paper.pdf` |
| Unit/integration tests | `tests/` |
| Benchmark scripts | `benchmarks/` |
| Raw benchmark outputs and figures | `benchmarks/results/` |
| Quickstart example | `examples/quickstart.py` |

The report summarizes the existing generated results. The headline results are:

| Experiment | Honest result |
| --- | --- |
| Classic black-box functions | `CasimirSwarm` beats random search, but tuned PSO/DE reach better final precision on smooth functions. |
| Poisson PINN | Pressure balancing prevents Adam failures; Casimir + balance gives similar accuracy with better flatness/robustness. |
| Heat equation PINN | Plain Adam is most accurate; Casimir + balance is more robust but less accurate. |
| Burgers equation PINN | Casimir + balance gives the best median error and best robustness in the included runs. |
| Digits classification | Casimir ties Adam median test error and improves perturbation robustness. |
| California housing | Casimir is less accurate than Adam but more robust. |
| Real Casimir data fit | `CasimirSwarm` matches scipy differential evolution and recovers a gold plasma frequency near `9.21 eV`; the fitted separation offset is pinned at the `-0.60 nm` bound and should be treated as a caveat. |

## Install from source

Use Python 3.11 or another PyTorch-supported Python version.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

For a lighter install without benchmark extras:

```bash
python -m pip install -e .[test]
```

## Run checks

```bash
pytest
python examples/quickstart.py
```

## Reproduce benchmark artifacts

The full benchmark suite can take a while on CPU.

```bash
python benchmarks/blackbox_benchmark.py --seeds 20
python benchmarks/pinn_benchmark.py
python benchmarks/pde_benchmark.py --pde heat
python benchmarks/pde_benchmark.py --pde burgers --budget 18000
python benchmarks/ml_benchmark.py --dataset digits
python benchmarks/ml_benchmark.py --dataset housing
python benchmarks/casimir_data_fit.py
```

Then compare the regenerated files in `benchmarks/results/` against the
tables in `REPORT.md`.

## Repository map

```text
casimir_opt/
  core/              # Lifshitz, Matsubara, and spectral utilities
  swarm.py           # CasimirSwarm black-box optimizer
  torch_optim.py     # CasimirOptimizer PyTorch optimizer
  pinn.py            # PINN pressure balancer and helpers
benchmarks/          # Reproducible experiment scripts
benchmarks/results/  # Raw CSV/JSON results and generated figures
tests/               # Unit and integration tests
paper/               # Paper source and compiled PDF
```

## Scope and caveats

- This is research code at version `0.1.0`.
- It is not currently published on PyPI.
- The strongest current use case is stiff/rugged optimization, especially
  PINN-style training and noisy derivative-free physics fitting.
- On smooth toy functions where machine-precision convergence is the goal,
  the deliberate zero-point noise floor is a disadvantage.
- The included results should be treated as reproducible experiments, not as
  proof of universal optimizer superiority.
