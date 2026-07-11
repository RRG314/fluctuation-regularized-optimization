# fluctuation-regularized-optimization

[![tests](https://github.com/RRG314/fluctuation-regularized-optimization/actions/workflows/tests.yml/badge.svg)](https://github.com/RRG314/fluctuation-regularized-optimization/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

Research code for fluctuation-regularized optimization: Lifshitz-style swarm
search, zero-point-smoothed PyTorch optimization, and gradient-pressure
balancing for PINNs and other multi-term losses.

This repository contains a research prototype, not a published PyPI package.
Install it from source when you want to reproduce the tests, run the
benchmarks, or inspect the implementation.

## What this is

`fluctuation-regularized-optimization` explores three concrete optimization
mechanisms: Lifshitz-style swarm coupling, zero-point-smoothed gradient
updates, and gradient-pressure loss balancing.
The Python import remains `fluctuation_opt`.

The name is intentionally descriptive. This is not "a Casimir optimizer" and
not a claim that the code is a complete physical simulator of the Casimir
effect. Casimir/Lifshitz physics is one source of the mathematical analogy.

| Component | Use it for | Main idea |
| --- | --- | --- |
| `LifshitzSwarm` | Derivative-free black-box search | Candidate solutions act like partially reflective mirrors coupled by short-range Lifshitz-style attraction. |
| `ZeroPointOptimizer` | PyTorch training | Optimizes a zero-point-smoothed effective loss, biasing training toward flatter minima. |
| `GradientPressureBalancer` | PINNs and multi-term losses | Reweights loss terms by measured gradient pressure so one term does not dominate the others. |
| `fluctuation_opt.core` | Physics/numerics utilities | Matsubara schedules, Lifshitz couplings, stochastic Lanczos quadrature, and zero-point-energy diagnostics. |

The strongest current evidence is for robustness and stiff/rugged problems,
especially PINNs. This is not claimed to beat tuned standard optimizers on
every benchmark.

For the actual mechanism mappings, equations, assumptions, and failure modes,
read [docs/mechanisms.md](docs/mechanisms.md).

## Quick links

- [Mechanisms and math](docs/mechanisms.md)
- [Public API](docs/api.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Benchmark report](REPORT.md)
- [Contributing](CONTRIBUTING.md)
- [Release checklist](docs/release.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)

## Current validation status

The repository includes:

| Artifact | Location |
| --- | --- |
| Validation report | `REPORT.md` |
| Paper source | `paper/paper.tex` |
| Mechanism/math explanation | `docs/mechanisms.md` |
| Unit/integration tests | `tests/` |
| Benchmark scripts | `benchmarks/` |
| Raw benchmark outputs and figures | `benchmarks/results/` |
| Quickstart example | `examples/quickstart.py` |
| Contributor/community docs | `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md` |

The report summarizes the existing generated results. The headline results are:

| Experiment | Honest result |
| --- | --- |
| Classic black-box functions | `LifshitzSwarm` beats random search, but tuned PSO/DE reach better final precision on smooth functions. |
| Poisson PINN | Gradient-pressure balancing prevents Adam failures; zero-point + pressure balancing gives similar accuracy with better flatness/robustness. |
| Heat equation PINN | Plain Adam is most accurate; zero-point + pressure balancing is more robust but less accurate. |
| Burgers equation PINN | Zero-point + pressure balancing gives the best median error and best robustness in the included runs. |
| Digits classification | Zero-point smoothing ties Adam median test error and improves perturbation robustness. |
| California housing | Zero-point smoothing is less accurate than Adam but more robust. |
| Real Casimir data fit | `LifshitzSwarm` matches scipy differential evolution and recovers a gold plasma frequency near `9.21 eV`; the fitted separation offset is pinned at the `-0.60 nm` bound and should be treated as a caveat. |

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
python -m build --wheel
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
fluctuation_opt/
  core/              # Lifshitz, Matsubara, and spectral utilities
  swarm.py           # LifshitzSwarm black-box optimizer
  torch_optim.py     # ZeroPointOptimizer PyTorch optimizer
  pinn.py            # PINN pressure balancer and helpers
benchmarks/          # Reproducible experiment scripts
benchmarks/results/  # Raw CSV/JSON results and generated figures
tests/               # Unit and integration tests
paper/               # Paper source and rebuild note
```

The paper source was updated with the accurate project/API names. A compiled
PDF is not tracked until it can be regenerated from the updated source with a
local LaTeX engine.

## Scope and caveats

- This is research code at version `0.1.0`.
- It is not currently published on PyPI.
- The strongest current use case is stiff/rugged optimization, especially
  PINN-style training and noisy derivative-free physics fitting.
- On smooth toy functions where machine-precision convergence is the goal,
  the deliberate zero-point noise floor is a disadvantage.
- The included results should be treated as reproducible experiments, not as
  proof of universal optimizer superiority.

## Contributing

Contributions are welcome when they are scoped, reproducible, and aligned with
the mechanism-based naming. Start with [CONTRIBUTING.md](CONTRIBUTING.md).

For bugs, feature requests, and benchmark reproduction questions, use the
GitHub issue templates. Pull requests should include tests, docs, or benchmark
evidence appropriate to the change.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). If you use this
repository, cite the repository and describe the exact commit or release used.
