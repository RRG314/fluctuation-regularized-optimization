# Reproducibility

This document describes how to reproduce the checked-in results and how to
report new ones.

## Environment

Recommended:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Record:

- operating system;
- CPU/GPU;
- Python version;
- PyTorch version;
- command used;
- seed count.

## Core validation

```bash
pytest
python examples/quickstart.py
```

## Full benchmark commands

```bash
python benchmarks/blackbox_benchmark.py --seeds 20
python benchmarks/pinn_benchmark.py
python benchmarks/pde_benchmark.py --pde heat
python benchmarks/pde_benchmark.py --pde burgers --budget 18000
python benchmarks/ml_benchmark.py --dataset digits
python benchmarks/ml_benchmark.py --dataset housing
python benchmarks/casimir_data_fit.py
```

Some benchmarks are CPU-expensive. Do not update checked-in result files unless
you know which command generated them.

## Result files

Saved artifacts live in `benchmarks/results/`.

| File | Meaning |
| --- | --- |
| `blackbox_results.csv` | Classic black-box function summary. |
| `pinn_results.csv` | Poisson PINN per-seed results. |
| `pde_results.csv` | Heat and Burgers PINN per-seed results. |
| `ml_results.csv` | Real-data ML per-seed results. |
| `casimir_fit_results.csv` | Real Casimir-pressure fit results. |

## Reporting changed results

In a pull request, include:

- the command;
- dependency versions;
- whether the result replaces or supplements existing output;
- a short explanation of meaningful changes;
- failures or regressions.

Negative results should remain visible.
