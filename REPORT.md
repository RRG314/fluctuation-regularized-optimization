# fluctuation-regularized-optimization — Benchmark & Validation Report

**Date:** 2026-07-11 · **Library version:** 0.1.0 · **Hardware:** CPU only (PyTorch 2.13.0+cpu, float64 for physics, float32 for ML/PINN)

This report summarizes the validation of the `fluctuation_opt` Python package in the `fluctuation-regularized-optimization` repository across five experiment suites: the unit/integration test suite, black-box optimization, physics-informed neural networks (PINNs) on real PDEs, real ML datasets, and a fit to real published Casimir-force experimental data.

All raw numbers, convergence curves, and figures live in `benchmarks/results/`. Every experiment uses multiple random seeds and reports medians with interquartile ranges (IQR). All optimizer comparisons use **matched gradient-evaluation budgets** (the ZeroPointOptimizer's smoothed mode costs 3 gradient evaluations per step, so it receives 1/3 the steps of Adam/SGD).

---

## 1. Test suite

**89 tests, all passing** (`pytest tests/`).

Coverage highlights:

- **Physics correctness:** Newton's third law for pairwise Lifshitz forces (net force < 1e-10), force = dE/dd by central differences (rel 1e-4), inverse-square short-range law exact to 1e-6, Matsubara sum → T=0 integral convergence, polylog Li₂(1) = π²/6, `coth` branch-switch accuracy on both branches.
- **Numerics:** Stochastic Lanczos Quadrature vs. dense eigendecomposition, Hutchinson trace reproducibility, Lanczos breakdown on rank-deficient matrices, third-order autograd derivatives.
- **API contracts:** swarm eval accounting (`n_evals = pop × (iters+1)`), float32/1D/callback support, optimizer `state_dict` round-trip, multiple param groups, weight decay, frozen params, PINN balancer weight clamps and caching.
- **End-to-end:** tiny PINN smoke test for the coupled MLP, balancer, and zero-point optimizer path.

## 2. Black-box optimization (`blackbox_benchmark.py`)

Five standard test functions, dimension 10, budget 40 particles × 300 iterations, **20 seeds** per (function, method). Baselines: standard PSO (constriction parameters), scipy differential evolution (DE), random search.

| Function | LifshitzSwarm (median) | PSO | DE | RandomSearch |
|---|---|---|---|---|
| sphere | 1.2e-07 | 5.7e-16 | 2.2e-24 | 13.4 |
| rosenbrock | 4.53 | 4.15 | 0.038 | 7685 |
| rastrigin | 6.47 | 4.98 | 2.99 | 69.9 |
| ackley | 0.0079 | 1.8e-07 | 1.4e-11 | 17.2 |
| griewank | 0.117 | 0.0996 | 0.112 | 47.1 |

**Honest finding:** LifshitzSwarm is a competent global optimizer — many orders of magnitude better than random search everywhere, improved substantially on sphere and ackley after the partial-quench refinement change, and is now close to PSO/DE on griewank. But it does **not** beat tuned DE or PSO on raw final precision for these classic smooth benchmarks. Its zero-point noise floor and population dynamics still prevent the last-digit convergence that DE achieves. Its comparative value shows up more clearly in robust derivative-free fitting and rugged/stiff settings than in machine-precision toy functions.

## 3. Poisson PINN (`pinn_benchmark.py`)

1D Poisson equation −u″ = f with a stiff two-scale solution u* = sin(3πx) + 0.3·sin(9πx). MLP [1,64,64,64,1], 9,000 gradient-eval budget, 3 seeds. Metrics: relative L2 error vs. exact solution, perturbation robustness (loss increase under σ = 0.02 parameter noise), and regularized zero-point energy (ZPE, a flatness diagnostic — lower = flatter minimum).

| Config | rel. L2 (median) | robustness ↓ | ZPE ↓ |
|---|---|---|---|
| adam | 6.90 (diverged 2/3 seeds) | 12,582 | 106 |
| adam + balance | **0.0019** | 11.5 | 104 |
| zero-point + pressure | 0.0172 | **8.45** | 1,024 |

Plain Adam with uniform loss weights catastrophically fails on 2 of 3 seeds (rel L2 ≈ 7 — worse than predicting zero). The `GradientPressureBalancer` rescues it completely and is the accuracy winner in this run. Zero-point smoothing plus pressure balancing is less accurate on this Poisson setup, but gives the best perturbation robustness. Its ZPE diagnostic is not lower here, so the honest claim for this benchmark is robustness, not universal flatness by every diagnostic.

## 4. Real PDEs: heat & viscous Burgers (`pde_benchmark.py`)

- **Heat equation:** u_t = 0.1·u_xx, exact solution sin(πx)·e^(−π²αt). Budget 6,000.
- **Burgers:** u_t + u·u_x = ν·u_xx with ν = 0.01/π (the Raissi et al. 2019 setup); exact solution computed by Cole–Hopf transform with 96-node Gauss–Hermite quadrature (verified: IC error 0, BC error ~1e-16, FD residual < 3e-4). Budget 18,000 — this problem develops a steep shock at x = 0. 3 seeds each.

| PDE | Config | rel. L2 (median) | robustness ↓ |
|---|---|---|---|
| heat | adam | **0.0019** | 0.074 |
| heat | adam+balance | 0.0023 | 0.052 |
| heat | zero-point+pressure | 0.0099 | **0.042** |
| burgers | adam | 0.285 | 1.07 |
| burgers | adam+balance | 0.321 | 0.115 |
| burgers | zero-point+pressure | **0.157** (best seed 0.093) | **0.086** |

**Honest finding:** on the easy, smooth heat equation, plain Adam is already excellent and the zero-point optimizer's noise floor costs a little accuracy (0.9% vs. 0.2% — both are good solutions). On the hard shock-forming Burgers problem, **zero-point + pressure balancing wins outright**: lowest median error, best single run, and ~13× better perturbation robustness than plain Adam. This matches the design intent — the zero-point exploration helps most on rugged, stiff loss landscapes.

## 5. Real ML datasets (`ml_benchmark.py`)

Real data, matched 9,000-grad-eval budget, 5 seeds each:

- **digits** (sklearn, 1,797 handwritten digit images, 10 classes) — MLP [64,32,10], cross-entropy; metric = test error rate.
- **California housing** (20,640 real census records) — MLP [8,32,32,1], MSE on standardized target; metric = test RMSE.

| Dataset | Method | test metric (median) | robustness ↓ |
|---|---|---|---|
| digits | sgd | 0.0241 | 0.0007 |
| digits | adam | **0.0222** | 0.0012 |
| digits | zero-point | **0.0222** | **0.0008** |
| housing | sgd | 0.4698 | 0.0180 |
| housing | adam | **0.4688** | 0.0198 |
| housing | zero-point | 0.4860 | **0.0150** |

On digits the zero-point optimizer **ties Adam exactly** (2.22% median test error) with lower perturbation sensitivity. On housing it trades ~3.7% higher RMSE for ~24% better robustness. The flatness/robustness benefit is visible, but it does not come for free on every dataset.

## 6. Real Casimir experiment data (`casimir_data_fit.py`)

The headline real-data test: fitting the **plasma-model Lifshitz theory** to the 16-point Casimir pressure dataset of **Decca et al., Eur. Phys. J. C 51, 963 (2007)** (arXiv:0706.3283, Table 1; sphere–plate, 162–746 nm, 95% CI half-widths).

The forward model is the full finite-temperature Lifshitz formula for gold (plasma model) — Matsubara sum to l = 300 at T = 300 K, 80-node Gauss–Laguerre quadrature, implemented in float64 torch. Model validation before fitting:

- Perfect-conductor limit (ω_p → ∞, T → 0): deviates from π²ħc/240z⁴ by **0.07%**.
- Against Decca's own tabulated plasma theory at ω_p = 9 eV: median 0.58%, max 2.88%.

Fit parameters: plasma frequency ω_p ∈ [6, 12] eV and separation offset Δz ∈ ±0.6 nm (the paper's quoted absolute-separation uncertainty). 8 LifshitzSwarm seeds + 2 scipy-DE seeds:

| Method | ω_p (eV) | Δz (nm) | χ² (16 pts) | reduced χ² |
|---|---|---|---|---|
| LifshitzSwarm (8 seeds) | 9.210–9.217 | −0.60 | 8.53 | 0.61 |
| scipy DE (2 seeds) | 9.215 | −0.60 | 8.53 | 0.61 |

**All ten runs land on the same optimum**: ω_p ≈ **9.21 eV**, in close agreement with the literature value of ~9.0 eV for gold, with reduced χ² = 0.61 (every residual inside the 95% band — see `casimir_fit.png`). The swarm matched scipy DE's answer within its evaluation budget with zero tuning, demonstrating it as a reliable derivative-free fitter on real, noisy laboratory data.

## 7. Overall assessment

**Where the fluctuation-regularized approach wins:**

1. Hard, stiff PINN problems: pressure balancing rescues Poisson from 2/3 catastrophic Adam failures; zero-point + pressure gives the best accuracy and robustness on shock-forming Burgers.
2. Perturbation robustness: zero-point variants often reduce sensitivity to parameter noise, although the ZPE diagnostic is mixed and should not be overread as universally lower.
3. Real-data fitting: reproducible, tuning-free convergence to the physically correct answer on real experimental data.

**Where it doesn't:**

1. Raw final precision on smooth classic black-box functions (DE/PSO converge deeper — the never-zero noise floor is a deliberate trade).
2. Easy problems (heat equation, housing regression) where plain Adam is already near-optimal; there the flatness comes at a small accuracy cost (≤ 4%).

**Reproducing everything:**

```bash
pip install -e . && pytest tests/                       # 89 tests
python benchmarks/blackbox_benchmark.py --seeds 20
python benchmarks/pinn_benchmark.py
python benchmarks/pde_benchmark.py --pde heat
python benchmarks/pde_benchmark.py --pde burgers --budget 18000
python benchmarks/ml_benchmark.py --dataset digits
python benchmarks/ml_benchmark.py --dataset housing
python benchmarks/casimir_data_fit.py
```

Figures: `blackbox_convergence.png`, `pinn_benchmark.png`, `pde_benchmark.png`, `ml_benchmark.png`, `casimir_fit.png` (all in `benchmarks/results/`).
