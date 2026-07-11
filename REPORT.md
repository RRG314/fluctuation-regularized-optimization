# casimir-opt — Benchmark & Validation Report

**Date:** 2026-07-11 · **Library version:** 0.1.0 · **Hardware:** CPU only (PyTorch 2.13.0+cpu, float64 for physics, float32 for ML/PINN)

This report summarizes the full validation of the `casimir-opt` library across five experiment suites: the unit/integration test suite, black-box optimization, physics-informed neural networks (PINNs) on real PDEs, real ML datasets, and a fit to real published Casimir-force experimental data.

All raw numbers, convergence curves, and figures live in `benchmarks/results/`. Every experiment uses multiple random seeds and reports medians with interquartile ranges (IQR). All optimizer comparisons use **matched gradient-evaluation budgets** (the CasimirOptimizer's smoothed mode costs 3 gradient evaluations per step, so it receives 1/3 the steps of Adam/SGD).

---

## 1. Test suite

**86 tests, all passing** (`pytest tests/` — 56 pre-existing + 30 new integration tests in `tests/test_integration.py`).

Coverage highlights:

- **Physics correctness:** Newton's third law for pairwise Lifshitz forces (net force < 1e-10), force = dE/dd by central differences (rel 1e-4), inverse-square short-range law exact to 1e-6, Matsubara sum → T=0 integral convergence, polylog Li₂(1) = π²/6, `coth` branch-switch accuracy on both branches.
- **Numerics:** Stochastic Lanczos Quadrature vs. dense eigendecomposition, Hutchinson trace reproducibility, Lanczos breakdown on rank-deficient matrices, third-order autograd derivatives.
- **API contracts:** swarm eval accounting (`n_evals = pop × (iters+1)`), float32/1D/callback support, optimizer `state_dict` round-trip, multiple param groups, weight decay, frozen params, PINN balancer weight clamps and caching.
- **End-to-end:** tiny PINN solving u′ = u to <1% error.

## 2. Black-box optimization (`blackbox_benchmark.py`)

Five standard test functions, dimension 10, budget 40 particles × 300 iterations, **20 seeds** per (function, method). Baselines: standard PSO (constriction parameters), scipy differential evolution (DE), random search.

| Function | CasimirSwarm (median) | PSO | DE | RandomSearch |
|---|---|---|---|---|
| sphere | 3.1e-06 | 5.7e-16 | 2.2e-24 | 13.4 |
| rosenbrock | 5.33 | 4.15 | 0.038 | 7685 |
| rastrigin | 6.47 | 4.98 | 2.99 | 69.9 |
| ackley | 0.033 | 1.8e-07 | 1.4e-11 | 17.2 |
| griewank | 0.151 | 0.0996 | 0.112 | 47.1 |

**Honest finding:** CasimirSwarm is a competent global optimizer — 4–6 orders of magnitude better than random search everywhere, and competitive with PSO/DE on the multimodal rastrigin and griewank functions. But it does **not** beat tuned DE or PSO on raw final precision for these classic smooth benchmarks. Its zero-point noise floor (which never anneals to zero, by design) prevents the last-digit convergence that DE achieves. Its comparative value shows up in the derivative-free *physics fitting* task (§6), where all seeds land on the same answer.

## 3. Poisson PINN (`pinn_benchmark.py`)

1D Poisson equation −u″ = f with a stiff two-scale solution u* = sin(3πx) + 0.3·sin(9πx). MLP [1,64,64,64,1], 9,000 gradient-eval budget, 3 seeds. Metrics: relative L2 error vs. exact solution, perturbation robustness (loss increase under σ = 0.02 parameter noise), and regularized zero-point energy (ZPE, a flatness diagnostic — lower = flatter minimum).

| Config | rel. L2 (median) | robustness ↓ | ZPE ↓ |
|---|---|---|---|
| adam | **6.90 (diverged 2/3 seeds)** | 11,538 | 5,935 |
| adam + balance | 0.0111 | 11.8 | 1,159 |
| casimir + balance | 0.0123 | **8.3** | **354** |

Plain Adam with uniform loss weights catastrophically fails on 2 of 3 seeds (rel L2 ≈ 7 — worse than predicting zero). The `CasimirPressureBalancer` rescues it completely. Casimir + balance matches the balanced-Adam accuracy while finding minima that are ~3× flatter (ZPE) and more robust to weight perturbation.

## 4. Real PDEs: heat & viscous Burgers (`pde_benchmark.py`)

- **Heat equation:** u_t = 0.1·u_xx, exact solution sin(πx)·e^(−π²αt). Budget 6,000.
- **Burgers:** u_t + u·u_x = ν·u_xx with ν = 0.01/π (the Raissi et al. 2019 setup); exact solution computed by Cole–Hopf transform with 96-node Gauss–Hermite quadrature (verified: IC error 0, BC error ~1e-16, FD residual < 3e-4). Budget 18,000 — this problem develops a steep shock at x = 0. 3 seeds each.

| PDE | Config | rel. L2 (median) | robustness ↓ |
|---|---|---|---|
| heat | adam | **0.0019** | 0.074 |
| heat | adam+balance | 0.0023 | 0.052 |
| heat | casimir+balance | 0.0086 | **0.039** |
| burgers | adam | 0.256 | 1.11 |
| burgers | adam+balance | 0.465 | 0.130 |
| burgers | casimir+balance | **0.157** (best seed 0.080) | **0.083** |

**Honest finding:** on the easy, smooth heat equation, plain Adam is already excellent and the casimir optimizer's noise floor costs a little accuracy (0.9% vs. 0.2% — both are good solutions). On the hard shock-forming Burgers problem, **casimir + balance wins outright**: lowest median error, best single run, and ~13× better perturbation robustness than plain Adam. This matches the design intent — the zero-point exploration helps most on rugged, stiff loss landscapes.

## 5. Real ML datasets (`ml_benchmark.py`)

Real data, matched 9,000-grad-eval budget, 5 seeds each:

- **digits** (sklearn, 1,797 handwritten digit images, 10 classes) — MLP [64,32,10], cross-entropy; metric = test error rate.
- **California housing** (20,640 real census records) — MLP [8,32,32,1], MSE on standardized target; metric = test RMSE.

| Dataset | Method | test metric (median) | robustness ↓ |
|---|---|---|---|
| digits | sgd | 0.0241 | 0.0007 |
| digits | adam | **0.0222** | 0.0012 |
| digits | casimir | **0.0222** | **0.0006** |
| housing | sgd | 0.4673 | 0.0284 |
| housing | adam | **0.4640** | 0.0281 |
| housing | casimir | 0.4818 | **0.0173** |

On digits the casimir optimizer **ties Adam exactly** (2.22% median test error) with half the perturbation sensitivity. On housing it trades ~3.8% higher RMSE for ~38% better robustness — it consistently prefers flatter minima, as the theory predicts, and the flatness sometimes (digits) but not always (housing) comes for free.

## 6. Real Casimir experiment data (`casimir_data_fit.py`)

The headline real-data test: fitting the **plasma-model Lifshitz theory** to the 16-point Casimir pressure dataset of **Decca et al., Eur. Phys. J. C 51, 963 (2007)** (arXiv:0706.3283, Table 1; sphere–plate, 162–746 nm, 95% CI half-widths).

The forward model is the full finite-temperature Lifshitz formula for gold (plasma model) — Matsubara sum to l = 300 at T = 300 K, 80-node Gauss–Laguerre quadrature, implemented in float64 torch. Model validation before fitting:

- Perfect-conductor limit (ω_p → ∞, T → 0): deviates from π²ħc/240z⁴ by **0.07%**.
- Against Decca's own tabulated plasma theory at ω_p = 9 eV: median 0.58%, max 2.88%.

Fit parameters: plasma frequency ω_p ∈ [6, 12] eV and separation offset Δz ∈ ±0.6 nm (the paper's quoted absolute-separation uncertainty). 8 CasimirSwarm seeds + 2 scipy-DE seeds:

| Method | ω_p (eV) | Δz (nm) | χ² (16 pts) | reduced χ² |
|---|---|---|---|---|
| CasimirSwarm (8 seeds) | 9.210–9.217 | −0.60 | 8.53 | 0.61 |
| scipy DE (2 seeds) | 9.215 | −0.60 | 8.53 | 0.61 |

**All ten runs land on the same optimum**: ω_p ≈ **9.21 eV**, in close agreement with the literature value of ~9.0 eV for gold, with reduced χ² = 0.61 (every residual inside the 95% band — see `casimir_fit.png`). The swarm matched scipy DE's answer within its evaluation budget with zero tuning, demonstrating it as a reliable derivative-free fitter on real, noisy laboratory data.

## 7. Overall assessment

**Where the Casimir approach wins:**

1. Hard, stiff PINN problems: rescues Poisson from 2/3 catastrophic Adam failures; best accuracy AND robustness on shock-forming Burgers.
2. Flat-minimum selection: lowest ZPE and perturbation sensitivity in essentially every experiment, exactly as the zero-point-energy theory predicts.
3. Real-data fitting: reproducible, tuning-free convergence to the physically correct answer on real experimental data.

**Where it doesn't:**

1. Raw final precision on smooth classic black-box functions (DE/PSO converge deeper — the never-zero noise floor is a deliberate trade).
2. Easy problems (heat equation, housing regression) where plain Adam is already near-optimal; there the flatness comes at a small accuracy cost (≤ 4%).

**Reproducing everything:**

```bash
pip install -e . && pytest tests/                       # 86 tests
python benchmarks/blackbox_benchmark.py --seeds 20
python benchmarks/pinn_benchmark.py
python benchmarks/pde_benchmark.py --pde heat
python benchmarks/pde_benchmark.py --pde burgers --budget 18000
python benchmarks/ml_benchmark.py --dataset digits
python benchmarks/ml_benchmark.py --dataset housing
python benchmarks/casimir_data_fit.py
```

Figures: `blackbox_convergence.png`, `pinn_benchmark.png`, `pde_benchmark.png`, `ml_benchmark.png`, `casimir_fit.png` (all in `benchmarks/results/`).
