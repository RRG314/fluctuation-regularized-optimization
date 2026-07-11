"""Black-box benchmark: LifshitzSwarm vs PSO, differential evolution, random search.

Protocol: fixed evaluation budget per run (population 40 x 300 iterations),
10 seeds per (function, method), dimension 10.  Reports final best value
(median / IQR) and convergence curves vs evaluations.

Run:  python benchmarks/blackbox_benchmark.py [--dim 10] [--seeds 10] [--device cpu]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fluctuation_opt import LifshitzSwarm  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------- functions
def sphere(X):
    return (X**2).sum(-1)


def rosenbrock(X):
    return (100 * (X[..., 1:] - X[..., :-1] ** 2) ** 2
            + (1 - X[..., :-1]) ** 2).sum(-1)


def rastrigin(X):
    return 10 * X.shape[-1] + (X**2 - 10 * torch.cos(2 * math.pi * X)).sum(-1)


def ackley(X):
    d = X.shape[-1]
    return (-20 * torch.exp(-0.2 * torch.sqrt((X**2).mean(-1)))
            - torch.exp(torch.cos(2 * math.pi * X).mean(-1)) + 20 + math.e)


def griewank(X):
    d = X.shape[-1]
    i = torch.arange(1, d + 1, device=X.device, dtype=X.dtype).sqrt()
    return 1 + (X**2).sum(-1) / 4000 - torch.cos(X / i).prod(-1)


FUNCTIONS = {
    "sphere": (sphere, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "ackley": (ackley, (-32.77, 32.77)),
    "griewank": (griewank, (-600.0, 600.0)),
}


# ---------------------------------------------------------------- baselines
def run_random_search(f, bounds, dim, n_pop, iters, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    lo, hi = bounds
    best, hist = float("inf"), []
    for _ in range(iters + 1):
        X = lo + (hi - lo) * torch.rand(n_pop, dim, generator=g, device=device,
                                        dtype=torch.float64)
        best = min(best, float(f(X).min()))
        hist.append(best)
    return best, hist


def run_pso(f, bounds, dim, n_pop, iters, seed, device):
    """Standard global-best PSO (Clerc constriction-ish parameters)."""
    g = torch.Generator(device=device).manual_seed(seed)
    lo, hi = bounds
    w, c1, c2 = 0.729, 1.494, 1.494
    X = lo + (hi - lo) * torch.rand(n_pop, dim, generator=g, device=device,
                                    dtype=torch.float64)
    V = torch.zeros_like(X)
    fit = f(X)
    P, pf = X.clone(), fit.clone()
    gi = int(fit.argmin())
    G, gf = X[gi].clone(), float(fit[gi])
    hist = [gf]
    vmax = 0.2 * (hi - lo)
    for _ in range(iters):
        r1 = torch.rand(n_pop, dim, generator=g, device=device, dtype=torch.float64)
        r2 = torch.rand(n_pop, dim, generator=g, device=device, dtype=torch.float64)
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
        V = V.clamp(-vmax, vmax)
        X = (X + V).clamp(lo, hi)
        fit = f(X)
        better = fit < pf
        P[better], pf[better] = X[better], fit[better]
        gi = int(pf.argmin())
        if float(pf[gi]) < gf:
            gf, G = float(pf[gi]), P[gi].clone()
        hist.append(gf)
    return gf, hist


def run_differential_evolution(f, bounds, dim, n_pop, iters, seed, device):
    from scipy.optimize import differential_evolution

    hist = []

    def fnp(x):
        return float(f(torch.as_tensor(x, dtype=torch.float64).unsqueeze(0))[0])

    def cb(xk, convergence=None):
        hist.append(fnp(xk))

    res = differential_evolution(
        fnp, [bounds] * dim, popsize=max(1, n_pop // dim), maxiter=iters,
        seed=seed, tol=0, polish=False, callback=cb, init="latinhypercube",
    )
    return float(res.fun), hist


def run_lifshitz(f, bounds, dim, n_pop, iters, seed, device):
    sw = LifshitzSwarm([bounds] * dim, n_particles=n_pop, seed=seed, device=device)
    res = sw.minimize(f, max_iter=iters)
    return res["fun"], res["history"]


METHODS = {
    "LifshitzSwarm": run_lifshitz,
    "PSO": run_pso,
    "DifferentialEvolution": run_differential_evolution,
    "RandomSearch": run_random_search,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = []
    curves = {}
    for fname, (f, bounds) in FUNCTIONS.items():
        for mname, runner in METHODS.items():
            finals, hists = [], []
            t0 = time.time()
            for seed in range(args.seeds):
                best, hist = runner(f, bounds, args.dim, args.pop, args.iters,
                                    seed, args.device)
                finals.append(best)
                hists.append(hist)
            dt = time.time() - t0
            q1, med, q3 = np.percentile(finals, [25, 50, 75])
            rows.append([fname, mname, med, q1, q3, min(finals), dt / args.seeds])
            curves[(fname, mname)] = hists
            print(f"{fname:12s} {mname:22s} median={med:12.4g} "
                  f"IQR=[{q1:.4g}, {q3:.4g}] best={min(finals):.4g} "
                  f"({dt/args.seeds:.1f}s/run)")

    with open(os.path.join(RESULTS, "blackbox_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "method", "median", "q25", "q75", "best",
                    "sec_per_run"])
        w.writerows(rows)

    # ---------------- plots ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = {"LifshitzSwarm": "#d62728", "PSO": "#1f77b4",
              "DifferentialEvolution": "#2ca02c", "RandomSearch": "#7f7f7f"}
    for ax, fname in zip(axes.flat, FUNCTIONS):
        for mname in METHODS:
            hists = curves[(fname, mname)]
            L = min(len(h) for h in hists)
            H = np.array([h[:L] for h in hists])
            med = np.median(H, axis=0)
            lo_, hi_ = np.percentile(H, [25, 75], axis=0)
            x = np.arange(L)
            ax.plot(x, np.maximum(med, 1e-16), label=mname, color=colors[mname])
            ax.fill_between(x, np.maximum(lo_, 1e-16), np.maximum(hi_, 1e-16),
                            alpha=0.15, color=colors[mname])
        ax.set_yscale("log")
        ax.set_title(f"{fname} (dim={args.dim})")
        ax.set_xlabel("iteration")
        ax.set_ylabel("best value")
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.05, 0.5,
                       "LifshitzSwarm: Lifshitz-coupled mirrors\n"
                       "+ quantum (zero-point) annealing\n"
                       f"pop={args.pop}, iters={args.iters}, "
                       f"seeds={args.seeds}",
                       fontsize=12, va="center")
    fig.suptitle("Black-box optimization benchmark", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "blackbox_convergence.png"), dpi=140)
    print(f"\nSaved results to {RESULTS}/")


if __name__ == "__main__":
    main()
