"""PINN benchmark: fluctuation-regularized machinery vs plain Adam on a 1D Poisson problem.

Problem:  -u''(x) = f(x) on (0,1),  u(0) = u(1) = 0
Exact:    u*(x) = sin(3 pi x) + 0.3 sin(9 pi x)   (multi-frequency: exposes
          spectral bias and residual/boundary imbalance, the classic PINN
          failure modes)

Configurations (matched by total gradient evaluations, not steps):
  A. Adam, fixed unit loss weights
  B. Adam + GradientPressureBalancer            (pressure-balanced boundaries)
  C. ZeroPointOptimizer + GradientPressureBalancer (vacuum-dressed gradient too)

Reports relative L2 error against the exact solution, the zero-point-energy
(flatness) diagnostic of the found minimum, and robustness of the solution
to parameter perturbation (the practical meaning of flatness).

Run:  python benchmarks/pinn_benchmark.py [--seeds 3] [--device cpu]
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
from fluctuation_opt import (ZeroPointOptimizer, GradientPressureBalancer, MLP,  # noqa: E402
                         partial_derivative)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)
CSV_PATH = os.path.join(RESULTS, "pinn_results.csv")


def u_exact(x):
    return torch.sin(3 * math.pi * x) + 0.3 * torch.sin(9 * math.pi * x)


def f_rhs(x):
    return ((3 * math.pi) ** 2 * torch.sin(3 * math.pi * x)
            + 0.3 * (9 * math.pi) ** 2 * torch.sin(9 * math.pi * x))


def make_losses(net, x_col, x_bc, device):
    def losses():
        x = x_col.clone().requires_grad_(True)
        u = net(x)
        u_xx = partial_derivative(u, x, order=2)
        residual = ((-u_xx - f_rhs(x)) ** 2).mean()
        boundary = (net(x_bc) ** 2).mean()
        return [residual, boundary]

    return losses


def rel_l2(net, device):
    with torch.no_grad():
        xg = torch.linspace(0, 1, 1001, device=device).reshape(-1, 1)
        err = net(xg) - u_exact(xg)
        return float(err.norm() / u_exact(xg).norm())


def perturbation_robustness(net, closure, sigma=3e-2, n=20, seed=0):
    """Mean loss increase under Gaussian parameter noise -- the operational
    meaning of 'flat minimum' (and of the zero-point energy)."""
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in net.parameters()]
    base = float(closure().detach())
    orig = [p.detach().clone() for p in params]
    incs = []
    for _ in range(n):
        with torch.no_grad():
            for p in params:
                p.add_(sigma * torch.randn(p.shape, generator=gen))
        incs.append(float(closure().detach()) - base)
        with torch.no_grad():
            for p, o in zip(params, orig):
                p.copy_(o)
    return float(np.mean(incs))


def run_config(config, seed, device, budget=9000):
    """budget = total gradient evaluations (backward passes through the loss)."""
    torch.manual_seed(seed)
    net = MLP([1, 64, 64, 64, 1]).to(device)
    x_col = torch.linspace(0, 1, 128, device=device).reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]], device=device)
    losses = make_losses(net, x_col, x_bc, device)

    balancer = None
    if config in ("adam+balance", "zero_point+pressure"):
        balancer = GradientPressureBalancer(net.parameters(), n_terms=2,
                                           update_every=25)

    def closure():
        terms = losses()
        return balancer(terms) if balancer is not None else sum(terms)

    curve = []  # (grad_evals, rel_l2)
    t0 = time.time()

    if config.startswith("adam"):
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        steps = budget  # 1 gradient eval per step
        for i in range(steps):
            opt.zero_grad()
            loss = closure()
            loss.backward()
            opt.step()
            if i % 200 == 0 or i == steps - 1:
                curve.append((i + 1, rel_l2(net, device)))
    else:
        n_probes = 2  # 1 antithetic pair -> 3 grad evals per step (center + pair)
        evals_per_step = 1 + 2 * ((n_probes + 1) // 2)
        steps = budget // evals_per_step
        opt = ZeroPointOptimizer(net.parameters(), lr=2e-3, sigma=5e-3,
                               n_probes=n_probes, floor_frac=0.3,
                               tau=steps / 4, seed=seed)
        for i in range(steps):
            opt.step(closure)
            if i % 70 == 0 or i == steps - 1:
                curve.append(((i + 1) * evals_per_step, rel_l2(net, device)))

    wall = time.time() - t0
    err = rel_l2(net, device)

    # flatness diagnostics at the found minimum
    diag_opt = ZeroPointOptimizer(net.parameters(), seed=0)
    zpe = diag_opt.zero_point_energy(closure, s=0.05, n_probes=4, m=15)
    robust = perturbation_robustness(net, closure, seed=seed)

    return {"config": config, "seed": seed, "rel_l2": err, "zpe": zpe,
            "robustness": robust, "wall_s": wall, "curve": curve, "net": net}


def plot_from_results():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(CSV_PATH) as fh:
        rows = list(csv.DictReader(fh))

    configs = ["adam", "adam+balance", "zero_point+pressure"]
    colors = {"adam": "#7f7f7f", "adam+balance": "#1f77b4",
              "zero_point+pressure": "#d62728"}
    labels = {"adam": "Adam",
              "adam+balance": "Adam + pressure",
              "zero_point+pressure": "Zero-point + pressure"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = [
        ("rel_l2", "relative L2 error", True),
        ("zpe", "regularized ZPE", True),
        ("robustness", "loss increase under noise", True),
    ]
    xpos = np.arange(len(configs))
    for ax, (field, title, logy) in zip(axes, metrics):
        med, lo, hi = [], [], []
        for cfg in configs:
            vals = np.array([float(r[field]) for r in rows if r["config"] == cfg])
            med.append(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            lo.append(np.median(vals) - q1)
            hi.append(q3 - np.median(vals))
        ax.bar(xpos, med, color=[colors[c] for c in configs],
               yerr=[lo, hi], capsize=4)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(xpos)
        ax.set_xticklabels([labels[c] for c in configs], fontsize=8)
        ax.set_title(title)
    fig.suptitle("Poisson PINN summary from saved results")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "pinn_benchmark.png"), dpi=140)
    print(f"saved {RESULTS}/pinn_benchmark.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=9000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        plot_from_results()
        return

    configs = ["adam", "adam+balance", "zero_point+pressure"]
    all_results = []
    for cfg in configs:
        for seed in range(args.seeds):
            r = run_config(cfg, seed, args.device, args.budget)
            all_results.append(r)
            print(f"{cfg:18s} seed={seed}  relL2={r['rel_l2']:.4f}  "
                  f"ZPE={r['zpe']:.3g}  robustness={r['robustness']:.3g}  "
                  f"({r['wall_s']:.0f}s)")

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "seed", "rel_l2", "zpe", "robustness", "wall_s"])
        for r in all_results:
            w.writerow([r["config"], r["seed"], r["rel_l2"], r["zpe"],
                        r["robustness"], r["wall_s"]])

    # ---------------- plots ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"adam": "#7f7f7f", "adam+balance": "#1f77b4",
              "zero_point+pressure": "#d62728"}
    labels = {"adam": "Adam (unit weights)",
              "adam+balance": "Adam + pressure balance",
              "zero_point+pressure": "Zero-point + pressure balance"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (1) error vs gradient evaluations
    ax = axes[0]
    for cfg in configs:
        runs = [r for r in all_results if r["config"] == cfg]
        L = min(len(r["curve"]) for r in runs)
        evals = [pt[0] for pt in runs[0]["curve"][:L]]
        errs = np.array([[pt[1] for pt in r["curve"][:L]] for r in runs])
        med = np.median(errs, axis=0)
        lo, hi = errs.min(axis=0), errs.max(axis=0)
        ax.plot(evals, med, color=colors[cfg], label=labels[cfg])
        ax.fill_between(evals, lo, hi, color=colors[cfg], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("gradient evaluations")
    ax.set_ylabel("relative L2 error")
    ax.set_title("1D Poisson PINN: error vs compute")
    ax.legend(fontsize=8)

    # (2) best solution per config vs exact
    ax = axes[1]
    xg = torch.linspace(0, 1, 500).reshape(-1, 1)
    ax.plot(xg.squeeze(), u_exact(xg).squeeze(), "k--", lw=2, label="exact")
    for cfg in configs:
        runs = [r for r in all_results if r["config"] == cfg]
        best = min(runs, key=lambda r: r["rel_l2"])
        with torch.no_grad():
            ax.plot(xg.squeeze(), best["net"](xg).squeeze(),
                    color=colors[cfg], lw=1.2,
                    label=f"{labels[cfg]} ({best['rel_l2']:.3f})")
    ax.set_title("solutions (best seed)")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)

    # (3) flatness: ZPE and perturbation robustness
    ax = axes[2]
    xpos = np.arange(len(configs))
    zpes = [np.median([r["zpe"] for r in all_results if r["config"] == c])
            for c in configs]
    robs = [np.median([r["robustness"] for r in all_results if r["config"] == c])
            for c in configs]
    ax2 = ax.twinx()
    ax.bar(xpos - 0.2, zpes, 0.35, color="#9467bd", label="zero-point energy")
    ax2.bar(xpos + 0.2, robs, 0.35, color="#ff7f0e",
            label="loss increase under param. noise")
    ax.set_yscale("log"); ax2.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Adam", "Adam\n+balance", "Zero-point\n+pressure"],
                       fontsize=8)
    ax.set_ylabel("regularized ZPE (flatness)")
    ax2.set_ylabel("robustness (lower = flatter)")
    ax.set_title("vacuum energy of the found minimum")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)

    fig.suptitle("Physics-informed NN benchmark: -u'' = f,  "
                 "u* = sin(3\u03c0x) + 0.3 sin(9\u03c0x)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "pinn_benchmark.png"), dpi=140)
    print(f"\nSaved results to {RESULTS}/")


if __name__ == "__main__":
    main()
