"""Real-PDE PINN benchmark: heat equation and viscous Burgers equation.

Problems
--------
heat:     u_t = alpha u_xx,           x in [0,1], t in [0,1], alpha = 0.1
          u(0,x) = sin(pi x),  u(t,0) = u(t,1) = 0
          exact:  u = sin(pi x) exp(-pi^2 alpha t)

burgers:  u_t + u u_x = nu u_xx,      x in [-1,1], t in [0,1], nu = 0.01/pi
          u(0,x) = -sin(pi x),  u(t,-1) = u(t,1) = 0
          exact:  Cole-Hopf transform evaluated with Gauss-Hermite
          quadrature (Basdevant et al. 1986) -- the classic PINN benchmark
          of Raissi, Perdikaris & Karniadakis (2019).  The solution steepens
          into a near-shock at x = 0, which is exactly the multi-scale
          regime where residual/boundary pressure imbalance kills PINNs.

Configurations (matched total gradient evaluations):
  A. adam              -- Adam, unit loss weights
  B. adam+balance      -- Adam + GradientPressureBalancer
  C. zero_point+pressure   -- ZeroPointOptimizer + GradientPressureBalancer

Run:  python benchmarks/pde_benchmark.py --pde heat    [--seeds 3]
      python benchmarks/pde_benchmark.py --pde burgers [--seeds 3]
      python benchmarks/pde_benchmark.py --plot        (after both)
"""

from __future__ import annotations

import argparse
import csv
import json
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
CSV_PATH = os.path.join(RESULTS, "pde_results.csv")
CURVES_PATH = os.path.join(RESULTS, "pde_curves.json")

NU = 0.01 / math.pi     # Burgers viscosity (Raissi et al. 2019)
ALPHA = 0.1             # heat diffusivity


# ---------------------------------------------------------------- exact sols
def heat_exact(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x) * np.exp(-math.pi**2 * ALPHA * t)


_GH_Z, _GH_W = np.polynomial.hermite.hermgauss(96)


def burgers_exact(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cole-Hopf solution via Gauss-Hermite quadrature (float64).

    u(x,t) = -[ sum_i w_i sin(pi(x - s z_i)) f(x - s z_i) ]
             /[ sum_i w_i f(x - s z_i) ],     s = sqrt(4 nu t),
    f(y) = exp(-cos(pi y) / (2 pi nu)).  The exponent reaches ~50, so the
    max is subtracted before exponentiation for stability.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(np.broadcast(t, x).shape, dtype=np.float64)
    tb, xb = np.broadcast_arrays(t, x)
    flat_t, flat_x = tb.ravel(), xb.ravel()
    res = np.empty_like(flat_t)
    c = 1.0 / (2.0 * math.pi * NU)
    for k, (tk, xk) in enumerate(zip(flat_t, flat_x)):
        if tk <= 0:
            res[k] = -math.sin(math.pi * xk)
            continue
        s = math.sqrt(4.0 * NU * tk)
        y = xk - s * _GH_Z
        logf = -c * np.cos(math.pi * y)
        logf -= logf.max()
        f = _GH_W * np.exp(logf)
        res[k] = -float(np.sum(f * np.sin(math.pi * y)) / np.sum(f))
    out.ravel()[:] = res
    return out


# ---------------------------------------------------------------- problems
class Problem:
    def __init__(self, name, device):
        self.name = name
        self.device = device
        g = torch.Generator().manual_seed(1234)
        if name == "heat":
            self.xlim, self.net_sizes = (0.0, 1.0), [2, 24, 24, 24, 1]
            n_col, n_ic, n_bc = 1024, 64, 64
        elif name == "burgers":
            self.xlim, self.net_sizes = (-1.0, 1.0), [2, 24, 24, 24, 1]
            n_col, n_ic, n_bc = 1536, 96, 96
        else:
            raise ValueError(name)
        lo, hi = self.xlim
        tx = torch.rand(n_col, 2, generator=g)
        tx[:, 1] = lo + (hi - lo) * tx[:, 1]
        self.tx_col = tx.to(device)
        x_ic = lo + (hi - lo) * torch.rand(n_ic, 1, generator=g)
        self.tx_ic = torch.cat([torch.zeros_like(x_ic), x_ic], 1).to(device)
        t_bc = torch.rand(n_bc, 1, generator=g)
        xb = torch.where(torch.rand(n_bc, 1, generator=g) < 0.5,
                         torch.full_like(t_bc, lo), torch.full_like(t_bc, hi))
        self.tx_bc = torch.cat([t_bc, xb], 1).to(device)
        if name == "heat":
            self.u_ic = torch.sin(math.pi * self.tx_ic[:, 1:2])
        else:
            self.u_ic = -torch.sin(math.pi * self.tx_ic[:, 1:2])

    def losses(self, net):
        tx = self.tx_col.clone().requires_grad_(True)
        u = net(tx)
        grads = torch.autograd.grad(u, tx, torch.ones_like(u),
                                    create_graph=True)[0]
        u_t, u_x = grads[:, 0:1], grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x),
                                   create_graph=True)[0][:, 1:2]
        if self.name == "heat":
            r = u_t - ALPHA * u_xx
        else:
            r = u_t + u * u_x - NU * u_xx
        residual = (r**2).mean()
        ic = ((net(self.tx_ic) - self.u_ic) ** 2).mean()
        bc = (net(self.tx_bc) ** 2).mean()
        return [residual, ic, bc]

    def rel_l2(self, net):
        lo, hi = self.xlim
        t = np.linspace(0, 1, 64)
        x = np.linspace(lo, hi, 128)
        T, X = np.meshgrid(t, x, indexing="ij")
        exact = heat_exact(T, X) if self.name == "heat" else burgers_exact(T, X)
        tx = torch.tensor(np.stack([T.ravel(), X.ravel()], 1),
                          dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = net(tx).cpu().numpy().reshape(T.shape)
        return float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))


def perturbation_robustness(net, closure, sigma=3e-2, n=20, seed=0):
    gen = torch.Generator().manual_seed(seed)
    params = list(net.parameters())
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


def uses_pressure_balancer(config):
    return config in ("adam+balance", "zero_point+pressure")


# ---------------------------------------------------------------- runner
def run_config(problem, config, seed, device, budget):
    torch.manual_seed(seed)
    prob = Problem(problem, device)
    net = MLP(prob.net_sizes).to(device)

    balancer = None
    if uses_pressure_balancer(config):
        balancer = GradientPressureBalancer(net.parameters(), n_terms=3,
                                           update_every=25)

    def closure():
        terms = prob.losses(net)
        return balancer(terms) if balancer is not None else sum(terms)

    curve = []
    t0 = time.time()
    if config.startswith("adam"):
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        for i in range(budget):
            opt.zero_grad()
            loss = closure()
            loss.backward()
            opt.step()
            if i % 250 == 0 or i == budget - 1:
                curve.append((i + 1, prob.rel_l2(net)))
    else:
        n_probes = 2
        evals_per_step = 1 + 2 * ((n_probes + 1) // 2)
        steps = budget // evals_per_step
        opt = ZeroPointOptimizer(net.parameters(), lr=2e-3, sigma=5e-3,
                               n_probes=n_probes, floor_frac=0.3,
                               tau=steps / 4, seed=seed)
        for i in range(steps):
            opt.step(closure)
            if i % 85 == 0 or i == steps - 1:
                curve.append(((i + 1) * evals_per_step, prob.rel_l2(net)))
    wall = time.time() - t0

    err = prob.rel_l2(net)
    diag = ZeroPointOptimizer(net.parameters(), seed=0)
    zpe = diag.zero_point_energy(closure, s=0.05, n_probes=4, m=15)
    robust = perturbation_robustness(net, closure, seed=seed)
    return {"pde": problem, "config": config, "seed": seed, "rel_l2": err,
            "zpe": zpe, "robustness": robust, "wall_s": wall, "curve": curve}


def append_results(rows):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(["pde", "config", "seed", "rel_l2", "zpe",
                        "robustness", "wall_s"])
        for r in rows:
            w.writerow([r["pde"], r["config"], r["seed"], r["rel_l2"],
                        r["zpe"], r["robustness"], r["wall_s"]])
    curves = {}
    if os.path.exists(CURVES_PATH):
        with open(CURVES_PATH) as fh:
            curves = json.load(fh)
    for r in rows:
        curves[f"{r['pde']}|{r['config']}|{r['seed']}"] = r["curve"]
    with open(CURVES_PATH, "w") as fh:
        json.dump(curves, fh)


# ---------------------------------------------------------------- plotting
def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(CURVES_PATH) as fh:
        curves = json.load(fh)
    rows = []
    with open(CSV_PATH) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)

    configs = ["adam", "adam+balance", "zero_point+pressure"]
    colors = {"adam": "#7f7f7f", "adam+balance": "#1f77b4",
              "zero_point+pressure": "#d62728"}
    labels = {"adam": "Adam (unit weights)",
              "adam+balance": "Adam + pressure balance",
              "zero_point+pressure": "Zero-point + pressure balance"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, pde in zip(axes[:2], ["heat", "burgers"]):
        for cfg in configs:
            runs = [json.loads(json.dumps(v)) for k, v in curves.items()
                    if k.startswith(f"{pde}|{cfg}|")]
            if not runs:
                continue
            L = min(len(c) for c in runs)
            evals = [pt[0] for pt in runs[0][:L]]
            errs = np.array([[pt[1] for pt in c[:L]] for c in runs])
            med = np.median(errs, axis=0)
            ax.plot(evals, med, color=colors[cfg], label=labels[cfg])
            ax.fill_between(evals, errs.min(0), errs.max(0),
                            color=colors[cfg], alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("gradient evaluations")
        ax.set_ylabel("relative L2 error")
        ax.set_title(f"{pde} equation")
        ax.legend(fontsize=8)

    # burgers final-time profile from the exact solution vs typical shapes
    ax = axes[2]
    x = np.linspace(-1, 1, 400)
    for tt, c in [(0.0, "0.8"), (0.25, "0.6"), (0.5, "0.4"), (0.75, "0.2"),
                  (1.0, "0.0")]:
        ax.plot(x, burgers_exact(np.full_like(x, tt), x), color=str(c),
                label=f"t={tt}")
    ax.set_title("Burgers exact (Cole-Hopf): shock steepening")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.legend(fontsize=8)

    fig.suptitle("PINNs on real PDEs: heat and viscous Burgers "
                 f"(nu = 0.01/pi)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "pde_benchmark.png"), dpi=140)
    print(f"saved {RESULTS}/pde_benchmark.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", choices=["heat", "burgers"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--configs", default="adam,adam+balance,zero_point+pressure")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.plot:
        make_plots()
        return
    if args.pde is None:
        ap.error("--pde required (or --plot)")

    rows = []
    for cfg in args.configs.split(","):
        for seed in range(args.seeds):
            r = run_config(args.pde, cfg, seed, args.device, args.budget)
            rows.append(r)
            print(f"{args.pde:8s} {cfg:18s} seed={seed}  "
                  f"relL2={r['rel_l2']:.4f}  ZPE={r['zpe']:.3g}  "
                  f"robustness={r['robustness']:.3g}  ({r['wall_s']:.0f}s)")
    append_results(rows)
    print("appended to", CSV_PATH)


if __name__ == "__main__":
    main()
