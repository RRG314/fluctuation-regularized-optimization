"""Real-data ML benchmark: CasimirOptimizer vs Adam vs SGD-momentum.

Datasets (both real, loaded through scikit-learn)
-------------------------------------------------
digits : 1,797 8x8 grayscale images of handwritten digits (UCI ML repo),
         10-class classification, MLP [64, 32, 10], cross-entropy.
housing: California housing, 20,640 census block groups (1990 US census),
         regression of median house value, MLP [8, 32, 32, 1], MSE.

Protocol
--------
- 70/30 train/test split, features standardized (regression target too).
- Fixed budget of *gradient evaluations* (backward passes), not steps:
  CasimirOptimizer spends 3 evaluations per step (center + antithetic
  pair), so Adam/SGD get 3x as many steps.  Same minibatch stream per seed.
- 5 seeds; report median and IQR of the test metric.
- Flat-minimum diagnostics at the end of training:
  * robustness: degradation of the TEST metric when parameters are
    perturbed with Gaussian noise (20 draws, sigma = 0.02),
  * regularized zero-point energy (SLQ estimate) of the TRAIN loss.

Run:  python benchmarks/ml_benchmark.py --dataset digits  [--seeds 5]
      python benchmarks/ml_benchmark.py --dataset housing [--seeds 5]
      python benchmarks/ml_benchmark.py --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from casimir_opt import CasimirOptimizer, MLP  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)
CSV_PATH = os.path.join(RESULTS, "ml_results.csv")
CURVES_PATH = os.path.join(RESULTS, "ml_curves.json")


# ---------------------------------------------------------------- data
def load(dataset, seed):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if dataset == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        X, y = d.data, d.target
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        return (torch.tensor(Xtr, dtype=torch.float32),
                torch.tensor(ytr, dtype=torch.long),
                torch.tensor(Xte, dtype=torch.float32),
                torch.tensor(yte, dtype=torch.long))
    elif dataset == "housing":
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        X, y = d.data, d.target
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              random_state=seed)
        scx = StandardScaler().fit(Xtr)
        scy = StandardScaler().fit(ytr.reshape(-1, 1))
        Xtr, Xte = scx.transform(Xtr), scx.transform(Xte)
        ytr = scy.transform(ytr.reshape(-1, 1)).ravel()
        yte = scy.transform(yte.reshape(-1, 1)).ravel()
        return (torch.tensor(Xtr, dtype=torch.float32),
                torch.tensor(ytr, dtype=torch.float32).unsqueeze(1),
                torch.tensor(Xte, dtype=torch.float32),
                torch.tensor(yte, dtype=torch.float32).unsqueeze(1))
    raise ValueError(dataset)


def make_net(dataset):
    return MLP([64, 32, 10]) if dataset == "digits" else MLP([8, 32, 32, 1])


def loss_fn(dataset, net, X, y):
    out = net(X)
    if dataset == "digits":
        return Fnn.cross_entropy(out, y)
    return Fnn.mse_loss(out, y)


@torch.no_grad()
def test_metric(dataset, net, Xte, yte):
    """digits -> test error rate (lower better); housing -> test RMSE
    in standardized units (lower better)."""
    out = net(Xte)
    if dataset == "digits":
        return float((out.argmax(1) != yte).float().mean())
    return float(torch.sqrt(Fnn.mse_loss(out, yte)))


def robustness(dataset, net, Xte, yte, sigma=0.02, n=20, seed=0):
    """Mean increase of the test metric under parameter noise."""
    gen = torch.Generator().manual_seed(seed)
    params = list(net.parameters())
    orig = [p.detach().clone() for p in params]
    base = test_metric(dataset, net, Xte, yte)
    incs = []
    for _ in range(n):
        with torch.no_grad():
            for p in params:
                p.add_(sigma * torch.randn(p.shape, generator=gen))
        incs.append(test_metric(dataset, net, Xte, yte) - base)
        with torch.no_grad():
            for p, o in zip(params, orig):
                p.copy_(o)
    return float(np.mean(incs))


# ---------------------------------------------------------------- runner
def run(dataset, method, seed, budget, batch):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = load(dataset, seed)
    net = make_net(dataset)
    ntr = Xtr.shape[0]
    gen = torch.Generator().manual_seed(seed + 999)

    def batch_idx():
        return torch.randint(0, ntr, (min(batch, ntr),), generator=gen)

    curve = []
    t0 = time.time()
    if method == "casimir":
        n_probes = 2
        evals_per_step = 1 + 2 * ((n_probes + 1) // 2)
        steps = budget // evals_per_step
        opt = CasimirOptimizer(net.parameters(), lr=1e-3, sigma=1e-2,
                               n_probes=n_probes, floor_frac=0.3,
                               tau=steps / 4, seed=seed)
        for i in range(steps):
            idx = batch_idx()
            opt.step(lambda: loss_fn(dataset, net, Xtr[idx], ytr[idx]))
            if i % (steps // 30) == 0 or i == steps - 1:
                curve.append(((i + 1) * evals_per_step,
                              test_metric(dataset, net, Xte, yte)))
    else:
        if method == "adam":
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        elif method == "sgd":
            opt = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
        else:
            raise ValueError(method)
        for i in range(budget):
            idx = batch_idx()
            opt.zero_grad()
            loss = loss_fn(dataset, net, Xtr[idx], ytr[idx])
            loss.backward()
            opt.step()
            if i % (budget // 30) == 0 or i == budget - 1:
                curve.append((i + 1, test_metric(dataset, net, Xte, yte)))
    wall = time.time() - t0

    metric = test_metric(dataset, net, Xte, yte)
    rob = robustness(dataset, net, Xte, yte, seed=seed)
    diag = CasimirOptimizer(net.parameters(), seed=0)
    zpe = diag.zero_point_energy(
        lambda: loss_fn(dataset, net, Xtr, ytr), s=0.05, n_probes=4, m=15)
    return {"dataset": dataset, "method": method, "seed": seed,
            "metric": metric, "robustness": rob, "zpe": zpe,
            "wall_s": wall, "curve": curve}


def append_results(rows):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(["dataset", "method", "seed", "test_metric",
                        "robustness", "zpe", "wall_s"])
        for r in rows:
            w.writerow([r["dataset"], r["method"], r["seed"], r["metric"],
                        r["robustness"], r["zpe"], r["wall_s"]])
    curves = {}
    if os.path.exists(CURVES_PATH):
        with open(CURVES_PATH) as fh:
            curves = json.load(fh)
    for r in rows:
        curves[f"{r['dataset']}|{r['method']}|{r['seed']}"] = r["curve"]
    with open(CURVES_PATH, "w") as fh:
        json.dump(curves, fh)


# ---------------------------------------------------------------- plots
def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(CURVES_PATH) as fh:
        curves = json.load(fh)
    with open(CSV_PATH) as fh:
        rows = list(csv.DictReader(fh))

    methods = ["sgd", "adam", "casimir"]
    colors = {"sgd": "#7f7f7f", "adam": "#1f77b4", "casimir": "#d62728"}
    labels = {"sgd": "SGD + momentum", "adam": "Adam",
              "casimir": "CasimirOptimizer"}
    metric_name = {"digits": "test error rate", "housing": "test RMSE (std.)"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for j, ds in enumerate(["digits", "housing"]):
        ax = axes[0, j]
        for m in methods:
            runs = [v for k, v in curves.items()
                    if k.startswith(f"{ds}|{m}|")]
            if not runs:
                continue
            L = min(len(c) for c in runs)
            evals = [pt[0] for pt in runs[0][:L]]
            vals = np.array([[pt[1] for pt in c[:L]] for c in runs])
            med = np.median(vals, axis=0)
            ax.plot(evals, med, color=colors[m], label=labels[m])
            ax.fill_between(evals, np.percentile(vals, 25, 0),
                            np.percentile(vals, 75, 0),
                            color=colors[m], alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("gradient evaluations")
        ax.set_ylabel(metric_name[ds])
        ax.set_title(f"{ds} (real data): learning curves")
        ax.legend(fontsize=9)

        ax = axes[1, j]
        w = 0.25
        for i, m in enumerate(methods):
            rs = [float(r["robustness"]) for r in rows
                  if r["dataset"] == ds and r["method"] == m]
            if not rs:
                continue
            med = np.median(rs)
            q1, q3 = np.percentile(rs, [25, 75])
            ax.bar(i, med, w * 2, color=colors[m],
                   yerr=[[med - q1], [q3 - med]], capsize=4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([labels[m] for m in methods], fontsize=9)
        ax.set_ylabel("metric increase under param. noise")
        ax.set_title(f"{ds}: robustness to parameter perturbation "
                     "(lower = flatter)")
    fig.suptitle("Real-data ML benchmark (matched gradient-evaluation budget,"
                 " 5 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "ml_benchmark.png"), dpi=140)
    print(f"saved {RESULTS}/ml_benchmark.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["digits", "housing"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--methods", default="sgd,adam,casimir")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.plot:
        make_plots()
        return
    if args.dataset is None:
        ap.error("--dataset required (or --plot)")

    rows = []
    for m in args.methods.split(","):
        for seed in range(args.seeds):
            r = run(args.dataset, m, seed, args.budget, args.batch)
            rows.append(r)
            print(f"{args.dataset:8s} {m:8s} seed={seed}  "
                  f"metric={r['metric']:.4f}  robust={r['robustness']:.4g}  "
                  f"ZPE={r['zpe']:.3g}  ({r['wall_s']:.0f}s)")
    append_results(rows)
    print("appended to", CSV_PATH)


if __name__ == "__main__":
    main()
