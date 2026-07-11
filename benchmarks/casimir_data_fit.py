"""Fit REAL Casimir experiment data with CasimirSwarm.

Data
----
Table 1 of R. S. Decca, D. Lopez, E. Fischbach, G. L. Klimchitskaya,
D. E. Krause, V. M. Mostepanenko, "Novel constraints on light elementary
particles and extra-dimensional physics from the Casimir effect",
Eur. Phys. J. C 51, 963-975 (2007), arXiv:0706.3283.

Column (a): mean Casimir pressure between two Au surfaces (sphere-plate
measurement mapped to equivalent parallel plates via the proximity force
approximation), measured with a micromechanical torsional oscillator at
T ~= 300 K.  Column (e): half-width of the 95% confidence interval.
Sphere radius R = 151.3 um; absolute separation error 0.6 nm.

Model
-----
Finite-temperature Lifshitz formula for two identical plasma-model metals:

    P(z,T) = -(kB T / pi) sum'_{l>=0} Int_0^inf dk k q_l
             sum_{alpha in TM,TE} [ r_alpha^{-2} e^{2 q_l z} - 1 ]^{-1}

with Matsubara frequencies xi_l = 2 pi kB T l / hbar,
q_l^2 = k^2 + xi_l^2/c^2, and plasma-model permittivity
eps(i xi) = 1 + wp^2/xi^2, for which the Fresnel reflectivities at
imaginary frequency are

    r_TE = (q - kbar)/(q + kbar),          kbar = sqrt(q^2 + wp^2/c^2),
    r_TM = (eps q - kbar)/(eps q + kbar),  r_TM(l=0) = 1.

Free parameters fitted to the data:
    wp  : plasma frequency of gold, in eV  (literature: ~9.0 eV)
    dz  : global separation offset, in nm, bounded by the quoted
          absolute separation error of +-0.6 nm

Objective: chi^2 = sum_i ( (P_model(z_i + dz) - P_i) / sigma_i )^2,
sigma_i = Xi_i / 1.96  (converting the 95% half-width to 1 s.d.).

The point of the exercise: a *Casimir-physics-based optimizer fitting
Casimir-physics data* -- and an independent check that the swarm recovers
the physically correct plasma frequency from real measurements.

Run:  python benchmarks/casimir_data_fit.py [--seeds 8]
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
from casimir_opt import CasimirSwarm  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------- constants
HBAR = 1.054571817e-34  # J s
KB = 1.380649e-23       # J / K
C = 2.99792458e8        # m / s
EV = 1.602176634e-19    # J

# ------------------------------------------------------- Decca 2007 Table 1
# z [nm], mean measured pressure magnitude [mPa], 95% CI half-width [mPa]
DECCA_2007 = np.array([
    [162.0, 1108.4, 21.2],
    [166.0, 1012.7, 19.0],
    [170.0, 926.85, 17.1],
    [180.0, 751.19, 13.3],
    [190.0, 616.00, 10.5],
    [200.0, 510.50, 8.40],
    [250.0, 225.16, 3.30],
    [300.0, 114.82, 1.63],
    [350.0, 64.634, 0.98],
    [400.0, 39.198, 0.69],
    [450.0, 25.155, 0.54],
    [500.0, 16.822, 0.47],
    [550.0, 11.678, 0.42],
    [600.0, 8.410, 0.39],
    [650.0, 6.216, 0.38],
    [746.0, 3.614, 0.35],
])


# ---------------------------------------------------------------- model
def lifshitz_plasma_pressure(z_m: torch.Tensor, wp_ev: torch.Tensor,
                             T: float = 300.0, l_max: int = 300,
                             n_nodes: int = 80) -> torch.Tensor:
    """|Casimir pressure| (Pa) between two identical plasma-model metals.

    Vectorized over separations ``z_m`` (meters).  Uses the substitution
    y = 2 q z, Gauss-Laguerre quadrature in y, and an explicit Matsubara
    sum (l = 0 term with weight 1/2).  float64 throughout.
    """
    z = z_m.to(torch.float64).reshape(-1, 1, 1)              # (Z,1,1)
    wp = wp_ev.to(torch.float64) * EV / HBAR                 # rad/s

    l = torch.arange(0, l_max + 1, dtype=torch.float64).reshape(1, -1, 1)
    xi = 2.0 * math.pi * KB * T / HBAR * l                   # (1,L,1)
    y_l = 2.0 * xi * z / C                                   # (Z,L,1)

    u, w_gl = np.polynomial.laguerre.laggauss(n_nodes)
    u = torch.tensor(u, dtype=torch.float64).reshape(1, 1, -1)
    w_gl = torch.tensor(w_gl, dtype=torch.float64).reshape(1, 1, -1)

    y = y_l + u                                              # (Z,L,N)
    wzc = 2.0 * z * wp / C                                   # (Z,1,1)
    kbar_y = torch.sqrt(y**2 + wzc**2)

    r_te = (y - kbar_y) / (y + kbar_y)
    # eps = 1 + wp^2/xi^2 ; for l = 0 set r_TM = 1 explicitly below
    eps = torch.ones_like(y) + (wzc / torch.clamp(y_l, min=1e-300))**2
    r_tm = (eps * y - kbar_y) / (eps * y + kbar_y)
    r_tm[:, 0, :] = 1.0

    ey = torch.exp(torch.clamp(y, max=700.0))
    integrand = (y**2) * (1.0 / (ey / torch.clamp(r_tm**2, min=1e-300) - 1.0)
                          + 1.0 / (ey / torch.clamp(r_te**2, min=1e-300) - 1.0))
    # Gauss-Laguerre: Int_{y_l}^inf f(y) dy = sum_i w_i e^{u_i} f(y_l + u_i)
    integral = torch.sum(w_gl * torch.exp(u) * integrand, dim=-1)  # (Z,L)
    integral[:, 0] *= 0.5                                          # l=0 weight

    pref = KB * T / (8.0 * math.pi * z.reshape(-1) ** 3)
    return pref * integral.sum(dim=1)


def chi2(theta: torch.Tensor, data=DECCA_2007, T=300.0) -> torch.Tensor:
    """theta = (wp [eV], dz [nm]) -> chi^2 against the measured pressures."""
    wp, dz = theta[..., 0], theta[..., 1]
    z = torch.tensor(data[:, 0], dtype=torch.float64) * 1e-9
    P_meas = torch.tensor(data[:, 1], dtype=torch.float64) * 1e-3   # Pa
    sig = torch.tensor(data[:, 2], dtype=torch.float64) * 1e-3 / 1.96
    out = []
    for wpi, dzi in zip(torch.atleast_1d(wp), torch.atleast_1d(dz)):
        zz = z + dzi * 1e-9
        P = lifshitz_plasma_pressure(zz, wpi, T=T)
        out.append((((P - P_meas) / sig) ** 2).sum())
    return torch.stack(out) if out[0].ndim == 0 and len(out) > 1 else out[0]


def chi2_batch(X: torch.Tensor) -> torch.Tensor:
    """Vectorized wrapper for the swarm: X is (N,2) -> (N,) chi^2."""
    return torch.stack([chi2(x) for x in X])


# ---------------------------------------------------------------- validation
def validate_model():
    """Model checks before fitting anything."""
    # 1) perfect-conductor limit: wp -> large, T -> small
    z = torch.tensor([300e-9], dtype=torch.float64)
    p = float(lifshitz_plasma_pressure(z, torch.tensor(5e3), T=1.0,
                                       l_max=40000, n_nodes=80))
    ideal = math.pi**2 * HBAR * C / (240.0 * (300e-9) ** 4)
    err_ideal = abs(p - ideal) / ideal
    # 2) against Decca's own plasma-model theory column (b) at wp = 9.0 eV
    z_all = torch.tensor(DECCA_2007[:, 0], dtype=torch.float64) * 1e-9
    p_all = lifshitz_plasma_pressure(z_all, torch.tensor(9.0), T=300.0)
    theory_b = np.array([1098.4, 1007.1, 923.71, 750.58, 616.71, 511.26,
                         225.71, 114.87, 64.574, 39.096, 25.034, 16.785,
                         11.669, 8.365, 6.151, 3.620]) * 1e-3
    dev = np.abs(p_all.numpy() - theory_b) / theory_b
    print(f"[validate] perfect-conductor limit deviation: {err_ideal:.2%}")
    print(f"[validate] vs Decca plasma theory (wp=9 eV): "
          f"median {np.median(dev):.2%}, max {dev.max():.2%}")
    return err_ideal, dev


# ---------------------------------------------------------------- fitting
def fit_swarm(seed: int, max_iter: int = 60, n_particles: int = 24):
    sw = CasimirSwarm(bounds=[(6.0, 12.0), (-0.6, 0.6)],
                      n_particles=n_particles, seed=seed,
                      dtype=torch.float64)
    t0 = time.time()
    res = sw.minimize(chi2_batch, max_iter=max_iter)
    res["wall_s"] = time.time() - t0
    return res


def fit_scipy_de(seed: int):
    from scipy.optimize import differential_evolution
    t0 = time.time()

    def f(x):
        return float(chi2(torch.tensor(x, dtype=torch.float64)))

    r = differential_evolution(f, [(6.0, 12.0), (-0.6, 0.6)], seed=seed,
                               popsize=12, maxiter=60, tol=0, polish=True)
    return {"x": torch.tensor(r.x), "fun": float(r.fun),
            "wall_s": time.time() - t0, "n_evals": r.nfev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    validate_model()

    rows = []
    for seed in range(args.seeds):
        r = fit_swarm(seed)
        wp, dz = float(r["x"][0]), float(r["x"][1])
        red = r["fun"] / (len(DECCA_2007) - 2)
        rows.append(["CasimirSwarm", seed, wp, dz, r["fun"], red,
                     r["n_evals"], r["wall_s"]])
        print(f"CasimirSwarm seed={seed}  wp={wp:.3f} eV  dz={dz:+.3f} nm  "
              f"chi2={r['fun']:.2f} (red. {red:.2f})  "
              f"[{r['n_evals']} evals, {r['wall_s']:.0f}s]")
    for seed in range(max(2, args.seeds // 4)):
        r = fit_scipy_de(seed)
        wp, dz = float(r["x"][0]), float(r["x"][1])
        red = r["fun"] / (len(DECCA_2007) - 2)
        rows.append(["scipy-DE", seed, wp, dz, r["fun"], red,
                     r["n_evals"], r["wall_s"]])
        print(f"scipy-DE     seed={seed}  wp={wp:.3f} eV  dz={dz:+.3f} nm  "
              f"chi2={r['fun']:.2f} (red. {red:.2f})  "
              f"[{r['n_evals']} evals, {r['wall_s']:.0f}s]")

    with open(os.path.join(RESULTS, "casimir_fit_results.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "wp_eV", "dz_nm", "chi2",
                    "chi2_reduced", "n_evals", "wall_s"])
        w.writerows(rows)

    # ---------------- plot ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    swarm_rows = [r for r in rows if r[0] == "CasimirSwarm"]
    best = min(swarm_rows, key=lambda r: r[4])
    wp_b, dz_b = best[2], best[3]

    z_dense = torch.linspace(155e-9, 760e-9, 300, dtype=torch.float64)
    P_fit = lifshitz_plasma_pressure(z_dense + dz_b * 1e-9,
                                     torch.tensor(wp_b)) * 1e3
    P_ideal = (math.pi**2 * HBAR * C / (240.0 * z_dense**4)) * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.errorbar(DECCA_2007[:, 0], DECCA_2007[:, 1], yerr=DECCA_2007[:, 2],
                fmt="o", ms=4, capsize=3, color="k",
                label="Decca et al. 2007 (measured)")
    ax.plot(z_dense * 1e9, P_fit.numpy(), color="#d62728",
            label=f"Lifshitz plasma fit: wp={wp_b:.2f} eV")
    ax.plot(z_dense * 1e9, P_ideal.numpy(), "--", color="0.6",
            label="ideal (perfect conductor)")
    ax.set_yscale("log")
    ax.set_xlabel("separation z (nm)")
    ax.set_ylabel("|Casimir pressure| (mPa)")
    ax.set_title("Real data vs swarm-fitted Lifshitz model")
    ax.legend(fontsize=8)

    ax = axes[1]
    z_pts = torch.tensor(DECCA_2007[:, 0], dtype=torch.float64) * 1e-9
    P_at = lifshitz_plasma_pressure(z_pts + dz_b * 1e-9,
                                    torch.tensor(wp_b)) * 1e3
    resid = (P_at.numpy() - DECCA_2007[:, 1]) / (DECCA_2007[:, 2] / 1.96)
    ax.axhline(0, color="0.7")
    ax.axhspan(-1.96, 1.96, color="0.9")
    ax.plot(DECCA_2007[:, 0], resid, "o-", color="#d62728", ms=4)
    ax.set_xlabel("separation z (nm)")
    ax.set_ylabel("residual / sigma")
    ax.set_title("normalized residuals (band = 95% CI)")

    ax = axes[2]
    wps = [r[2] for r in swarm_rows]
    ax.hist(wps, bins=8, color="#d62728", alpha=0.7)
    ax.axvline(9.0, color="k", ls="--", label="literature wp(Au) = 9.0 eV")
    ax.axvline(np.median(wps), color="#d62728",
               label=f"swarm median = {np.median(wps):.2f} eV")
    ax.set_xlabel("fitted plasma frequency (eV)")
    ax.set_title(f"recovered wp across {len(wps)} seeds")
    ax.legend(fontsize=8)

    fig.suptitle("CasimirSwarm fitting real Casimir-force measurements "
                 "(Decca et al., EPJ C 51, 963 (2007))")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "casimir_fit.png"), dpi=140)
    print(f"\nSaved results to {RESULTS}/")


if __name__ == "__main__":
    main()
