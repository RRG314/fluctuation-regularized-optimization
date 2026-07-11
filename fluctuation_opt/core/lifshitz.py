"""Lifshitz theory: fluctuation-induced couplings between imperfect mirrors.

Physics background
------------------
For a 1D scalar field between two mirrors with (frequency-independent)
reflectivities ``r1, r2`` at separation ``d``, the zero-temperature Casimir
interaction energy, written over imaginary frequency ``xi`` (the standard
Lifshitz rotation), is

    E(d) = (1/2 pi) * Integral_0^inf  d xi  log(1 - r1 r2 e^{-2 xi d}).

Expanding the log and integrating term by term:

    Integral_0^inf log(1 - z e^{-2 xi d}) d xi = -(1/2d) * Li_2(z),

so the interaction energy and force are *exactly* polylogarithmic:

    E(d) = - Li_2(r1 r2) / (4 pi d),          (attractive well)
    F(d) = - dE/dd = - Li_2(r1 r2) / (4 pi d^2).

Perfect mirrors give ``Li_2(1) = zeta(2) = pi^2 / 6`` -- the classic Casimir
zeta value.  At finite temperature the integral becomes the Matsubara sum

    F_T(d) = T * Sum'_n  log(1 - z e^{-2 xi_n d}),   xi_n = 2 pi n T,

(prime: the n=0 term carries weight 1/2).

Optimization mapping
--------------------
In :class:`fluctuation_opt.swarm.LifshitzSwarm`, candidate solutions are mirrors
and *fitness maps to reflectivity*: good solutions are nearly perfect mirrors
(r -> 1), bad ones are nearly transparent (r -> 0).  The resulting attraction

- is strongest toward *pairs* of good solutions (the coupling depends on the
  product ``r_i r_j``, not on a single "global best"),
- is short-ranged (``1/d^2`` from a ``1/d`` energy well) compared to the
  fitness-weighted long-range pulls of PSO/gravitational-search methods,
- vanishes smoothly for bad solutions instead of being switched off by
  heuristics.

All functions are torch, vectorized, and device-agnostic.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

Tensor = torch.Tensor

ZETA_2 = math.pi ** 2 / 6.0


def polylog(s: float, z: Tensor, n_terms: int = 64) -> Tensor:
    """Polylogarithm ``Li_s(z) = sum_{n>=1} z^n / n^s`` for ``0 <= z <= 1``.

    The series converges on [0, 1] for s >= 2 (at z=1 it is zeta(s)).
    ``n_terms=64`` gives ~1e-4 absolute accuracy for Li_2 at z=1 and much
    better for z < 1; increase for tighter tolerances.
    """
    z = torch.clamp(torch.as_tensor(z), 0.0, 1.0)
    n = torch.arange(1, n_terms + 1, device=z.device, dtype=z.dtype)
    # z^n / n^s  -- broadcast over trailing dim
    powers = z.unsqueeze(-1) ** n
    return torch.sum(powers / n.pow(s), dim=-1)


def lifshitz_energy(z: Tensor, d: Tensor, a0: float = 1e-3, n_terms: int = 64) -> Tensor:
    """Interaction energy ``E = -Li_2(z) / (4 pi (d + a0))``.

    ``z = r_i * r_j`` is the reflectivity product; ``a0`` is a short-distance
    cutoff playing the role of the mirrors' plasma wavelength (real mirrors
    become transparent to modes shorter than their skin depth, which caps
    the force at contact -- and keeps the optimizer's forces finite).
    """
    return -polylog(2.0, z, n_terms) / (4.0 * math.pi * (d + a0))


def lifshitz_force(z: Tensor, d: Tensor, a0: float = 1e-3, n_terms: int = 64) -> Tensor:
    """Magnitude of the attractive force, ``|F| = Li_2(z) / (4 pi (d + a0)^2)``."""
    return polylog(2.0, z, n_terms) / (4.0 * math.pi * (d + a0) ** 2)


def lifshitz_free_energy(z: Tensor, d: Tensor, T: float, n_max: int = 32,
                         a0: float = 1e-3) -> Tensor:
    """Finite-temperature Matsubara form of the interaction free energy.

    ``F_T = T * [ (1/2) log(1 - z e^{-2 xi_0 d}) + sum_{n=1}^{n_max} log(1 - z e^{-2 xi_n d}) ]``
    with ``xi_n = 2 pi n T``.  Reduces to :func:`lifshitz_energy` as the
    Matsubara sum approaches the T -> 0 integral.
    """
    z = torch.clamp(torch.as_tensor(z), 0.0, 1.0 - 1e-9)
    d = torch.as_tensor(d) + a0
    n = torch.arange(0, n_max + 1, device=z.device, dtype=z.dtype)
    xi = 2.0 * math.pi * T * n
    w = torch.ones_like(xi)
    w[0] = 0.5
    terms = torch.log1p(-z.unsqueeze(-1) * torch.exp(-2.0 * xi * d.unsqueeze(-1)))
    return T * torch.sum(w * terms, dim=-1)


# --------------------------------------------------------------------------
# Fitness -> reflectivity maps
# --------------------------------------------------------------------------

def reflectivity_from_rank(fitness: Tensor, r_max: float = 0.98, gamma: float = 2.0) -> Tensor:
    """Rank-based reflectivity (robust to fitness scaling; minimization).

    The best particle gets ``r_max``, the worst gets ~0, interpolating as
    ``r = r_max * ((N - rank) / N)^gamma``.  ``gamma`` sharpens the contrast
    between good and bad mirrors.
    """
    n = fitness.numel()
    if n == 1:
        return torch.full_like(fitness, float(r_max))
    spread = fitness.max() - fitness.min()
    if not torch.isfinite(spread) or float(spread) == 0.0:
        return torch.full_like(fitness, float(r_max))
    order = torch.argsort(torch.argsort(fitness))  # rank 0 = best (lowest)
    frac = (n - 1 - order).to(fitness.dtype) / (n - 1)
    return r_max * frac.pow(gamma)


def reflectivity_from_boltzmann(fitness: Tensor, r_max: float = 0.98,
                                T_f: Optional[float] = None) -> Tensor:
    """Boltzmann reflectivity ``r = r_max * exp(-(f - f_min) / T_f)``.

    ``T_f`` defaults to the interquartile spread of the current fitness
    values, making the map scale-free.
    """
    f_min = fitness.min()
    if T_f is None:
        q = torch.quantile(fitness, torch.tensor([0.25, 0.75],
                                                 device=fitness.device,
                                                 dtype=fitness.dtype))
        T_f = float(q[1] - q[0]) + 1e-12
    return r_max * torch.exp(-(fitness - f_min) / T_f)


def pairwise_lifshitz_forces(X: Tensor, r: Tensor, a0: float = 1e-3,
                             n_terms: int = 32) -> Tensor:
    """Net Lifshitz force on every particle from every other particle.

    Parameters
    ----------
    X : (N, D) particle positions.
    r : (N,) reflectivities in [0, 1).
    a0 : short-distance (plasma-wavelength) cutoff.

    Returns
    -------
    (N, D) tensor of force vectors ``F_i = sum_j |F|(z_ij, d_ij) * u_ij``
    where ``u_ij`` is the unit vector from i toward j (attraction).
    Fully vectorized: O(N^2 D), runs on GPU.
    """
    diff = X.unsqueeze(0) - X.unsqueeze(1)          # (N, N, D): x_j - x_i
    dist = diff.norm(dim=-1)                        # (N, N)
    z = r.unsqueeze(0) * r.unsqueeze(1)             # (N, N)
    mag = lifshitz_force(z, dist, a0=a0, n_terms=n_terms)
    eye = torch.eye(X.shape[0], device=X.device, dtype=torch.bool)
    mag = mag.masked_fill(eye, 0.0)
    unit = diff / (dist.unsqueeze(-1) + 1e-12)
    return torch.sum(mag.unsqueeze(-1) * unit, dim=1)
