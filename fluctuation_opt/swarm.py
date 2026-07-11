"""LifshitzSwarm: population black-box optimization via Lifshitz couplings.

The physical picture
--------------------
Each candidate solution is an imperfect mirror suspended in a fluctuating
vacuum.  Its *reflectivity* is set by its fitness (good solution -> nearly
perfect mirror, bad solution -> nearly transparent).  Every pair of mirrors
feels the fluctuation-induced (Casimir-Lifshitz) attraction

    |F_ij| = Li_2(r_i r_j) / (4 pi (d_ij + a0)^2),

derived exactly for a 1D scalar field in ``core.lifshitz``.  Because the
coupling depends on the *product* of reflectivities, structure emerges that
is qualitatively different from PSO or gravitational search:

- pairs/clusters of good solutions bind strongly to each other (multi-modal
  niching for free) instead of everything collapsing onto one global best;
- bad solutions are nearly transparent -- they feel and exert almost no
  force, and drift ballistically as free explorers;
- the force is short-ranged (1/d^2 from a 1/d energy well, saturated below
  the plasma cutoff a0), so distant clusters do not disturb each other.

On top of the deterministic forces, every mirror is shaken by the vacuum:
the noise amplitude follows the quantum-oscillator variance
``(1/2 omega) coth(omega / 2T)`` with an annealed temperature (see
``core.matsubara``).  Crucially the noise never freezes to zero -- it
converges to the zero-point amplitude -- so late-stage escape from spurious
sharp basins remains possible, biasing the population toward flat, robust
optima.

Everything is vectorized torch; pass ``device="cuda"`` to run on GPU.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Union

import torch

from fluctuation_opt.core.lifshitz import (
    pairwise_lifshitz_forces,
    reflectivity_from_boltzmann,
    reflectivity_from_rank,
)
from fluctuation_opt.core.matsubara import QuantumAnnealingSchedule, thermal_variance

Tensor = torch.Tensor


class LifshitzSwarm:
    """Lifshitz-coupled swarm optimizer (minimization).

    Parameters
    ----------
    bounds : sequence of (low, high) pairs, one per dimension.
    n_particles : population size.
    inertia : velocity memory ``mu`` in [0, 1).
    eta : force-to-velocity coupling (mobility).
    a0 : plasma-wavelength cutoff in *normalized* units (search space is
        internally mapped to the unit cube); caps the contact force.
    r_max : maximal reflectivity (< 1; perfect mirrors are unphysical and
        would give infinite n=1 Matsubara couplings at contact).
    gamma : contrast exponent of the rank -> reflectivity map.
    reflectivity : "rank" (default, scale-free) or "boltzmann".
    noise : initial vacuum-noise amplitude in normalized units.
    floor_frac : fraction of ``noise`` that survives as the zero-point floor
        (T -> 0 limit).  Set the *quantum-ness* of the annealing.
    max_step : per-iteration displacement cap in normalized units.
    anchor_best : if True, the best-ever solution acts as an additional
        fixed perfect mirror that gently binds the population.
    device, dtype, seed : usual torch controls; fully GPU-compatible.
    """

    def __init__(
        self,
        bounds: Sequence[Sequence[float]],
        n_particles: int = 40,
        inertia: float = 0.7,
        eta: float = 0.05,
        a0: float = 0.05,
        r_max: float = 0.98,
        gamma: float = 2.0,
        reflectivity: str = "rank",
        noise: float = 0.25,
        floor_frac: float = 0.35,
        max_step: float = 0.15,
        anchor_best: bool = True,
        quench_frac: float = 0.25,
        quench_contraction: float = 0.8,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        seed: Optional[int] = None,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        b = torch.as_tensor(bounds, device=self.device, dtype=dtype)
        if b.ndim != 2 or b.shape[1] != 2 or (b[:, 1] <= b[:, 0]).any():
            raise ValueError("bounds must be a (D, 2) array with high > low")
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.dim = b.shape[0]
        self.n = int(n_particles)
        self.inertia = float(inertia)
        self.eta = float(eta)
        self.a0 = float(a0)
        self.r_max = float(r_max)
        self.gamma = float(gamma)
        self.reflectivity_mode = reflectivity
        self.max_step = float(max_step)
        self.anchor_best = bool(anchor_best)
        self.quench_frac = min(max(float(quench_frac), 0.0), 0.9)
        self.quench_contraction = min(max(float(quench_contraction), 0.1), 0.999)

        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

        # ---- quantum annealing schedule calibration -------------------
        # floor sigma / initial sigma = 1 / sqrt(coth(omega / 2 T0)); choose
        # omega = 1 and solve coth(x) = 1/floor_frac^2 for x = 1/(2 T0).
        floor_frac = min(max(float(floor_frac), 1e-3), 0.999)
        target_coth = 1.0 / floor_frac**2
        x = 0.5 * math.log((target_coth + 1.0) / (target_coth - 1.0))  # arccoth
        self._omega = 1.0
        self._T0 = self._omega / (2.0 * x)
        var0 = float(thermal_variance(torch.tensor(self._omega), self._T0))
        scale = float(noise) / math.sqrt(var0)
        # tau is set per-run (depends on max_iter); stored T0/omega/scale here
        self._noise_scale = scale

    # ------------------------------------------------------------------
    def _reflectivities(self, fitness: Tensor) -> Tensor:
        if self.reflectivity_mode == "rank":
            return reflectivity_from_rank(fitness, r_max=self.r_max, gamma=self.gamma)
        elif self.reflectivity_mode == "boltzmann":
            return reflectivity_from_boltzmann(fitness, r_max=self.r_max)
        raise ValueError(f"unknown reflectivity mode {self.reflectivity_mode!r}")

    def _denorm(self, Xn: Tensor) -> Tensor:
        return self.lo + Xn * (self.hi - self.lo)

    def _rand(self, *shape) -> Tensor:
        return torch.rand(*shape, device=self.device, dtype=self.dtype,
                          generator=self.generator)

    def _randn(self, *shape) -> Tensor:
        return torch.randn(*shape, device=self.device, dtype=self.dtype,
                           generator=self.generator)

    # ------------------------------------------------------------------
    def minimize(
        self,
        f: Callable[[Tensor], Tensor],
        max_iter: int = 200,
        vectorized: bool = True,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """Minimize ``f``.

        ``f`` receives a (N, D) tensor of candidate points (on the swarm's
        device) and must return a (N,) tensor of fitness values when
        ``vectorized=True``; otherwise a scalar function of a 1D tensor is
        wrapped automatically.

        Returns a dict with ``x`` (best point), ``fun`` (best value),
        ``history`` (best value per iteration) and ``n_evals``.
        """
        if not vectorized:
            scalar_f = f

            def f(X: Tensor) -> Tensor:  # noqa: F811
                return torch.stack([torch.as_tensor(scalar_f(x), device=self.device,
                                                    dtype=self.dtype) for x in X])

        schedule = QuantumAnnealingSchedule(
            T0=self._T0, tau=max(max_iter / 6.0, 1.0), omega=self._omega,
            scale=self._noise_scale,
        )

        X = self._rand(self.n, self.dim)                      # normalized positions
        V = torch.zeros_like(X)
        fit = f(self._denorm(X)).to(self.dtype)
        n_evals = self.n

        best_i = int(torch.argmin(fit))
        best_x = X[best_i].clone()
        best_f = fit[best_i].clone()
        history = [float(best_f)]

        quench_start = int(max_iter * (1.0 - self.quench_frac))
        quench_spread = None  # collapse-cloud width (1/5th-rule adapted)

        for t in range(max_iter):
            r = self._reflectivities(fit)
            quenching = t >= quench_start

            if quenching:
                # Decoherence quench ("measurement collapse"): the vacuum is
                # switched off and the mirror cloud becomes a Gaussian
                # uncertainty cloud around the best solution whose width
                # follows success-based (1/5th-rule) contraction --
                # exponential local refinement of the located basin.
                if quench_spread is None:
                    quench_spread = float(
                        (X - best_x.unsqueeze(0)).norm(dim=1).mean()
                        / math.sqrt(self.dim)) + 1e-12
                X = (best_x.unsqueeze(0)
                     + quench_spread * self._randn(self.n, self.dim))
                X = X.clamp(0.0, 1.0)
            else:
                if self.anchor_best:
                    Xa = torch.cat([X, best_x.unsqueeze(0)], dim=0)
                    ra = torch.cat([r, torch.tensor([self.r_max],
                                                    device=self.device,
                                                    dtype=self.dtype)])
                    F = pairwise_lifshitz_forces(Xa, ra, a0=self.a0)[: self.n]
                else:
                    F = pairwise_lifshitz_forces(X, r, a0=self.a0)

                # per-particle vacuum noise: a good mirror (high reflectivity)
                # is a heavy, strongly-coupled object and fluctuates less; a
                # transparent (bad) mirror decouples from the cavity and
                # explores ballistically.
                sigma_i = schedule(t) * (1.0 - 0.7 * (r / self.r_max)).unsqueeze(1)
                V = (self.inertia * V + self.eta * F
                     + sigma_i * self._randn(self.n, self.dim))

                # cap per-particle displacement (mirror inertia is finite)
                step_norm = V.norm(dim=1, keepdim=True)
                V = V * torch.clamp(self.max_step / (step_norm + 1e-12), max=1.0)

                X = X + V
                # reflective boundaries
                over_lo, over_hi = X < 0.0, X > 1.0
                X = torch.where(over_lo, -X, X)
                X = torch.where(over_hi, 2.0 - X, X)
                X = X.clamp(0.0, 1.0)
                V = torch.where(over_lo | over_hi, -V, V)

            fit = f(self._denorm(X)).to(self.dtype)
            n_evals += self.n

            i = int(torch.argmin(fit))
            improved = bool(fit[i] < best_f)
            if improved:
                best_f = fit[i].clone()
                best_x = X[i].clone()
            history.append(float(best_f))

            if quenching and quench_spread is not None:
                # 1/5th-rule: expand the collapse cloud on success, contract
                # on failure (expansion^1 * contraction^4 = 1 at 20% success)
                quench_spread *= (self.quench_contraction ** -0.25 if improved
                                  else self.quench_contraction)

            if callback is not None:
                callback({"iter": t, "best_f": float(best_f),
                          "sigma": 0.0 if quenching else schedule(t),
                          "X": self._denorm(X), "fitness": fit})

        return {
            "x": self._denorm(best_x),
            "fun": float(best_f),
            "history": history,
            "n_evals": n_evals,
        }
