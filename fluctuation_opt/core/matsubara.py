"""Finite-temperature (Matsubara) machinery and the quantum annealing schedule.

Physics background
------------------
At temperature ``T`` a quantum harmonic mode of frequency ``omega`` has
free energy

    f(omega, T) = T log(2 sinh(omega / 2T))
                = omega/2 + T log(1 - exp(-omega/T)),

i.e. the zero-point energy plus a thermal part, and fluctuation variance

    <x^2> = (1 / 2 omega) * coth(omega / 2T).

Two limits matter for optimization:

- high T:   <x^2> -> T / omega^2      (classical thermal noise, simulated
                                       annealing regime)
- T -> 0:   <x^2> -> 1 / (2 omega)    (zero-point fluctuations: *never zero*)

Classical simulated annealing freezes completely; a Casimir/quantum schedule
freezes only down to the zero-point floor, whose size is set by the local
stiffness ``omega``.  Sharp minima (large omega) get small residual noise,
flat minima (small omega) keep larger exploratory fluctuations -- a built-in,
physically derived flat-minimum bias that both optimizers in this package
share.

The Matsubara frequencies ``xi_n = 2 pi n T`` are the discrete imaginary
frequencies over which finite-temperature Lifshitz/Casimir quantities are
summed; the swarm optimizer uses them to thermalize its Lifshitz couplings.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

Tensor = torch.Tensor
Scalar = Union[float, Tensor]


def matsubara_frequencies(T: float, n_max: int, device=None, dtype=torch.float32) -> Tensor:
    """Bosonic Matsubara frequencies ``xi_n = 2 pi n T`` for n = 0..n_max."""
    n = torch.arange(0, n_max + 1, device=device, dtype=dtype)
    return 2.0 * math.pi * T * n


def mode_free_energy(omega: Scalar, T: float) -> Tensor:
    """Free energy of one harmonic mode, ``omega/2 + T log(1 - e^{-omega/T})``."""
    omega = torch.as_tensor(omega)
    if not torch.is_floating_point(omega):
        omega = omega.to(torch.get_default_dtype())
    if T <= 0:
        return omega / 2.0
    x = omega / T
    return omega / 2.0 + T * torch.log(-torch.expm1(-x))


def thermal_variance(omega: Scalar, T: float) -> Tensor:
    """Position variance of a quantum mode: ``(1/2 omega) coth(omega / 2T)``.

    Numerically safe implementation.  Limits: ``T/omega^2`` for large T,
    ``1/(2 omega)`` for ``T -> 0`` (the zero-point floor).
    """
    omega = torch.as_tensor(omega)
    if not torch.is_floating_point(omega):
        omega = omega.to(torch.get_default_dtype())
    omega = torch.clamp(omega, min=1e-12)
    if T <= 0:
        return 1.0 / (2.0 * omega)
    x = omega / (2.0 * T)
    # coth(x) = 1 + 2/(e^{2x} - 1), stable for large x; series for small x
    coth = torch.where(
        x > 1e-4,
        1.0 + 2.0 / torch.expm1(2.0 * x),
        1.0 / x + x / 3.0,
    )
    return coth / (2.0 * omega)


class QuantumAnnealingSchedule:
    """Noise schedule sigma(t) derived from the quantum-oscillator variance.

    ``sigma(t)^2 = scale^2 * thermal_variance(omega, T(t))`` with the
    temperature decaying as ``T(t) = T0 / (1 + t / tau)``.

    Unlike classical annealing, ``sigma`` never reaches zero: it converges to
    the zero-point amplitude ``scale / sqrt(2 omega)``.  ``omega`` is the
    stiffness scale of the problem; it may be updated online (e.g. from
    curvature probes) via :meth:`update_omega`.

    Parameters
    ----------
    T0 : initial temperature.
    tau : decay time constant (in steps).
    omega : characteristic mode frequency (stiffness) of the landscape.
    scale : overall amplitude multiplier mapping the dimensionless model
        onto the parameter space of the problem.
    """

    def __init__(self, T0: float = 1.0, tau: float = 100.0, omega: float = 1.0,
                 scale: float = 1.0):
        if T0 < 0 or tau <= 0 or omega <= 0 or scale < 0:
            raise ValueError("T0>=0, tau>0, omega>0, scale>=0 required")
        self.T0 = T0
        self.tau = tau
        self.omega = omega
        self.scale = scale
        # exponential moving average state for online stiffness updates
        self._omega_ema_beta = 0.9

    def temperature(self, t: Union[int, float]) -> float:
        return self.T0 / (1.0 + float(t) / self.tau)

    def sigma(self, t: Union[int, float]) -> float:
        var = thermal_variance(self.omega, self.temperature(t))
        return self.scale * math.sqrt(float(var))

    def zero_point_sigma(self) -> float:
        """The residual (t -> infinity) fluctuation amplitude."""
        return self.scale * math.sqrt(1.0 / (2.0 * self.omega))

    def update_omega(self, omega_estimate: float) -> None:
        """EMA update of the stiffness scale from an online curvature probe."""
        if omega_estimate > 0 and math.isfinite(omega_estimate):
            b = self._omega_ema_beta
            self.omega = b * self.omega + (1.0 - b) * omega_estimate

    def __call__(self, t: Union[int, float]) -> float:
        return self.sigma(t)
