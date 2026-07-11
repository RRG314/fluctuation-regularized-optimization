"""Fluctuation-regularized optimization mechanisms.

The package implements Lifshitz-style swarm coupling, zero-point-smoothed
gradient updates, and gradient-pressure loss balancing. Casimir/Lifshitz
physics motivates part of the math, but the package is not a full physical
Casimir simulator. Two optimizers share one mathematical core:

- :class:`fluctuation_opt.torch_optim.ZeroPointOptimizer` -- a gradient-based (PyTorch)
  optimizer that descends the *one-loop effective potential* of the loss
  landscape instead of the bare loss, i.e. it feels the regularized zero-point
  energy of parameter fluctuations and is thereby pulled toward flat minima.

- :class:`fluctuation_opt.swarm.LifshitzSwarm` -- a population black-box optimizer in
  which candidate solutions are imperfect mirrors coupled through the Lifshitz
  formula: fitness maps to reflectivity, and the resulting fluctuation-induced
  attraction is short-ranged (polylog / inverse-square in the 1D scalar-field
  model), unlike the long-range gravity of PSO-style methods.

Shared core (:mod:`fluctuation_opt.core`):

- heat-kernel / zeta regularization of mode sums (``core.spectral``)
- stochastic Lanczos quadrature spectral estimators (``core.spectral``)
- Matsubara finite-temperature machinery and the quantum annealing schedule
  that freezes to *zero-point* (not zero) fluctuations (``core.matsubara``)
- Lifshitz reflectivity couplings and polylogarithms (``core.lifshitz``)

PINN support (:mod:`fluctuation_opt.pinn`): gradient-pressure balancing of
multi-term physics-informed losses, plus small device-agnostic helpers.

Everything is device-agnostic: pass CUDA tensors / ``device="cuda"`` and the
whole library runs on GPU.
"""

from fluctuation_opt.core import lifshitz, matsubara, spectral
from fluctuation_opt.core.matsubara import QuantumAnnealingSchedule
from fluctuation_opt.swarm import LifshitzSwarm
from fluctuation_opt.torch_optim import ZeroPointOptimizer
from fluctuation_opt.pinn import GradientPressureBalancer, MLP, partial_derivative

__version__ = "0.1.0"

__all__ = [
    "ZeroPointOptimizer",
    "LifshitzSwarm",
    "GradientPressureBalancer",
    "QuantumAnnealingSchedule",
    "MLP",
    "partial_derivative",
    "spectral",
    "matsubara",
    "lifshitz",
]
