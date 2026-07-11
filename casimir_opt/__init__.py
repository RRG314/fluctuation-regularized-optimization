"""casimir-opt: optimization algorithms built from the mathematics of the Casimir effect.

Two optimizers share one physical core:

- :class:`casimir_opt.torch_optim.CasimirOptimizer` -- a gradient-based (PyTorch)
  optimizer that descends the *one-loop effective potential* of the loss
  landscape instead of the bare loss, i.e. it feels the regularized zero-point
  energy of parameter fluctuations and is thereby pulled toward flat minima.

- :class:`casimir_opt.swarm.CasimirSwarm` -- a population black-box optimizer in
  which candidate solutions are imperfect mirrors coupled through the Lifshitz
  formula: fitness maps to reflectivity, and the resulting fluctuation-induced
  attraction is short-ranged (polylog / inverse-square in the 1D scalar-field
  model), unlike the long-range gravity of PSO-style methods.

Shared core (:mod:`casimir_opt.core`):

- heat-kernel / zeta regularization of mode sums (``core.spectral``)
- stochastic Lanczos quadrature spectral estimators (``core.spectral``)
- Matsubara finite-temperature machinery and the quantum annealing schedule
  that freezes to *zero-point* (not zero) fluctuations (``core.matsubara``)
- Lifshitz reflectivity couplings and polylogarithms (``core.lifshitz``)

PINN support (:mod:`casimir_opt.pinn`): Casimir pressure balancing of
multi-term physics-informed losses, plus small device-agnostic helpers.

Everything is device-agnostic: pass CUDA tensors / ``device="cuda"`` and the
whole library runs on GPU.
"""

from casimir_opt.core import lifshitz, matsubara, spectral
from casimir_opt.core.matsubara import QuantumAnnealingSchedule
from casimir_opt.swarm import CasimirSwarm
from casimir_opt.torch_optim import CasimirOptimizer
from casimir_opt.pinn import CasimirPressureBalancer, MLP, partial_derivative

__version__ = "0.1.0"

__all__ = [
    "CasimirOptimizer",
    "CasimirSwarm",
    "CasimirPressureBalancer",
    "QuantumAnnealingSchedule",
    "MLP",
    "partial_derivative",
    "spectral",
    "matsubara",
    "lifshitz",
]
