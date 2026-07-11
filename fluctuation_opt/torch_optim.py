"""ZeroPointOptimizer: gradient descent on the one-loop effective potential.

The physical picture
--------------------
Quantum field theory never minimizes the bare potential ``V``: the physical
vacuum minimizes the *effective potential*, which at one loop is

    V_eff(theta) = V(theta) + (1/2) Tr log-ish corrections
                 ~ V(theta) + zero-point energy of fluctuations around theta.

Mapped to learning: the bare loss ``L(theta)`` is the classical potential,
and the parameters permanently jitter (finite data, SGD noise, test-time
distribution shift), so the *felt* loss is the fluctuation-dressed one.
For Gaussian fluctuations of amplitude sigma,

    L_sigma(theta) = E_eps[ L(theta + sigma * eps) ]
                  ~= L(theta) + (sigma^2 / 2) Tr H(theta) + O(sigma^4),

whose second term is exactly the leading heat-kernel coefficient of the
regularized zero-point energy (see ``core.spectral``).  Descending
``L_sigma`` therefore adds a fluctuation-regularized smoothing force pulling toward flat minima --
which is what generalization (and PINN robustness) wants.

Two modes
---------
- ``mode="smoothed"`` (default, first-order only, cheap, GPU-friendly):
  the gradient of ``L_sigma`` is estimated by averaging gradients at
  ``n_probes`` vacuum-fluctuated parameter copies; sigma follows the
  Matsubara quantum annealing schedule and -- like a real quantum system --
  never freezes to zero, only to the zero-point floor set by the *measured
  local stiffness* (curvature along the probes updates omega online).

- ``mode="trace"`` (exact, third-order autograd, for small models):
  adds ``zpe_coeff * grad_theta( v^T H v )`` with Rademacher ``v``, an
  unbiased single-probe estimate of ``grad Tr H`` -- the exact gradient of
  the leading zero-point term.

Usage
-----
Unlike torch's built-ins, ``step`` takes a closure that RETURNS the loss
WITHOUT calling ``backward()`` -- the optimizer drives autograd itself
(it needs to re-evaluate the loss at fluctuated parameter copies)::

    opt = ZeroPointOptimizer(model.parameters(), lr=1e-3, sigma=1e-2)
    for batch in data:
        def closure():
            return loss_fn(model, batch)
        loss = opt.step(closure)

Device-agnostic: runs wherever the parameters live (CPU or CUDA).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional

import torch

from fluctuation_opt.core.matsubara import QuantumAnnealingSchedule, thermal_variance
from fluctuation_opt.core import spectral

Tensor = torch.Tensor


def _flatten(tensors: Iterable[Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _grad_or_zeros(
    loss: Tensor,
    params: List[Tensor],
    *,
    create_graph: bool = False,
    retain_graph: Optional[bool] = None,
) -> List[Tensor]:
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=create_graph if retain_graph is None else retain_graph,
        allow_unused=True,
    )
    return [
        torch.zeros_like(p) if g is None else g
        for p, g in zip(params, grads)
    ]


class ZeroPointOptimizer(torch.optim.Optimizer):
    """Adam-based optimizer with a zero-point / flatness regularization force.

    Parameters
    ----------
    params : iterable of parameters.
    lr, betas, eps, weight_decay : as in Adam.
    mode : "smoothed" (default) or "trace"; see module docstring.
    sigma : initial vacuum fluctuation amplitude (parameter units).
    n_probes : number of fluctuated copies per step (smoothed mode).
    floor_frac : zero-point floor as a fraction of ``sigma`` (at the initial
        stiffness estimate); the quantum annealing schedule interpolates.
    tau : temperature decay constant in steps.
    adapt_omega : if True, the local stiffness omega (and with it the
        zero-point floor) is estimated online from curvature along probes.
    zpe_coeff : coefficient of the exact Tr H penalty (trace mode).
    seed : RNG seed for the vacuum noise (reproducibility).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        mode: str = "smoothed",
        sigma: float = 1e-2,
        n_probes: int = 2,
        floor_frac: float = 0.3,
        tau: float = 1000.0,
        adapt_omega: bool = True,
        zpe_coeff: float = 1e-3,
        seed: Optional[int] = None,
    ):
        if mode not in ("smoothed", "trace"):
            raise ValueError("mode must be 'smoothed' or 'trace'")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.mode = mode
        self.n_probes = int(n_probes)
        if self.n_probes < 0:
            raise ValueError("n_probes must be non-negative")
        self.zpe_coeff = float(zpe_coeff)
        self.adapt_omega = bool(adapt_omega)
        self._seed = seed
        self._generator: Optional[torch.Generator] = None
        self._t = 0

        # calibrate the quantum annealing schedule exactly as LifshitzSwarm:
        # floor/initial = 1/sqrt(coth(omega / 2 T0)) at omega = 1
        floor_frac = min(max(float(floor_frac), 1e-3), 0.999)
        target_coth = 1.0 / floor_frac**2
        x = 0.5 * math.log((target_coth + 1.0) / (target_coth - 1.0))
        T0 = 1.0 / (2.0 * x)
        var0 = float(thermal_variance(torch.tensor(1.0), T0))
        self.schedule = QuantumAnnealingSchedule(
            T0=T0, tau=float(tau), omega=1.0, scale=float(sigma) / math.sqrt(var0)
        )

    # ------------------------------------------------------------------
    def _params(self) -> List[Tensor]:
        return [p for group in self.param_groups for p in group["params"]
                if p.requires_grad]

    def _gen(self, device) -> torch.Generator:
        if self._generator is None or self._generator.device != torch.device(device):
            self._generator = torch.Generator(device=device)
            if self._seed is not None:
                self._generator.manual_seed(int(self._seed))
        return self._generator

    def current_sigma(self) -> float:
        """The vacuum fluctuation amplitude at the current step."""
        return self.schedule(self._t)

    # ------------------------------------------------------------------
    def _regularized_gradient(self, closure: Callable[[], Tensor]) -> Tensor:
        """Return (loss, list-of-effective-gradients)."""
        params = self._params()
        if not params:
            raise ValueError("ZeroPointOptimizer has no trainable parameters")
        device = params[0].device

        with torch.enable_grad():
            loss = closure()
            create_graph = self.mode == "trace"
            g0 = _grad_or_zeros(loss, params, create_graph=create_graph)

        if self.mode == "trace":
            gen = self._gen(device)
            v = [ (torch.randint(0, 2, p.shape, device=device, generator=gen)
                   .to(p.dtype) * 2.0 - 1.0) for p in params]
            gv = sum((gi * vi).sum() for gi, vi in zip(g0, v))
            Hv = _grad_or_zeros(gv, params, create_graph=True)
            s = sum((hi * vi).sum() for hi, vi in zip(Hv, v))   # ~ v^T H v
            gs = _grad_or_zeros(s, params)                       # grad of Tr H est.
            eff = [g.detach() + self.zpe_coeff * gsi.detach()
                   for g, gsi in zip(g0, gs)]
            return loss.detach(), eff

        # ---- smoothed (vacuum-dressed gradient) ----
        # Pure Monte-Carlo estimate of grad E_eps[L(theta + sigma eps)] using
        # ANTITHETIC pairs (+eps, -eps): odd-order sampling noise cancels
        # exactly, which is what lets the estimator's small mean drift
        # (the tunneling force out of sharp basins) survive the huge
        # gradient spikes of stiff directions.
        sigma = self.current_sigma()
        if sigma <= 0:
            return loss.detach(), [g.detach() for g in g0]

        gen = self._gen(device)
        n_pairs = (self.n_probes + 1) // 2
        if n_pairs == 0:
            return loss.detach(), [g.detach() for g in g0]

        # The common low-cost mode uses center + antithetic gradients for a
        # stable three-evaluation estimator. Larger probe counts are usually
        # chosen specifically for exploration, so they keep the pure
        # antithetic smoothing signal; a center pull can trap those runs in
        # the sharp basin they are meant to escape.
        include_center = n_pairs == 1
        acc = [g.detach().clone() if include_center else torch.zeros_like(g) for g in g0]
        normalizer = (2 * n_pairs + 1) if include_center else (2 * n_pairs)
        curv_sum, curv_n = 0.0, 0

        originals = [p.detach().clone() for p in params]
        for _ in range(n_pairs):
            noises = [sigma * torch.randn(p.shape, device=p.device, dtype=p.dtype,
                                          generator=gen) for p in params]
            pair_grads = []
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    for p, o, n in zip(params, originals, noises):
                        p.copy_(o + sign * n)
                with torch.enable_grad():
                    loss_p = closure()
                    gp = _grad_or_zeros(loss_p, params)
                pair_grads.append([g.detach() for g in gp])
                for a, g in zip(acc, pair_grads[-1]):
                    a.add_(g)
            with torch.no_grad():
                for p, o in zip(params, originals):
                    p.copy_(o)

            if self.adapt_omega:
                # central-difference directional curvature:
                # eps^T H eps / |eps|^2 ~= eps.(g+ - g-) / (2 |eps|^2)
                gplus, gminus = pair_grads
                num = sum(((gp_ - gm_) * n).sum()
                          for gp_, gm_, n in zip(gplus, gminus, noises))
                den = 2.0 * sum((n * n).sum() for n in noises)
                lam = float(num / (den + 1e-30))
                if lam > 0 and math.isfinite(lam):
                    curv_sum += math.sqrt(lam)
                    curv_n += 1

        # Stiffness (omega) tracking is gated to the cold regime T < omega:
        # at high temperature the dynamics are classical and exploratory, and
        # the zero-point floor -- the only thing omega controls -- is not yet
        # active.  Updating omega while hot would quench exploration noise
        # prematurely inside stiff basins.
        if curv_n > 0 and self.schedule.temperature(self._t) < self.schedule.omega:
            self.schedule.update_omega(curv_sum / curv_n)

        eff = [a / normalizer for a in acc]
        return loss.detach(), eff

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _adam_update(self, eff_grads: List[Tensor]) -> None:
        i = 0
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = (group["lr"], group["betas"],
                                     group["eps"], group["weight_decay"])
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                g = eff_grads[i]
                i += 1
                if wd != 0.0:
                    g = g + wd * p
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                mh = m / (1 - b1 ** state["step"])
                vh = v / (1 - b2 ** state["step"])
                p.addcdiv_(mh, vh.sqrt().add_(eps), value=-lr)

    def step(self, closure: Callable[[], Tensor]) -> Tensor:  # type: ignore[override]
        """One optimization step.  ``closure`` returns the loss tensor
        (do NOT call ``backward()`` inside it)."""
        if closure is None:
            raise ValueError("ZeroPointOptimizer requires a closure returning the loss")
        loss, eff = self._regularized_gradient(closure)
        self._adam_update(eff)
        self._t += 1
        return loss

    # ------------------------------------------------------------------
    def zero_point_energy(self, closure: Callable[[], Tensor],
                          s: float = 0.1, n_probes: int = 4, m: int = 15) -> float:
        """Diagnostic: heat-kernel-regularized zero-point energy
        ``(1/2) Tr[sqrt(H_+) e^{-s sqrt(H_+)}]`` of the current point,
        estimated matrix-free with stochastic Lanczos quadrature.
        Lower = flatter minimum."""
        params = self._params()
        device, dtype = params[0].device, params[0].dtype
        dim = sum(p.numel() for p in params)

        with torch.enable_grad():
            loss = closure()
            g = torch.autograd.grad(loss, params, create_graph=True)

        def hvp(vec: Tensor) -> Tensor:
            vs, off = [], 0
            for p in params:
                n = p.numel()
                vs.append(vec[off:off + n].view_as(p))
                off += n
            gv = sum((gi * vi).sum() for gi, vi in zip(g, vs))
            Hv = torch.autograd.grad(gv, params, retain_graph=True)
            return _flatten([h.detach() for h in Hv])

        zpe = spectral.zero_point_energy(
            hvp, dim, s=s, n_probes=n_probes, m=m, device=device, dtype=dtype,
            generator=self._gen(device),
        )
        return float(zpe)
