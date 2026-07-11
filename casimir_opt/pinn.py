"""PINN utilities: Casimir pressure balancing of multi-term physics losses.

The physical picture
--------------------
A physics-informed neural network minimizes a sum of competing terms

    L = w_r * L_residual + w_b * L_boundary + w_i * L_initial + ...

Each term restricts a different part of function space -- exactly like the
plates of a Casimir cavity restrict different field modes.  A boundary in a
fluctuating field feels *radiation pressure* proportional to the mode flux
hitting it; a cavity is in mechanical equilibrium only when pressures on all
boundaries balance.  A PINN trains well only when the "pressure" each loss
term exerts on the parameters -- its gradient flux

    P_k = || grad_theta L_k ||  (optionally spectrally weighted by the
    term's local stiffness omega_k = sqrt(v^T H_k v / dim))

is balanced across terms; otherwise the stiffest term (usually the PDE
residual, whose differential operator amplifies high frequencies) crushes
the boundary terms, the classic PINN failure mode.

:class:`CasimirPressureBalancer` measures these pressures during training
and adapts the weights ``w_k`` so all terms push equally:
``w_k ∝ mean_pressure / P_k`` (EMA-smoothed, renormalized).

Also provided: a small device-agnostic tanh :class:`MLP` and the
:func:`partial_derivative` autograd helper for building PDE residuals.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor


class CasimirPressureBalancer:
    """Adaptive loss weighting by radiation-pressure balance.

    Parameters
    ----------
    params : model parameters (iterable; stored as a list).
    n_terms : number of loss terms.
    ema : smoothing of pressure estimates in [0, 1).
    update_every : recompute pressures every k calls (grads per term cost
        one backward pass each).
    mode : "grad" (P_k = grad norm) or "spectral"
        (P_k = grad norm * sqrt(directional curvature), one extra
        double-backward per term -- for small models).
    w_min, w_max : clamp on weights, keeps every boundary present.
    generator : optional torch.Generator for the spectral probes.

    Call with a list of per-term loss tensors (graph attached); returns the
    weighted total loss ready for the optimizer.  Weights are detached
    scalars, normalized to sum to ``n_terms`` (so the magnitude of the total
    loss stays comparable to the unweighted sum).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        n_terms: int,
        ema: float = 0.9,
        update_every: int = 10,
        mode: str = "grad",
        w_min: float = 1e-3,
        w_max: float = 1e3,
        generator: Optional[torch.Generator] = None,
    ):
        if mode not in ("grad", "spectral"):
            raise ValueError("mode must be 'grad' or 'spectral'")
        self.params = [p for p in params if p.requires_grad]
        self.n_terms = int(n_terms)
        if self.n_terms <= 0:
            raise ValueError("n_terms must be positive")
        self.ema = float(ema)
        self.update_every = max(1, int(update_every))
        self.mode = mode
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.generator = generator
        self._t = 0
        self._pressure: Optional[Tensor] = None
        device = self.params[0].device if self.params else "cpu"
        self.weights = torch.ones(self.n_terms, device=device)

    # ------------------------------------------------------------------
    def _term_pressure(self, loss_k: Tensor) -> float:
        create_graph = self.mode == "spectral"
        grads = torch.autograd.grad(loss_k, self.params, retain_graph=True,
                                    create_graph=create_graph, allow_unused=True)
        grad_pairs = [(p, g) for p, g in zip(self.params, grads) if g is not None]
        if not grad_pairs:
            return 0.0
        term_params = [p for p, _ in grad_pairs]
        grads = [g for _, g in grad_pairs]
        gnorm = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
        if self.mode == "grad":
            return float(gnorm)
        # spectral: multiply by local stiffness along a Rademacher probe
        device = self.params[0].device
        v = [(torch.randint(0, 2, g.shape, device=device,
                            generator=self.generator).to(g.dtype) * 2 - 1)
             for g in grads]
        gv = sum((gi * vi).sum() for gi, vi in zip(grads, v))
        Hv = torch.autograd.grad(gv, term_params, retain_graph=True,
                                 allow_unused=True)
        num = sum(
            torch.zeros((), device=device, dtype=gnorm.dtype)
            if h is None else (h.detach() * vi).sum()
            for h, vi in zip(Hv, v)
        )
        dim = sum(vi.numel() for vi in v)
        omega = torch.sqrt(torch.clamp(num / dim, min=0.0)) + 1e-12
        return float(gnorm * omega)

    def update(self, losses: Sequence[Tensor]) -> None:
        """Measure pressures and rebalance the weights."""
        p = torch.tensor([self._term_pressure(l) for l in losses],
                         device=self.weights.device, dtype=self.weights.dtype)
        p = torch.clamp(p, min=1e-12)
        if self._pressure is None:
            self._pressure = p
        else:
            self._pressure = self.ema * self._pressure + (1 - self.ema) * p
        w = self._pressure.mean() / self._pressure
        w = torch.clamp(w, self.w_min, self.w_max)
        self.weights = w * (self.n_terms / w.sum())

    def __call__(self, losses: Sequence[Tensor]) -> Tensor:
        if len(losses) != self.n_terms:
            raise ValueError(f"expected {self.n_terms} loss terms, got {len(losses)}")
        self.weights = self.weights.to(device=losses[0].device, dtype=losses[0].dtype)
        if self._t % self.update_every == 0:
            self.update(losses)
        self._t += 1
        total = losses[0].new_zeros(())
        for w, l in zip(self.weights, losses):
            total = total + w.to(device=l.device, dtype=l.dtype) * l
        return total


# --------------------------------------------------------------------------
# Small helpers for building PINNs
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """Small tanh MLP, Xavier-initialized -- the standard PINN backbone.

    Device-agnostic: ``MLP([1, 64, 64, 1]).to("cuda")`` just works.
    """

    def __init__(self, sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            lin = nn.Linear(sizes[i], sizes[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def partial_derivative(u: Tensor, x: Tensor, order: int = 1) -> Tensor:
    """n-th derivative of ``u`` w.r.t. ``x`` via autograd (graph retained).

    ``x`` must have ``requires_grad=True`` and ``u`` must be computed
    from it.
    """
    du = u
    for _ in range(order):
        du = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(du),
                                 create_graph=True)[0]
    return du
