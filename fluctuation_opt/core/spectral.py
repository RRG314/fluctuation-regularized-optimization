"""Regularized mode sums and stochastic spectral estimators.

Physics background
------------------
Around a point ``theta`` in parameter space, expand the loss
``L(theta + delta) ~= L + g.delta + (1/2) delta^T H delta``.  Treating each
Hessian eigendirection as a harmonic mode of a fluctuating field, the mode
frequencies are ``omega_i = sqrt(lambda_i)`` and the vacuum (zero-point)
energy of the landscape is the Casimir-style mode sum

    E_0 = (1/2) * sum_i omega_i .

Exactly as in Casimir physics this sum is dominated by the ultraviolet
(large-eigenvalue) end and must be regularized before it carries usable
information.  We provide the two standard regulators:

- heat-kernel (exponential) regulator:  E_0(s) = (1/2) sum omega e^{-s omega}
- spectral zeta function:               zeta_H(s) = sum (lambda/mu^2)^{-s}

For high-dimensional models the spectrum is never available explicitly, so
``Tr f(H)`` is estimated with Hutchinson probing and stochastic Lanczos
quadrature (SLQ), which only require Hessian-vector products.

All functions are pure torch and device-agnostic.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

Tensor = torch.Tensor
HVP = Callable[[Tensor], Tensor]  # v -> H @ v


# --------------------------------------------------------------------------
# Deterministic regularized sums (given an explicit spectrum)
# --------------------------------------------------------------------------

def mode_frequencies(eigs: Tensor) -> Tensor:
    """Mode frequencies omega_i = sqrt(max(lambda_i, 0)).

    Negative eigenvalues correspond to unstable (non-oscillatory) directions
    and carry no zero-point energy; they are clamped to zero, mirroring the
    fact that only bound modes contribute to the Casimir sum.
    """
    return torch.sqrt(torch.clamp(eigs, min=0.0))


def heat_kernel_zpe(eigs: Tensor, s: float) -> Tensor:
    """Heat-kernel regularized zero-point energy.

    ``E_0(s) = (1/2) sum_i omega_i exp(-s * omega_i)`` with ``s > 0`` the
    regulator (inverse UV cutoff).  ``s -> 0`` recovers the bare divergent
    sum; a finite ``s`` suppresses stiff (sharp-curvature) modes smoothly,
    exactly like the exponential regulator in Casimir's original 1948
    computation.
    """
    if s <= 0:
        raise ValueError("heat-kernel regulator s must be > 0")
    omega = mode_frequencies(eigs)
    return 0.5 * torch.sum(omega * torch.exp(-s * omega))


def zeta_spectral_sum(eigs: Tensor, s: float, mu: float = 1.0) -> Tensor:
    """Spectral zeta function  zeta_H(s) = sum_i (lambda_i / mu^2)^(-s).

    ``mu`` is the renormalization scale.  Eigenvalues <= 0 are excluded
    (they are not part of the oscillatory spectrum).  For ``s = -1/2`` this
    formally equals ``2/mu * E_0`` -- the zeta-regularized vacuum energy.
    """
    lam = eigs[eigs > 0] / (mu * mu)
    if lam.numel() == 0:
        return torch.zeros((), device=eigs.device, dtype=eigs.dtype)
    return torch.sum(lam.pow(-s))


def casimir_energy_density(eigs_constrained: Tensor, eigs_free: Tensor, s: float) -> Tensor:
    """Regularized energy *difference* between two spectra.

    The physically meaningful Casimir quantity is never an absolute vacuum
    energy but the difference between a constrained and a reference
    spectrum; the regulator dependence cancels in the difference as s -> 0.
    """
    return heat_kernel_zpe(eigs_constrained, s) - heat_kernel_zpe(eigs_free, s)


# --------------------------------------------------------------------------
# Matrix-free stochastic estimators (Hutchinson & SLQ)
# --------------------------------------------------------------------------

def _rademacher(dim: int, device, dtype, generator: Optional[torch.Generator]) -> Tensor:
    v = torch.randint(0, 2, (dim,), device=device, generator=generator).to(dtype)
    return 2.0 * v - 1.0


def hutchinson_trace(
    hvp: HVP,
    dim: int,
    n_probes: int = 16,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Hutchinson estimator of ``Tr H`` using Rademacher probes.

    ``E[v^T H v] = Tr H`` for ``v`` with i.i.d. +-1 entries.  ``Tr H`` is the
    leading (a_0) term of the heat-kernel expansion of the vacuum energy and
    equals the curvature seen by isotropic Gaussian parameter fluctuations:
    ``E[L(theta + sigma eps)] - L(theta) ~= (sigma^2 / 2) Tr H``.
    """
    device = device or torch.device("cpu")
    total = torch.zeros((), device=device, dtype=dtype)
    for _ in range(n_probes):
        v = _rademacher(dim, device, dtype, generator)
        total = total + torch.dot(v, hvp(v))
    return total / n_probes


def lanczos_tridiag(
    hvp: HVP,
    dim: int,
    m: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    v0: Optional[Tensor] = None,
):
    """m-step Lanczos: returns (alphas, betas, actual_steps).

    Builds the tridiagonal Jacobi matrix whose eigenvalues (Ritz values)
    approximate the extremal spectrum of H.  Full reorthogonalization is
    used -- affordable because m is small (typically <= 30).
    """
    device = device or torch.device("cpu")
    m = min(m, dim)
    if v0 is None:
        v0 = torch.randn(dim, device=device, dtype=dtype, generator=generator)
    v = v0 / v0.norm()
    V = [v]
    alphas, betas = [], []
    beta = None
    for j in range(m):
        w = hvp(V[j])
        alpha = torch.dot(V[j], w)
        alphas.append(alpha)
        w = w - alpha * V[j]
        if j > 0:
            w = w - betas[-1] * V[j - 1]
        # full reorthogonalization for numerical stability
        for u in V:
            w = w - torch.dot(u, w) * u
        beta = w.norm()
        if j < m - 1:
            if beta < 1e-10 * max(1.0, float(abs(alpha))):
                break
            betas.append(beta)
            V.append(w / beta)
    a = torch.stack(alphas)
    b = torch.stack(betas) if betas else torch.zeros(0, device=device, dtype=dtype)
    return a, b, len(alphas)


def _tridiag_eig(a: Tensor, b: Tensor):
    """Eigendecomposition of the symmetric tridiagonal (Jacobi) matrix."""
    k = a.numel()
    T = torch.diag(a)
    if b.numel() > 0:
        idx = torch.arange(b.numel(), device=a.device)
        T[idx, idx + 1] = b
        T[idx + 1, idx] = b
    evals, evecs = torch.linalg.eigh(T)
    return evals, evecs


def slq_spectral_sum(
    hvp: HVP,
    dim: int,
    f: Callable[[Tensor], Tensor],
    n_probes: int = 8,
    m: int = 20,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Stochastic Lanczos quadrature estimate of ``Tr f(H)``.

    For each random probe the Lanczos Jacobi matrix yields Ritz values
    ``theta_k`` and quadrature weights ``tau_k^2`` (squared first components
    of its eigenvectors); then

        Tr f(H) ~= (dim / n_probes) * sum_probes sum_k tau_k^2 f(theta_k).

    This is the workhorse for regularized vacuum energies of models whose
    Hessian is only accessible through Hessian-vector products.
    """
    device = device or torch.device("cpu")
    total = torch.zeros((), device=device, dtype=dtype)
    for _ in range(n_probes):
        v0 = _rademacher(dim, device, dtype, generator)
        a, b, _ = lanczos_tridiag(hvp, dim, m, device, dtype, generator, v0=v0)
        theta, U = _tridiag_eig(a, b)
        tau2 = U[0, :] ** 2
        total = total + torch.sum(tau2 * f(theta))
    return total * dim / n_probes


def zero_point_energy(
    hvp: HVP,
    dim: int,
    s: float = 0.1,
    n_probes: int = 8,
    m: int = 20,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Heat-kernel-regularized zero-point energy of a loss landscape,
    ``E_0(s) = (1/2) Tr[ sqrt(H_+) exp(-s sqrt(H_+)) ]``, matrix-free via SLQ.

    This is the scalar the Casimir gradient optimizer implicitly descends;
    exposed here as a diagnostic ("how much vacuum energy does my minimum
    hold?" -- lower means flatter).
    """

    def f(lam: Tensor) -> Tensor:
        omega = torch.sqrt(torch.clamp(lam, min=0.0))
        return 0.5 * omega * torch.exp(-s * omega)

    return slq_spectral_sum(hvp, dim, f, n_probes, m, device, dtype, generator)
