import math

import pytest
import torch

from casimir_opt.core import spectral


def make_spd(dim=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    H = A @ A.T / dim + 0.1 * torch.eye(dim, dtype=torch.float64)
    return H


def test_mode_frequencies_clamps_negative():
    eigs = torch.tensor([4.0, -1.0, 0.0, 9.0])
    omega = spectral.mode_frequencies(eigs)
    assert torch.allclose(omega, torch.tensor([2.0, 0.0, 0.0, 3.0]))


def test_heat_kernel_zpe_matches_manual():
    eigs = torch.tensor([1.0, 4.0], dtype=torch.float64)
    s = 0.5
    expected = 0.5 * (1.0 * math.exp(-0.5) + 2.0 * math.exp(-1.0))
    assert float(spectral.heat_kernel_zpe(eigs, s)) == pytest.approx(expected)


def test_heat_kernel_regulator_suppresses_uv():
    """Stiff modes must contribute LESS as the regulator grows -- the defining
    property of the Casimir regularization."""
    eigs = torch.tensor([1e6], dtype=torch.float64)
    e_small = float(spectral.heat_kernel_zpe(eigs, 1e-6))
    e_large = float(spectral.heat_kernel_zpe(eigs, 1e-2))
    assert e_large < e_small


def test_zeta_spectral_sum_positive_spectrum_only():
    eigs = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float64)
    val = float(spectral.zeta_spectral_sum(eigs, s=1.0))
    assert val == pytest.approx(1.0 + 0.5)


def test_casimir_energy_density_difference_sign():
    """Constraining (stiffening) the spectrum raises the regularized vacuum
    energy; the difference must be positive."""
    free = torch.ones(10, dtype=torch.float64)
    constrained = 4.0 * torch.ones(10, dtype=torch.float64)
    diff = float(spectral.casimir_energy_density(constrained, free, s=0.1))
    assert diff > 0


def test_hutchinson_trace_accuracy():
    H = make_spd(dim=40)
    tr = float(torch.trace(H))
    gen = torch.Generator().manual_seed(1)
    est = float(spectral.hutchinson_trace(lambda v: H @ v, dim=40, n_probes=400,
                                          dtype=torch.float64, generator=gen))
    assert est == pytest.approx(tr, rel=0.1)


def test_lanczos_extremal_eigenvalues():
    H = make_spd(dim=50)
    evals = torch.linalg.eigvalsh(H)
    gen = torch.Generator().manual_seed(2)
    a, b, k = spectral.lanczos_tridiag(lambda v: H @ v, dim=50, m=30,
                                       dtype=torch.float64, generator=gen)
    ritz, _ = spectral._tridiag_eig(a, b)
    assert float(ritz.max()) == pytest.approx(float(evals.max()), rel=1e-6)


def test_slq_trace_of_identity_function():
    """Tr f(H) with f = identity must reproduce Tr H."""
    H = make_spd(dim=40)
    gen = torch.Generator().manual_seed(3)
    est = float(spectral.slq_spectral_sum(lambda v: H @ v, dim=40,
                                          f=lambda lam: lam, n_probes=100, m=25,
                                          dtype=torch.float64, generator=gen))
    assert est == pytest.approx(float(torch.trace(H)), rel=0.1)


def test_zero_point_energy_matches_exact_spectrum():
    H = make_spd(dim=30)
    s = 0.1
    evals = torch.linalg.eigvalsh(H)
    exact = float(spectral.heat_kernel_zpe(evals, s))
    gen = torch.Generator().manual_seed(4)
    est = float(spectral.zero_point_energy(lambda v: H @ v, dim=30, s=s,
                                           n_probes=100, m=25,
                                           dtype=torch.float64, generator=gen))
    assert est == pytest.approx(exact, rel=0.15)


def test_zero_point_energy_flat_below_sharp():
    """The whole point: a flat minimum must carry less vacuum energy."""
    sharp = 100.0 * torch.eye(10, dtype=torch.float64)
    flat = 1.0 * torch.eye(10, dtype=torch.float64)
    gen = torch.Generator().manual_seed(5)
    e_sharp = float(spectral.zero_point_energy(lambda v: sharp @ v, 10, s=0.05,
                                               n_probes=20, m=10,
                                               dtype=torch.float64, generator=gen))
    e_flat = float(spectral.zero_point_energy(lambda v: flat @ v, 10, s=0.05,
                                              n_probes=20, m=10,
                                              dtype=torch.float64, generator=gen))
    assert e_flat < e_sharp
