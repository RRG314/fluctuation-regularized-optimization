import math

import pytest
import torch

from casimir_opt.core import lifshitz


def test_polylog_at_one_is_zeta2():
    v = float(lifshitz.polylog(2.0, torch.tensor(1.0), n_terms=5000))
    assert v == pytest.approx(math.pi**2 / 6, rel=1e-3)


def test_polylog_small_z_linear():
    z = torch.tensor(1e-4)
    assert float(lifshitz.polylog(2.0, z)) == pytest.approx(1e-4, rel=1e-3)


def test_energy_is_negative_and_force_attractive():
    z = torch.tensor(0.5)
    d = torch.tensor(1.0)
    assert float(lifshitz.lifshitz_energy(z, d)) < 0
    assert float(lifshitz.lifshitz_force(z, d)) > 0


def test_force_consistent_with_energy_derivative():
    """|F| = +dE/dd (attractive: energy rises with separation), checked by
    central differences."""
    z = torch.tensor(0.8, dtype=torch.float64)
    d0, h = 1.3, 1e-6
    ep = float(lifshitz.lifshitz_energy(z, torch.tensor(d0 + h, dtype=torch.float64)))
    em = float(lifshitz.lifshitz_energy(z, torch.tensor(d0 - h, dtype=torch.float64)))
    dE = (ep - em) / (2 * h)
    F = float(lifshitz.lifshitz_force(z, torch.tensor(d0, dtype=torch.float64)))
    assert F == pytest.approx(dE, rel=1e-4)


def test_force_monotone_in_reflectivity_product():
    d = torch.tensor(1.0)
    f1 = float(lifshitz.lifshitz_force(torch.tensor(0.2), d))
    f2 = float(lifshitz.lifshitz_force(torch.tensor(0.9), d))
    assert f2 > f1


def test_force_short_ranged_inverse_square():
    z = torch.tensor(0.9)
    f_near = float(lifshitz.lifshitz_force(z, torch.tensor(0.5), a0=0.0))
    f_far = float(lifshitz.lifshitz_force(z, torch.tensor(5.0), a0=0.0))
    assert f_near / f_far == pytest.approx(100.0, rel=1e-6)  # (5/0.5)^2


def test_finite_T_free_energy_reduces_toward_integral():
    """The Matsubara sum at low T approximates the T=0 integral."""
    z = torch.tensor(0.7)
    d = torch.tensor(1.0)
    e0 = float(lifshitz.lifshitz_energy(z, d, a0=0.0))
    eT = float(lifshitz.lifshitz_free_energy(z, d, T=0.01, n_max=20000, a0=0.0))
    assert eT == pytest.approx(e0, rel=0.05)


def test_reflectivity_rank_ordering():
    fitness = torch.tensor([3.0, 1.0, 2.0])  # best is index 1
    r = lifshitz.reflectivity_from_rank(fitness, r_max=0.98)
    assert r.argmax() == 1
    assert r.argmin() == 0
    assert float(r.max()) == pytest.approx(0.98)
    assert (r >= 0).all() and (r < 1).all()


def test_reflectivity_boltzmann_scale_free():
    f1 = torch.tensor([0.0, 1.0, 2.0])
    f2 = 1000.0 * f1
    r1 = lifshitz.reflectivity_from_boltzmann(f1)
    r2 = lifshitz.reflectivity_from_boltzmann(f2)
    assert torch.allclose(r1, r2, rtol=1e-5)


def test_pairwise_forces_symmetry_and_direction():
    """Two mirrors must attract each other with equal and opposite force."""
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    r = torch.tensor([0.9, 0.9], dtype=torch.float64)
    F = lifshitz.pairwise_lifshitz_forces(X, r)
    assert torch.allclose(F[0], -F[1], atol=1e-12)
    assert F[0, 0] > 0  # particle 0 pulled toward particle 1
    assert F[1, 0] < 0


def test_transparent_mirror_feels_nothing():
    X = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    r = torch.tensor([0.0, 0.9], dtype=torch.float64)
    F = lifshitz.pairwise_lifshitz_forces(X, r)
    assert float(F.abs().max()) == pytest.approx(0.0, abs=1e-12)
