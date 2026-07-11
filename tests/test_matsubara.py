import math

import pytest
import torch

from casimir_opt.core import matsubara


def test_matsubara_frequencies():
    xi = matsubara.matsubara_frequencies(T=0.5, n_max=3)
    expected = torch.tensor([0.0, math.pi, 2 * math.pi, 3 * math.pi])
    assert torch.allclose(xi, expected)


def test_thermal_variance_zero_point_limit():
    """T -> 0 must give the zero-point variance 1/(2 omega), NOT zero."""
    omega = 2.0
    v = float(matsubara.thermal_variance(omega, T=1e-9))
    assert v == pytest.approx(1.0 / (2 * omega), rel=1e-4)


def test_thermal_variance_classical_limit():
    """High T must give equipartition T / omega^2."""
    omega, T = 1.0, 100.0
    v = float(matsubara.thermal_variance(omega, T))
    assert v == pytest.approx(T / omega**2, rel=1e-2)


def test_thermal_variance_monotone_in_T():
    omega = 1.0
    vs = [float(matsubara.thermal_variance(omega, T)) for T in (0.1, 1.0, 10.0)]
    assert vs[0] < vs[1] < vs[2]


def test_mode_free_energy_zero_T_is_zero_point():
    omega = 3.0
    f = float(matsubara.mode_free_energy(omega, T=0.0))
    assert f == pytest.approx(omega / 2)


def test_schedule_monotone_decreasing_with_positive_floor():
    sch = matsubara.QuantumAnnealingSchedule(T0=5.0, tau=50.0, omega=2.0, scale=1.0)
    sigmas = [sch(t) for t in range(0, 2000, 100)]
    assert all(a >= b for a, b in zip(sigmas, sigmas[1:]))
    floor = sch.zero_point_sigma()
    assert floor > 0
    assert sigmas[-1] >= floor
    # late-time sigma approaches (but never crosses) the zero-point floor
    assert sch(10**7) == pytest.approx(floor, rel=1e-2)


def test_schedule_omega_update_shrinks_floor():
    sch = matsubara.QuantumAnnealingSchedule(T0=1.0, tau=10.0, omega=1.0, scale=1.0)
    f0 = sch.zero_point_sigma()
    for _ in range(50):
        sch.update_omega(25.0)  # stiff region measured
    assert sch.zero_point_sigma() < f0


def test_schedule_rejects_bad_args():
    with pytest.raises(ValueError):
        matsubara.QuantumAnnealingSchedule(T0=-1.0)
    with pytest.raises(ValueError):
        matsubara.QuantumAnnealingSchedule(tau=0.0)
