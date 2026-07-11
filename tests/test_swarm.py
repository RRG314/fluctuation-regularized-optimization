import math

import pytest
import torch

from fluctuation_opt import LifshitzSwarm


def sphere(X):
    return (X**2).sum(-1)


def rastrigin(X):
    return 10 * X.shape[-1] + (X**2 - 10 * torch.cos(2 * math.pi * X)).sum(-1)


def test_solves_sphere_5d():
    sw = LifshitzSwarm([(-5, 5)] * 5, n_particles=30, seed=0)
    res = sw.minimize(sphere, max_iter=200)
    assert res["fun"] < 1e-2


def test_reasonable_on_rastrigin_5d():
    sw = LifshitzSwarm([(-5.12, 5.12)] * 5, n_particles=40, seed=0)
    res = sw.minimize(rastrigin, max_iter=300)
    assert res["fun"] < 10.0  # multimodal; near-global basin


def test_respects_bounds():
    lo, hi = 2.0, 3.0
    visited = []
    sw = LifshitzSwarm([(lo, hi)] * 3, n_particles=15, seed=1)
    sw.minimize(sphere, max_iter=50,
                callback=lambda info: visited.append(info["X"].clone()))
    allX = torch.cat(visited)
    assert float(allX.min()) >= lo - 1e-9
    assert float(allX.max()) <= hi + 1e-9


def test_deterministic_with_seed():
    r1 = LifshitzSwarm([(-5, 5)] * 4, n_particles=20, seed=42).minimize(sphere, 60)
    r2 = LifshitzSwarm([(-5, 5)] * 4, n_particles=20, seed=42).minimize(sphere, 60)
    assert r1["fun"] == r2["fun"]
    assert torch.allclose(r1["x"], r2["x"])


def test_history_monotone_nonincreasing():
    res = LifshitzSwarm([(-5, 5)] * 3, n_particles=20, seed=3).minimize(sphere, 80)
    h = res["history"]
    assert all(a >= b for a, b in zip(h, h[1:]))


def test_scalar_function_wrapper():
    sw = LifshitzSwarm([(-2, 2)] * 2, n_particles=15, seed=0)
    res = sw.minimize(lambda x: float((x**2).sum()), max_iter=60, vectorized=False)
    assert res["fun"] < 0.1


def test_shifted_optimum():
    """Optimum away from the domain center must still be found."""
    target = torch.tensor([3.0, -2.0, 1.5])
    sw = LifshitzSwarm([(-5, 5)] * 3, n_particles=30, seed=0)
    res = sw.minimize(lambda X: ((X - target)**2).sum(-1), max_iter=200)
    assert res["fun"] < 1e-2


def test_flat_vs_sharp_blackbox():
    """Zero-point noise floor + Lifshitz clustering should prefer the flat
    basin when depths are equal (the sharp well is 'hard to stay inside'
    under residual fluctuations)."""

    def landscape(X):
        x = X[..., 0]
        sharp = -1.0 * torch.exp(-((x + 2.0) / 0.05) ** 2)
        flat = -1.0 * torch.exp(-((x - 2.0) / 1.5) ** 2)
        return sharp + flat + 0.01 * x**2

    hits_flat = 0
    for seed in range(5):
        sw = LifshitzSwarm([(-5, 5)], n_particles=30, seed=seed)
        res = sw.minimize(landscape, max_iter=150)
        if res["x"][0] > 0:
            hits_flat += 1
    assert hits_flat >= 4  # strongly prefers the flat basin


def test_bad_bounds_raise():
    with pytest.raises(ValueError):
        LifshitzSwarm([(1.0, 1.0)])
    with pytest.raises(ValueError):
        LifshitzSwarm([(2.0, 1.0)])
