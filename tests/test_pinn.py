import math

import pytest
import torch

from casimir_opt import CasimirPressureBalancer, MLP, partial_derivative


def test_mlp_shapes_and_device_agnostic():
    net = MLP([2, 32, 32, 1])
    x = torch.randn(10, 2)
    assert net(x).shape == (10, 1)


def test_partial_derivative_polynomial():
    x = torch.linspace(-1, 1, 20).reshape(-1, 1).requires_grad_(True)
    u = x**3
    du = partial_derivative(u, x, order=1)
    d2u = partial_derivative(u, x, order=2)
    assert torch.allclose(du, 3 * x**2, atol=1e-5)
    assert torch.allclose(d2u, 6 * x, atol=1e-5)


def test_partial_derivative_sin():
    x = torch.linspace(0, 1, 30).reshape(-1, 1).requires_grad_(True)
    u = torch.sin(math.pi * x)
    d2 = partial_derivative(u, x, order=2)
    assert torch.allclose(d2, -math.pi**2 * torch.sin(math.pi * x), atol=1e-4)


def _two_term_setup(scale=100.0):
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

    def losses():
        return [scale * p[0]**2, p[1]**2]  # wildly imbalanced pressures

    return p, losses


def test_balancer_equalizes_pressures():
    p, losses = _two_term_setup(scale=100.0)
    bal = CasimirPressureBalancer([p], n_terms=2, ema=0.0, update_every=1)
    bal(losses())
    w = bal.weights
    # strong term must be down-weighted, weak term up-weighted
    assert w[0] < w[1]
    # weighted pressures now match: w_k * P_k equal across terms
    P = torch.tensor([200.0, 2.0])
    assert float(w[0] * P[0]) == pytest.approx(float(w[1] * P[1]), rel=1e-4)


def test_balancer_weights_normalized():
    p, losses = _two_term_setup()
    bal = CasimirPressureBalancer([p], n_terms=2, ema=0.0, update_every=1)
    bal(losses())
    assert float(bal.weights.sum()) == pytest.approx(2.0, rel=1e-5)
    assert (bal.weights > 0).all()


def test_balancer_total_is_differentiable():
    p, losses = _two_term_setup()
    bal = CasimirPressureBalancer([p], n_terms=2, update_every=1)
    total = bal(losses())
    total.backward()
    assert p.grad is not None


def test_balancer_spectral_mode_runs():
    p, losses = _two_term_setup()
    bal = CasimirPressureBalancer([p], n_terms=2, update_every=1, mode="spectral")
    total = bal(losses())
    assert torch.isfinite(total)


def test_balancer_wrong_term_count_raises():
    p, losses = _two_term_setup()
    bal = CasimirPressureBalancer([p], n_terms=3)
    with pytest.raises(ValueError):
        bal(losses())


def test_balancer_improves_toy_pinn_conditioning():
    """With one dominating term, plain sum ignores the weak term; the
    balancer must reduce the weak term's loss much further."""

    def run(balanced):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        bal = CasimirPressureBalancer([p], n_terms=2, update_every=1)
        opt = torch.optim.SGD([p], lr=1e-3)
        for _ in range(200):
            opt.zero_grad()
            terms = [1000.0 * p[0]**2, p[1]**2]
            total = bal(terms) if balanced else sum(terms)
            total.backward()
            opt.step()
        return float(p.detach()[1]**2)

    assert run(True) < run(False)
