import pytest
import torch

from casimir_opt import CasimirOptimizer


def test_converges_on_quadratic():
    p = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
    opt = CasimirOptimizer([p], lr=0.05, sigma=0.01, seed=0)
    for _ in range(300):
        opt.step(lambda: (p**2).sum())
    assert float((p.detach()**2).sum()) < 1e-6


def test_trace_mode_converges():
    p = torch.nn.Parameter(torch.tensor([2.0]))
    opt = CasimirOptimizer([p], lr=0.05, mode="trace", zpe_coeff=1e-2, seed=0)
    for _ in range(200):
        opt.step(lambda: (p**2).sum())
    assert abs(float(p.detach())) < 0.05


def test_escapes_sharp_minimum_to_flat():
    """The defining behavior: started INSIDE a sharp basin of equal depth,
    the vacuum-dressed gradient must tunnel out to the flat basin, while
    plain Adam stays trapped."""

    def well(x):
        return (-torch.exp(-((x + 1) / 0.07) ** 2)
                - torch.exp(-((x - 1) / 0.8) ** 2) + 0.02 * x**2)

    # Adam control: trapped
    q = torch.nn.Parameter(torch.tensor([-1.02]))
    adam = torch.optim.Adam([q], lr=0.03)
    for _ in range(800):
        adam.zero_grad()
        well(q).backward()
        adam.step()
    assert float(q.detach()) < -0.9

    # Casimir: escapes (all seeds)
    for seed in range(3):
        q = torch.nn.Parameter(torch.tensor([-1.02]))
        opt = CasimirOptimizer([q], lr=0.03, sigma=1.0, n_probes=8,
                               floor_frac=0.1, tau=150, seed=seed)
        for _ in range(800):
            opt.step(lambda: well(q))
        assert float(q.detach()) > 0.8, f"failed to escape (seed={seed})"


def test_sigma_anneals_but_never_freezes():
    p = torch.nn.Parameter(torch.tensor([0.0]))
    opt = CasimirOptimizer([p], lr=1e-3, sigma=0.5, floor_frac=0.2, tau=10, seed=0)
    s0 = opt.current_sigma()
    for _ in range(200):
        opt.step(lambda: (p**2).sum())
    s_late = opt.current_sigma()
    assert s_late < s0
    assert s_late > 0.5 * 0.2 * 0.9  # above ~the zero-point floor


def test_zero_point_energy_diagnostic_orders_flat_below_sharp():
    ps = torch.nn.Parameter(torch.zeros(5))
    sharp = CasimirOptimizer([ps], seed=0)
    e_sharp = sharp.zero_point_energy(lambda: 50.0 * (ps**2).sum(), s=0.05)

    pf = torch.nn.Parameter(torch.zeros(5))
    flat = CasimirOptimizer([pf], seed=0)
    e_flat = flat.zero_point_energy(lambda: 0.5 * (pf**2).sum(), s=0.05)
    assert e_flat < e_sharp


def test_deterministic_with_seed():
    outs = []
    for _ in range(2):
        torch.manual_seed(7)
        p = torch.nn.Parameter(torch.randn(4))
        opt = CasimirOptimizer([p], lr=0.02, sigma=0.1, seed=123)
        for _ in range(50):
            opt.step(lambda: ((p - 1.0)**2).sum())
        outs.append(p.detach().clone())
    assert torch.equal(outs[0], outs[1])


def test_requires_closure():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = CasimirOptimizer([p])
    with pytest.raises(ValueError):
        opt.step(None)


def test_invalid_mode_raises():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError):
        CasimirOptimizer([p], mode="banana")


def test_trains_small_net():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(),
                              torch.nn.Linear(16, 1))
    X = torch.randn(64, 2)
    y = (X[:, :1] * X[:, 1:]).detach()
    opt = CasimirOptimizer(net.parameters(), lr=5e-3, sigma=5e-3, seed=0)

    def closure():
        return ((net(X) - y)**2).mean()

    l0 = float(closure())
    for _ in range(400):
        opt.step(closure)
    assert float(closure()) < 0.3 * l0
