"""Extended integration, edge-case, and physics-consistency tests.

These complement the per-module unit tests with checks that cut across
modules: conservation laws of the swarm forces, dtype/device plumbing,
reproducibility, optimizer state handling, and odd-but-legal configurations.
"""

import math

import pytest
import torch

from fluctuation_opt import (
    ZeroPointOptimizer,
    LifshitzSwarm,
    GradientPressureBalancer,
    MLP,
    QuantumAnnealingSchedule,
    partial_derivative,
)
from fluctuation_opt.core import lifshitz, matsubara, spectral


# ---------------------------------------------------------------------------
# Lifshitz physics consistency
# ---------------------------------------------------------------------------

def test_pairwise_forces_newtons_third_law():
    """Total internal force on the swarm must vanish (action = reaction)."""
    torch.manual_seed(0)
    X = torch.randn(12, 4, dtype=torch.float64)
    r = torch.rand(12, dtype=torch.float64) * 0.9
    F = lifshitz.pairwise_lifshitz_forces(X, r)
    assert float(F.sum(dim=0).abs().max()) < 1e-10


def test_polylog_monotone_in_z():
    z = torch.linspace(0.0, 1.0, 50)
    v = lifshitz.polylog(2.0, z, n_terms=256)
    assert bool((v[1:] >= v[:-1]).all())
    assert float(v[0]) == pytest.approx(0.0, abs=1e-12)


def test_polylog_zeta3_at_one():
    # Li_3(1) = zeta(3) = 1.2020569...
    v = float(lifshitz.polylog(3.0, torch.tensor(1.0), n_terms=4000))
    assert v == pytest.approx(1.2020569, rel=1e-4)


def test_free_energy_weaker_at_higher_temperature_short_range():
    """More thermal smearing cannot deepen the short-distance binding well
    faster than T itself: check the well remains attractive at all T and
    that the T->0 Matsubara sum approaches the zero-T integral."""
    z = torch.tensor(0.9, dtype=torch.float64)
    d = torch.tensor(0.5, dtype=torch.float64)
    e0 = float(lifshitz.lifshitz_energy(z, d, a0=0.0, n_terms=512))
    eT = float(lifshitz.lifshitz_free_energy(z, d, T=0.02, n_max=2048, a0=0.0))
    assert eT < 0
    assert eT == pytest.approx(e0, rel=2e-2)


def test_reflectivity_rank_handles_ties_and_single_element():
    f = torch.tensor([1.0, 1.0, 1.0])
    r = lifshitz.reflectivity_from_rank(f)
    assert r.shape == (3,)
    assert bool(((r >= 0) & (r <= 0.98)).all())
    r1 = lifshitz.reflectivity_from_rank(torch.tensor([2.0]))
    assert r1.shape == (1,)
    assert torch.isfinite(r1).all()


# ---------------------------------------------------------------------------
# Matsubara / annealing schedule
# ---------------------------------------------------------------------------

def test_sigma_limits_to_zero_point_sigma():
    sched = QuantumAnnealingSchedule(T0=2.0, tau=10.0, omega=4.0, scale=0.5)
    late = sched.sigma(10**9)
    assert late == pytest.approx(sched.zero_point_sigma(), rel=1e-3)


def test_thermal_variance_matches_exact_coth_on_both_branches():
    """The coth implementation switches branches at x = 1e-4; both branches
    must agree with the exact analytic coth."""
    T = 1.0
    for x in (0.5e-4, 2e-4, 0.1, 5.0):  # below/above the branch point
        omega = 2 * T * x
        v = float(matsubara.thermal_variance(torch.tensor(omega), T))
        exact = (1.0 / math.tanh(x)) / (2.0 * omega)
        assert v == pytest.approx(exact, rel=1e-3)


def test_update_omega_ignores_bad_estimates():
    sched = QuantumAnnealingSchedule(omega=2.0)
    sched.update_omega(float("nan"))
    sched.update_omega(-5.0)
    sched.update_omega(0.0)
    assert sched.omega == 2.0


# ---------------------------------------------------------------------------
# Spectral estimators
# ---------------------------------------------------------------------------

def test_hutchinson_reproducible_with_generator():
    A = torch.diag(torch.arange(1.0, 33.0))
    hvp = lambda v: A @ v  # noqa: E731
    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(7)
        outs.append(float(spectral.hutchinson_trace(hvp, 32, n_probes=8,
                                                    generator=g)))
    assert outs[0] == outs[1]


def test_slq_heat_kernel_zpe_against_dense_spectrum():
    torch.manual_seed(1)
    Q, _ = torch.linalg.qr(torch.randn(40, 40))
    lam = torch.linspace(0.1, 25.0, 40)
    A = Q @ torch.diag(lam) @ Q.T
    hvp = lambda v: A @ v  # noqa: E731
    exact = float(spectral.heat_kernel_zpe(lam, s=0.2))
    g = torch.Generator().manual_seed(3)
    est = float(spectral.zero_point_energy(hvp, 40, s=0.2, n_probes=20, m=30,
                                           generator=g))
    assert est == pytest.approx(exact, rel=0.15)


def test_lanczos_handles_rank_deficient_operator():
    """Early breakdown (Krylov space smaller than m) must not crash."""
    P = torch.zeros(10, 10)
    P[0, 0] = 3.0  # rank-1
    hvp = lambda v: P @ v  # noqa: E731
    a, b, k = spectral.lanczos_tridiag(hvp, 10, m=8)
    assert k <= 3
    theta, _ = torch.linalg.eigh(torch.diag(a) if b.numel() == 0 else
                                 torch.diag(a) + torch.diag(b, 1) + torch.diag(b, -1))
    assert float(theta.max()) == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# LifshitzSwarm integration
# ---------------------------------------------------------------------------

def test_swarm_float32_dtype_runs():
    swarm = LifshitzSwarm([(-3, 3)] * 3, n_particles=12, dtype=torch.float32,
                         seed=0)
    res = swarm.minimize(lambda X: (X**2).sum(-1), max_iter=30)
    assert res["fun"] < 3.0
    assert res["x"].dtype == torch.float32


def test_swarm_one_dimensional_problem():
    swarm = LifshitzSwarm([(-10.0, 10.0)], n_particles=16, seed=1)
    res = swarm.minimize(lambda X: ((X - 4.0) ** 2).sum(-1), max_iter=60)
    assert float(res["x"][0]) == pytest.approx(4.0, abs=0.2)


def test_swarm_callback_contract():
    seen = []
    swarm = LifshitzSwarm([(-1, 1)] * 2, n_particles=8, seed=2)
    swarm.minimize(lambda X: (X**2).sum(-1), max_iter=15,
                   callback=lambda info: seen.append(info))
    assert len(seen) == 15
    for k in ("iter", "best_f", "sigma", "X", "fitness"):
        assert k in seen[0]
    assert seen[-1]["best_f"] <= seen[0]["best_f"]


def test_swarm_eval_budget_accounting():
    n, it = 10, 12
    swarm = LifshitzSwarm([(-1, 1)] * 2, n_particles=n, seed=3)
    res = swarm.minimize(lambda X: (X**2).sum(-1), max_iter=it)
    assert res["n_evals"] == n * (it + 1)


def test_swarm_no_anchor_no_quench():
    swarm = LifshitzSwarm([(-2, 2)] * 2, n_particles=12, anchor_best=False,
                         quench_frac=0.0, seed=4)
    res = swarm.minimize(lambda X: (X**2).sum(-1), max_iter=50)
    assert res["fun"] < 1.0


def test_swarm_boltzmann_reflectivity_mode():
    swarm = LifshitzSwarm([(-2, 2)] * 2, n_particles=12,
                         reflectivity="boltzmann", seed=5)
    res = swarm.minimize(lambda X: (X**2).sum(-1), max_iter=50)
    assert res["fun"] < 1.0


def test_swarm_unknown_reflectivity_raises():
    swarm = LifshitzSwarm([(-1, 1)], reflectivity="nope", seed=0)
    with pytest.raises(ValueError):
        swarm.minimize(lambda X: (X**2).sum(-1), max_iter=2)


def test_swarm_asymmetric_bounds_respected():
    lo, hi = 2.0, 9.0
    swarm = LifshitzSwarm([(lo, hi)] * 3, n_particles=10, seed=6)
    box = []
    swarm.minimize(lambda X: (X**2).sum(-1), max_iter=25,
                   callback=lambda info: box.append(info["X"]))
    allX = torch.cat(box)
    assert float(allX.min()) >= lo - 1e-9
    assert float(allX.max()) <= hi + 1e-9
    # optimum of x^2 inside [2,9] is at the lower boundary
    res = swarm.minimize(lambda X: (X**2).sum(-1), max_iter=80)
    assert float(res["x"].max()) < 3.5


# ---------------------------------------------------------------------------
# ZeroPointOptimizer integration
# ---------------------------------------------------------------------------

def _quadratic_problem(seed=0):
    torch.manual_seed(seed)
    theta = torch.nn.Parameter(torch.randn(6))
    target = torch.arange(6.0)

    def closure():
        return ((theta - target) ** 2).sum()

    return theta, closure


def test_optimizer_multiple_param_groups():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(3))
    b = torch.nn.Parameter(torch.randn(2))
    opt = ZeroPointOptimizer([{"params": [a], "lr": 5e-2},
                            {"params": [b], "lr": 2e-2}], sigma=1e-3, seed=0)
    for _ in range(600):
        loss = opt.step(lambda: (a**2).sum() + ((b - 1.0) ** 2).sum())
    assert float(loss) < 0.2


def test_optimizer_weight_decay_shrinks_solution():
    theta, closure = _quadratic_problem()
    opt = ZeroPointOptimizer([theta], lr=5e-2, sigma=1e-3, weight_decay=1.0,
                           seed=0)
    for _ in range(400):
        opt.step(closure)
    target = torch.arange(6.0)
    # with L2 pull toward 0, the solution must sit strictly inside the target
    assert float(theta.detach().norm()) < float(target.norm())


def test_optimizer_odd_probe_count():
    theta, closure = _quadratic_problem()
    opt = ZeroPointOptimizer([theta], lr=5e-2, sigma=1e-3, n_probes=3, seed=0)
    for _ in range(500):
        loss = opt.step(closure)
    assert float(loss) < 0.5


def test_optimizer_state_dict_roundtrip():
    theta, closure = _quadratic_problem()
    opt = ZeroPointOptimizer([theta], lr=1e-2, sigma=1e-3, seed=0)
    for _ in range(5):
        opt.step(closure)
    sd = opt.state_dict()
    opt2 = ZeroPointOptimizer([theta], lr=1e-2, sigma=1e-3, seed=0)
    opt2.load_state_dict(sd)
    st = list(opt2.state.values())[0]
    assert st["step"] == 5
    assert torch.isfinite(st["m"]).all() and torch.isfinite(st["v"]).all()


def test_optimizer_frozen_params_untouched():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(3))
    frozen = torch.nn.Parameter(torch.randn(3), requires_grad=False)
    before = frozen.detach().clone()
    opt = ZeroPointOptimizer([a, frozen], lr=5e-2, sigma=1e-3, seed=0)
    for _ in range(20):
        opt.step(lambda: (a**2).sum() + 0.0 * frozen.sum())
    assert torch.equal(frozen.detach(), before)


def test_optimizer_restores_params_after_probing():
    """The smoothed mode perturbs parameters in place; after step() the
    update must come from the Adam rule, not a leftover perturbation.
    With lr=0 parameters must be exactly unchanged."""
    theta, closure = _quadratic_problem()
    before = theta.detach().clone()
    opt = ZeroPointOptimizer([theta], lr=0.0, sigma=0.5, n_probes=4, seed=0)
    opt.step(closure)
    assert torch.allclose(theta.detach(), before, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# PINN utilities integration
# ---------------------------------------------------------------------------

def test_balancer_weight_clamps_respected():
    p = torch.nn.Parameter(torch.randn(4))
    bal = GradientPressureBalancer([p], n_terms=2, w_min=0.5, w_max=2.0,
                                  update_every=1)
    # extremely mismatched pressures
    l1 = 1e6 * (p**2).sum()
    l2 = 1e-6 * (p**2).sum()
    bal([l1, l2])
    ratio = float(bal.weights.max() / bal.weights.min())
    assert ratio <= 2.0 / 0.5 + 1e-6


def test_balancer_update_every_caches_weights():
    p = torch.nn.Parameter(torch.randn(4))
    bal = GradientPressureBalancer([p], n_terms=2, update_every=10)
    bal([(p**2).sum(), 100 * (p**2).sum()])
    w_after_first = bal.weights.clone()
    bal([(p**2).sum(), 1e6 * (p**2).sum()])  # would rebalance if measured
    assert torch.equal(bal.weights, w_after_first)


def test_partial_derivative_third_order():
    x = torch.linspace(-1, 1, 32).unsqueeze(1).requires_grad_(True)
    u = x**4
    d3 = partial_derivative(u, x, order=3)
    assert torch.allclose(d3, 24 * x, atol=1e-5)


def test_mlp_bias_zero_init_and_forward_double():
    net = MLP([2, 8, 1]).double()
    for m in net.net:
        if isinstance(m, torch.nn.Linear):
            assert float(m.bias.abs().sum()) == 0.0
    y = net(torch.randn(5, 2, dtype=torch.float64))
    assert y.dtype == torch.float64 and y.shape == (5, 1)


def test_end_to_end_tiny_pinn_smoke():
    """One coupled run of MLP + balancer + ZeroPointOptimizer on a 1D ODE:
    u' = u, u(0) = 1 on [0, 0.5]; just verify the machinery trains."""
    torch.manual_seed(0)
    net = MLP([1, 16, 1])
    bal = GradientPressureBalancer(net.parameters(), n_terms=2, update_every=5)
    opt = ZeroPointOptimizer(net.parameters(), lr=2e-2, sigma=1e-3, seed=0)
    x = torch.linspace(0, 0.5, 32).unsqueeze(1)

    def closure():
        xr = x.clone().requires_grad_(True)
        u = net(xr)
        du = partial_derivative(u, xr)
        residual = ((du - u) ** 2).mean()
        ic = (net(torch.zeros(1, 1)) - 1.0).pow(2).mean()
        return bal([residual, ic])

    first = float(opt.step(closure))
    for _ in range(300):
        last = float(opt.step(closure))
    assert last < first
    xt = torch.linspace(0, 0.5, 20).unsqueeze(1)
    rel = float(((net(xt) - torch.exp(xt)).norm() / torch.exp(xt).norm()))
    assert rel < 0.15


def test_pde_benchmark_pressure_configs_are_labeled_correctly():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "pde_benchmark.py"
    spec = importlib.util.spec_from_file_location("pde_benchmark", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    assert not mod.uses_pressure_balancer("adam")
    assert mod.uses_pressure_balancer("adam+balance")
    assert mod.uses_pressure_balancer("zero_point+pressure")
