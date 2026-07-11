"""Minimal source-install smoke example for casimir-opt."""

import torch

from casimir_opt import CasimirOptimizer, CasimirSwarm


def run_swarm() -> None:
    def sphere(x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum(dim=-1)

    swarm = CasimirSwarm([(-5.0, 5.0)] * 3, n_particles=24, seed=0)
    result = swarm.minimize(sphere, max_iter=120)
    print(f"CasimirSwarm sphere: f={result['fun']:.3e}, x={result['x'].tolist()}")


def run_torch_optimizer() -> None:
    theta = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
    opt = CasimirOptimizer([theta], lr=0.05, sigma=0.01, seed=0)

    for _ in range(250):
        opt.step(lambda: (theta**2).sum())

    print(f"CasimirOptimizer quadratic: theta={theta.detach().tolist()}")


if __name__ == "__main__":
    run_swarm()
    run_torch_optimizer()
