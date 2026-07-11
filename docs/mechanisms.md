# Mechanisms and Math

This document explains the mechanisms in `fluctuation-regularized-optimization`.
The repository name is descriptive: the code implements fluctuation-based
regularization mechanisms for optimization. Some mechanisms are motivated by
Casimir/Lifshitz mathematics, but the project is not named or presented as a
"Casimir optimizer."

The algorithms are not a complete physical simulation of real plates, real
electromagnetic fields, or real materials.

The implemented mechanisms are:

| Code | Mechanism | What is physical | What is algorithmic |
| --- | --- | --- | --- |
| `LifshitzSwarm` | Fitness-graded Lifshitz-style attraction | The pair interaction is modeled after a scalar Lifshitz energy between partially reflective mirrors. | Candidate solutions are treated as mirrors; objective values are mapped to reflectivities. |
| `QuantumAnnealingSchedule` | Zero-point floor | A quantum oscillator has nonzero variance as temperature goes to zero. | Exploration noise anneals to a nonzero floor instead of freezing completely. |
| `ZeroPointOptimizer` | Zero-point-smoothed loss | Gaussian smoothing has a second-order Hessian trace term, analogous to a one-loop/zero-point correction. | PyTorch gradients are averaged over small parameter perturbations to bias toward flatter minima. |
| `GradientPressureBalancer` | Radiation-pressure balance | Boundaries in field problems feel pressure; equilibrium means competing pressures balance. | PINN loss terms are reweighted by their gradient norms so one term does not crush the others. |
| `zero_point_energy` diagnostic | Regularized mode sum | Vacuum energy is a regularized sum over mode frequencies. | Hessian eigenvalues are treated as local loss-landscape modes and summarized with SLQ. |

## Notation

- The optimization objective is minimized.
- Candidate positions are `x_i in R^D`.
- The objective value at a candidate is `f_i = f(x_i)`.
- Reflectivity is `r_i in [0, r_max]`, where better candidates receive higher
  reflectivity.
- `theta` denotes trainable neural-network parameters.
- `H(theta)` is the Hessian of a differentiable loss.

## 1. Fitness-graded Lifshitz-style swarm

The swarm first normalizes each candidate into the unit box so all dimensions
have comparable geometry.

A candidate's objective value is converted into a mirror reflectivity. The
default rank map is scale-free:

```text
rank_i = 0 for the best candidate, N - 1 for the worst
rho_i  = (N - 1 - rank_i) / (N - 1)
r_i    = r_max * rho_i^gamma
```

The optional Boltzmann map is:

```text
r_i = r_max * exp(-(f_i - f_min) / T_f)
```

where `T_f` defaults to the interquartile spread of the current fitness
values.

For two partially reflective one-dimensional scalar mirrors separated by
distance `d`, the zero-temperature Lifshitz-style energy is modeled as:

```text
E(d, z) = (1 / 2 pi) * integral_0^inf log(1 - z exp(-2 xi d)) d xi
        = -Li_2(z) / (4 pi d)
```

where:

```text
z = r_i r_j
Li_2(z) = sum_{n>=1} z^n / n^2
```

The optimizer uses a short-distance regulator `a0`, giving:

```text
E_ij = -Li_2(r_i r_j) / (4 pi (d_ij + a0))
|F_ij| = Li_2(r_i r_j) / (4 pi (d_ij + a0)^2)
```

The force direction points from candidate `i` toward candidate `j`:

```text
F_i = sum_{j != i} |F_ij| * (x_j - x_i) / (||x_j - x_i|| + eps)
```

This creates three intended behaviors:

1. Good candidates attract more strongly because they have larger `r`.
2. Poor candidates become nearly transparent and do not dominate the swarm.
3. The interaction is short-ranged, so separated basins can coexist longer
   than in a purely global-best method.

This is an optimization mechanism based on the Lifshitz-form interaction. It is
not a full electromagnetic material model.

## 2. Quantum annealing with a zero-point floor

The noise schedule uses the position variance of a quantum harmonic mode:

```text
Var[x](omega, T) = (1 / (2 omega)) * coth(omega / (2T))
```

The two useful limits are:

```text
T large: Var[x] approximately T / omega^2
T -> 0: Var[x] -> 1 / (2 omega)
```

The temperature decays as:

```text
T(t) = T0 / (1 + t / tau)
```

So the noise amplitude is:

```text
sigma(t) = scale * sqrt(Var[x](omega, T(t)))
```

Unlike classical simulated annealing, the noise does not go to zero. The
nonzero floor is deliberate: it helps avoid brittle sharp basins, but it can
also prevent machine-precision convergence on smooth functions. This tradeoff
is visible in the black-box benchmark results.

## 3. Zero-point-smoothed PyTorch optimizer

For a differentiable loss `L(theta)`, the optimizer uses perturbation-smoothed
gradients. Define:

```text
L_sigma(theta) = E_epsilon[L(theta + sigma epsilon)]
epsilon ~ N(0, I)
```

The Taylor expansion is:

```text
L_sigma(theta)
  = L(theta) + (sigma^2 / 2) Tr H(theta) + higher-order terms
```

The trace term penalizes curvature. This is the algorithmic meaning of the
"zero-point" or "one-loop" language in the code: flatter minima have smaller
local curvature contributions.

The default smoothed mode estimates the gradient using antithetic probes:

```text
g_plus  = grad L(theta + epsilon)
g_minus = grad L(theta - epsilon)
```

For the common low-cost setting with one antithetic pair, the center gradient
is included as a stabilizer:

```text
g_eff = (grad L(theta) + g_plus + g_minus) / 3
```

For larger probe counts, the code keeps the pure antithetic smoothing signal:

```text
g_eff = mean over pairs of (g_plus + g_minus) / 2
```

That distinction is intentional. Large-probe settings are used for exploration;
including a center pull there can trap the optimizer in sharp wells.

The effective gradient is then passed through an Adam-style moment update.
This means `ZeroPointOptimizer` is best understood as Adam with a
zero-point-smoothed gradient estimator, not as a replacement for all adaptive
gradient methods.

## 4. Pressure balancing for PINNs

PINN losses often combine several terms:

```text
L_total = sum_k w_k L_k
```

Examples include PDE residual loss, boundary loss, and initial-condition loss.
If one term has much larger gradients, it can dominate training and prevent the
network from satisfying the other constraints.

The pressure balancer measures:

```text
P_k = ||grad_theta L_k||
```

In spectral mode it uses a curvature-weighted pressure:

```text
P_k = ||grad_theta L_k|| * sqrt(max(v^T H_k v / dim, 0))
```

Weights are set by inverse pressure:

```text
w_k proportional to mean(P) / (P_k + delta)
```

Then the weights are:

1. Smoothed by an exponential moving average.
2. Clamped to `[w_min, w_max]`.
3. Renormalized so `sum_k w_k = number_of_terms`.

Mechanistically, terms pushing too hard get reduced and drowned-out terms get
amplified. This is why the Poisson and Burgers PINN results are the strongest
current evidence in the repository.

## 5. Zero-point-energy diagnostic

Around a trained point, approximate:

```text
L(theta + delta) approximately L(theta) + g^T delta + 1/2 delta^T H delta
```

Positive Hessian eigenvalues are treated as local mode stiffnesses:

```text
omega_i = sqrt(max(lambda_i, 0))
```

The diagnostic reports a heat-kernel-regularized mode sum:

```text
E0(s) = 1/2 sum_i omega_i exp(-s omega_i)
```

For neural networks the Hessian is not formed explicitly. The code estimates
`Tr f(H)` using stochastic Lanczos quadrature and Hessian-vector products.

This value is a flatness diagnostic. It should not be read as a measured
physical vacuum energy of the neural network.

## What this should be used for

The mechanisms are most defensible when the problem has one or more of these
features:

- stiff, multi-term losses;
- rugged or noisy objectives;
- derivative-free fitting where local polishing is expensive;
- cases where perturbation robustness matters more than last-digit precision.

## What this should not be used for

The current algorithms are not the best choice when:

- a smooth convex or nearly convex problem already works with Adam/L-BFGS;
- the goal is machine-precision convergence on classic test functions;
- wall-clock budget is tighter than gradient-evaluation budget;
- the problem has not been checked against standard baselines.

## Main failure modes observed so far

| Failure mode | Where it appears | Why it happens |
| --- | --- | --- |
| Worse final precision than DE/PSO | Smooth black-box functions | The zero-point floor keeps exploration alive instead of collapsing to machine precision. |
| Lower accuracy on easy heat equation | Smooth PINN problem | Plain Adam is already strong; extra smoothing costs accuracy. |
| Worse RMSE on housing | Real ML benchmark | Flatness/robustness tradeoff is not always free. |
| Separation offset pinned at fit bound | Real Casimir data fit | `omega_p` and `Delta z` are partially degenerate under the chosen uncertainty window. |

## Evidence standard

The repository should be judged by the included tests, raw benchmark outputs,
and reproducible scripts, not by the analogy alone. The analogy supplies
mechanisms; the benchmarks decide whether those mechanisms are useful.
