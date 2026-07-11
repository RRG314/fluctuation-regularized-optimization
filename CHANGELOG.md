# Changelog

All notable changes to this repository will be documented here.

The project follows a practical research-code versioning policy:

- patch changes: bug fixes, docs, tests, small reproducibility improvements;
- minor changes: new validated mechanisms, benchmark suites, or public API;
- major changes: incompatible API or benchmark protocol changes.

## [0.1.0] - 2026-07-11

Initial public research repository.

### Added

- `LifshitzSwarm` derivative-free optimizer.
- `ZeroPointOptimizer` PyTorch optimizer.
- `GradientPressureBalancer` for multi-term PINN losses.
- Lifshitz, Matsubara, and spectral helper modules.
- Unit and integration test suite.
- Benchmark scripts and saved results for black-box optimization, PINNs,
  real PDEs, real ML datasets, and a real Casimir data fit.
- Mechanism documentation in `docs/mechanisms.md`.
- Source manuscript in `paper/paper.tex`.

### Changed

- Tuned `LifshitzSwarm` local refinement to use a partial final quench: the
  worst particles become local probes while the rest continue exploring.
- Regenerated benchmark tables and figures after the optimizer/benchmark
  changes.

### Fixed

- Corrected `pde_benchmark.py` so `zero_point+pressure` actually uses
  `GradientPressureBalancer`.

### Known limitations

- The package is not published on PyPI.
- A compiled paper PDF is not currently tracked; rebuild it from
  `paper/paper.tex` before attaching one to a release.
- Smooth benchmark functions still favor tuned PSO or differential evolution
  for final precision.
- The real Casimir data fit pins the separation offset at the configured lower
  bound, which should be treated as a caveat.
