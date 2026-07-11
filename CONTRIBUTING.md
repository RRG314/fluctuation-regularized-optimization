# Contributing

Thank you for considering a contribution. This project is research code, so
the standard for changes is slightly different from a normal utility library:
claims need evidence, results need reproducibility, and new mechanisms should
be separated from validated behavior.

## Project scope

This repository is about fluctuation-regularized optimization mechanisms:

- Lifshitz-style derivative-free swarm search;
- zero-point-smoothed PyTorch optimization;
- gradient-pressure balancing for PINNs and multi-term losses;
- tests, benchmarks, and documentation that explain when these mechanisms
  help and when they do not.

Please do not add unrelated research ideas, unrelated optimizer families, or
speculative claims without tests and benchmark evidence.

## Development setup

Use Python 3.11 when possible.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Run the core checks:

```bash
pytest
python examples/quickstart.py
python -m build --wheel
```

## Contribution workflow

1. Open an issue before large changes.
2. Create a focused branch from `main`.
3. Keep code, tests, docs, and benchmark claims aligned.
4. Run the relevant tests before opening a pull request.
5. Use the pull request template and explain what evidence changed.

Small documentation fixes can go straight to a pull request.

## Evidence standard

For algorithmic changes, include at least one of:

- a unit or integration test that captures the intended mechanism;
- a benchmark result showing the change improves or preserves behavior;
- a clear explanation of why a previous result should change.

For benchmark result changes, include:

- command used;
- hardware/software context when relevant;
- seed count;
- raw output or regenerated files;
- honest notes on regressions or failures.

Do not delete negative results just because they are inconvenient. The goal is
to make the repository useful and credible, not to make every table look good.

## Code style

- Prefer simple, explicit PyTorch and NumPy code.
- Keep public API names mechanism-based: `LifshitzSwarm`,
  `ZeroPointOptimizer`, and `GradientPressureBalancer`.
- Keep Casimir references tied to actual Casimir/Lifshitz physics or the real
  Casimir data fit.
- Add comments only where they clarify a non-obvious mechanism or numerical
  choice.

## Tests

The test suite covers:

- physics-inspired mathematical identities;
- optimizer behavior;
- dtype/device plumbing;
- state serialization;
- PINN utility behavior;
- small end-to-end smoke cases.

If a change touches shared optimizer behavior, add or update tests in
`tests/test_integration.py` or the relevant module test file.

## Benchmarks

Benchmark scripts live in `benchmarks/`. Existing generated outputs live in
`benchmarks/results/`.

Before changing checked-in benchmark outputs, explain why the results changed.
If a benchmark is expensive, it is acceptable to update docs first and note the
pending full rerun in the pull request.

## Documentation

Keep these files synchronized:

- `README.md`: entry point for users and contributors;
- `docs/mechanisms.md`: math and mechanism explanation;
- `REPORT.md`: current validation summary;
- `paper/paper.tex`: manuscript source;
- `CHANGELOG.md`: release-facing changes.

## Pull request review checklist

Reviewers should ask:

- Does the change fit the project scope?
- Are the names honest and mechanism-based?
- Are tests or benchmark evidence included?
- Are limitations and regressions disclosed?
- Do docs match the code?
