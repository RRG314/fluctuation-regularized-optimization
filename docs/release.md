# Release Checklist

Use this checklist before creating a GitHub release.

## Before release

- [ ] Confirm version in `pyproject.toml`.
- [ ] Confirm version in `fluctuation_opt/__init__.py`.
- [ ] Update `CHANGELOG.md`.
- [ ] Run `pytest`.
- [ ] Run `python examples/quickstart.py`.
- [ ] Run `python -m build`.
- [ ] Confirm the wheel contains `fluctuation_opt`, not stale build output.
- [ ] Rebuild `paper/paper.pdf` from `paper/paper.tex` if attaching a PDF.
- [ ] Check that README claims match current benchmark results.

## Suggested release title

`v0.1.0 - Initial fluctuation-regularized optimization research release`

## Suggested release summary

This initial release contains research code for three fluctuation-regularized
optimization mechanisms: Lifshitz-style swarm search, zero-point-smoothed
PyTorch optimization, and gradient-pressure balancing for PINNs. It includes
tests, examples, benchmark scripts, saved benchmark outputs, mechanism docs,
and manuscript source.

## Known caveats for release notes

- Not published on PyPI.
- The compiled paper PDF must be regenerated from current source before
  attaching.
- Smooth black-box functions still favor tuned PSO/DE for final precision.
- The real Casimir data fit pins the separation offset at the configured lower
  bound.
