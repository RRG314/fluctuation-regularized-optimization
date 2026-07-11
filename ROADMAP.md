# Roadmap

This roadmap is intentionally conservative. The goal is to make the existing
mechanisms clearer, better tested, and more reproducible before adding new
ideas.

## Near term

- Rebuild the paper PDF from the updated `paper/paper.tex`.
- Add API reference documentation for the three public classes.
- Add a lightweight benchmark command that runs quickly in CI.
- Add release artifacts for source distributions and wheels.
- Improve benchmark metadata so generated result files record environment,
  seed count, and command.

## Research hardening

- Increase seed counts for the strongest PINN claims.
- Add confidence intervals where current reports use only medians/IQR.
- Separate quick smoke benchmarks from full CPU benchmark suites.
- Add wall-clock reporting alongside matched evaluation budgets.
- Add ablations for zero-point smoothing vs pressure balancing.

## Not planned without evidence

- Adding unrelated optimizer families.
- Renaming the project back to a Casimir-centered name.
- Claiming universal superiority over Adam, PSO, DE, or L-BFGS.
- Publishing benchmark claims without raw outputs and reproduction commands.
