# Contributing

## Scope

This repository is primarily a research-reproducibility codebase for the GC-LPR paper. Contributions should preserve that goal:

- keep experiment protocols explicit
- prefer reproducibility over convenience
- avoid adding secondary execution paths that drift from the maintained runners

Contributions to the paper are also welcome, including corrections, clearer explanations, improvements to figures, and extensions to the theory and experiments.

## Development Setup

1. Install Poetry.
2. Install the project with development dependencies:

```bash
poetry install --with dev
```

3. Run commands either through `poetry run` or inside `poetry shell`.

## Project Conventions

- Maintained experiment code lives under `src/gclpr/experiments/`.
- Tests live under `tests/experiments_tests/`.
- Use type annotations and Google-style docstrings for maintained experiment code.
- Keep the public package name as `gclpr`.

## Data Policy

Datasets are not committed to git. They belong under `data/`, which is intentionally ignored.

When adding a new experiment:

- document the dataset source
- document the expected local file layout
- keep the runner deterministic where practical
- add tests that do not rely on large external downloads

## Before Opening a Change

Run:

```bash
poetry run pytest
./tests/lint.sh
```

If your change affects the manuscript, also rebuild the paper:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error gclpr.tex
```

## Experiment Changes

If you change an experiment protocol, search grid, or dataset preprocessing:

- update the corresponding tests
- update the paper text if the reported results depend on that change
- avoid silent methodological drift

For experiment additions, mirror the existing structure:

- one experiment package under `src/gclpr/experiments/`
- one runner module
- one helper module where needed
- one dedicated pytest module under `tests/experiments_tests/`

## Pull Request Expectations

A good change set should make it clear:

- what changed
- why the change is methodologically or operationally necessary
- how it was validated

For public-facing docs and metadata changes, keep the instructions executable. If the README says a command works, it should actually work on a clean clone once the documented data is in place.
