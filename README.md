
![Tests](https://github.com/yaniv-shulman/gclpr/actions/workflows/linting_and_tests.yml/badge.svg?branch=main) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# GCLPR

`gclpr` contains the experiment code, tests, and paper sources for the GC-LPR study:

> Generalized Context-Aware Local Polynomial Regression

GC-LPR extends local polynomial regression to settings where observations carry structured context, such as geospatial neighborhoods, transit networks, flight graphs, or spatiotemporal disease signals. The method uses decomposed context-aware kernels to combine standard feature-space locality with context-derived relationships, so smoothing adapts to both covariate similarity and the surrounding graph or spatial structure. A concise overview is available in the paper [Generalized Local Polynomial Regression with Decomposed Context-Aware Kernels](https://arxiv.org/abs/2604.25237).

This repository is organized around reproducible experiment runners. The maintained entry points live under `src/gclpr/experiments/`, and the generated outputs used in the paper are written to `out/experiment_*`.

## Paper

Paper: [Generalized Local Polynomial Regression with Decomposed Context-Aware Kernels](https://arxiv.org/abs/2604.25237)

Citation:

```bibtex
@misc{shulman2026generalizedlocalpolynomialregression,
      title={Generalized Local Polynomial Regression with Decomposed Context-Aware Kernels},
      author={Yaniv Shulman},
      year={2026},
      eprint={2604.25237},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2604.25237},
}
```

## Scope

This is research code for reproducing the experiments and paper figures. It is not yet packaged as a general-purpose end-user library API.

The maintained experiments are:

1. California housing with geospatial context
2. NYC Airbnb price regression with subway-network context
3. Synthetic airport-delay regression on a flight graph
4. Hungary chickenpox graph-signal forecasting

## Repository Layout

```text
src/gclpr/experiments/
  california_housing_exp1/
  nyc_airbnb_exp2/
  us_airports_exp3/
  hungary_chickenpox_exp4/
tests/experiments_tests/
paper/
data/
out/
```

- `src/gclpr/experiments/`: maintained experiment runners and helper modules
- `tests/experiments_tests/`: pytest coverage for all maintained experiments
- `paper/`: LaTeX source for the accompanying manuscript
- `data/`: local dataset cache directory; ignored by git
- `out/`: generated experiment outputs; ignored by git

## Requirements

- Python 3.12
- Poetry
- A working LaTeX toolchain if you want to build the paper PDF

## Installation

Install the project and development dependencies with Poetry:

```bash
poetry install --with dev
```

You can then either:

- prefix commands with `poetry run`, or
- open a Poetry shell with `poetry shell`

There is also a helper script:

```bash
./configure.sh
```

## Data

This repository does not commit the experiment datasets. The code expects them in `data/`, which is git-ignored.

Current data expectations:

- Experiment 1
  - Uses `sklearn.datasets.fetch_california_housing`
  - Cached automatically in `data/.sklearn_data/`
- Experiment 2
  - Requires `data/air_bnb_nyc_listings/listings.csv`
  - Requires `data/gtfs_subway/nyc_subway_graph.graphml`
  - The GTFS source files used to build the graph can also be stored under `data/gtfs_subway/`
- Experiment 3
  - Requires `data/airports/airports.csv`
  - Requires `data/airports/flights-airport.csv`
- Experiment 4
  - Downloads the public Hungary chickenpox JSON automatically into `data/hungary_chickenpox/chickenpox.json`

## Running the Experiments

The supported way to run the experiments is with the Python module entry points.

### Experiment 1

```bash
poetry run python -m gclpr.experiments.california_housing_exp1.california_housing_exp_1_cv \
  --protocol pilot_tune_then_refit \
  --grid-profile full \
  --search-mode grid \
  --outer-folds 5 \
  --inner-folds 3 \
  --output-prefix california_exp1_main
```

### Experiment 2

```bash
poetry run python -m gclpr.experiments.nyc_airbnb_exp2.nyc_airbnb_exp_2_cv \
  --protocol holdout_cv \
  --holdout-fraction 0.2 \
  --holdout-repeats 5 \
  --inner-folds 4 \
  --grid-profile full \
  --search-mode grid \
  --output-prefix nyc_airbnb_exp2_holdout80_20_rep5
```

### Experiment 3

```bash
poetry run python -m gclpr.experiments.us_airports_exp3.us_airports_exp_3_cv \
  --protocol holdout_cv \
  --holdout-fraction 0.2 \
  --holdout-repeats 5 \
  --inner-folds 4 \
  --grid-profile full \
  --search-mode grid \
  --output-prefix us_airports_exp3_main
```

### Experiment 4

```bash
poetry run python -m gclpr.experiments.hungary_chickenpox_exp4.hungary_chickenpox_exp_4_cv \
  --protocol rolling_origin_cv \
  --outer-splits 5 \
  --inner-folds 4 \
  --lags 4 \
  --grid-profile full \
  --search-mode grid \
  --output-prefix hungary_chickenpox_exp4_main
```

Each runner writes fold-level raw results, aggregated summaries, metadata, and plots into `out/experiment_*`.

## Testing and Quality Checks

Run the full test suite:

```bash
poetry run pytest
```

Run linting and type checking:

```bash
./tests/lint.sh
```

If you want autofixes where possible:

```bash
./tests/lint.sh -f
```

## Building the Paper

From the `paper/` directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error gclpr.tex
```

The generated PDF is written to `paper/gclpr.pdf` during the build process.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This repository is released under the [MIT License](LICENSE).
