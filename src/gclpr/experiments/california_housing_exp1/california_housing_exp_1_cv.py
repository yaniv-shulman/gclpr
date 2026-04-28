"""Experiment 1 runner for the California Housing study."""

import argparse
import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rsklpr.kernels import laplacian_normalized_metric, tricube_normalized_metric
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from gclpr.experiments.california_housing_exp1.california_housling_exp_1 import (
    cv_gclpr_geo,
    cv_knn,
    cv_rsklpr,
    haversine_rbf_kernel_factory,
    plot_geospatial_error_maps,
    plot_prediction_scatter_grid,
)

_repo_dir: Path = Path(__file__).resolve().parents[4]
_data_dir = _repo_dir / "data" / ".sklearn_data"
_out_dir = _repo_dir / "out" / "experiment_1"

_model_order: List[str] = [
    "knn",
    "lpr",
    "rsklpr",
    "gclpr_geo",
    "grclpr_geo",
]

_model_labels: Dict[str, str] = {
    "knn": "KNN",
    "lpr": "LPR",
    "rsklpr": "RSKLPR",
    "gclpr_geo": "GC-LPR (Geospatial)",
    "grclpr_geo": "GRC-LPR (Geospatial)",
}

_search_dispatch: Dict[str, Callable[..., Any]] = {
    "knn": cv_knn,
    "lpr": cv_rsklpr,
    "rsklpr": cv_rsklpr,
    "gclpr_geo": cv_gclpr_geo,
    "grclpr_geo": cv_gclpr_geo,
}

_non_geo_cols: List[int] = [0, 1]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for California Housing Experiment 1."""

    protocol: str = "holdout_cv"
    search_mode: str = "grid"
    grid_profile: str = "reduced"
    scoring: str = "strict_rmse"
    outer_folds: int = 5
    inner_folds: int = 4
    pilot_fraction: float = 0.30
    holdout_fraction: float = 0.20
    holdout_repeats: int = 5
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0
    random_search_n_iter: int | None = None
    generate_plots: bool = True
    output_prefix: str = "california_exp1"
    smoke: bool = False
    max_samples: int | None = None


def load_raw_data(
    max_samples: int | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the California Housing dataset for Experiment 1.

    Args:
        max_samples: Optional number of observations to subsample.
        seed: Random seed used when subsampling.

    Returns:
        The feature matrix ``x`` and target vector ``y``.
    """
    housing: Any = fetch_california_housing(as_frame=True, data_home=str(_data_dir))
    frame: pd.DataFrame = housing.frame.copy()

    features: list[str] = ["MedInc", "AveRooms", "Latitude", "Longitude"]
    x: np.ndarray = frame[features].to_numpy(copy=True)
    y: np.ndarray = frame["MedHouseVal"].to_numpy(copy=True)

    if max_samples is not None and max_samples < len(y):
        rng: np.random.Generator = np.random.default_rng(seed=seed)
        indices: np.ndarray = rng.choice(len(y), size=max_samples, replace=False)
        x = x[indices]
        y = y[indices]

    return x, y


def scale_non_geo_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale non-geospatial columns while preserving latitude/longitude.

    Args:
        x_train: Training feature matrix.
        x_test: Test feature matrix.

    Returns:
        The scaled training and test matrices.
    """
    scaler: StandardScaler = StandardScaler()
    x_train_scaled: np.ndarray = x_train.copy()
    x_test_scaled: np.ndarray = x_test.copy()
    x_train_scaled[:, _non_geo_cols] = scaler.fit_transform(x_train[:, _non_geo_cols])
    x_test_scaled[:, _non_geo_cols] = scaler.transform(x_test[:, _non_geo_cols])
    return x_train_scaled, x_test_scaled


def build_geo_kernel_grid(
    scales_km: List[float],
) -> List[List[Callable[..., np.ndarray]]]:
    """
    Build compound geospatial kernels for the search space.

    Args:
        scales_km: Haversine RBF length scales in kilometers.

    Returns:
        A list of kernel lists combining a base kernel and geospatial kernel.
    """
    base_kernels: List[Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]] = [
        laplacian_normalized_metric,
        tricube_normalized_metric,
    ]

    kernel_grid: List[List[Callable[..., np.ndarray]]] = []
    base_kernel: Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]

    for base_kernel in base_kernels:
        scale: float

        for scale in scales_km:
            kernel_grid.append([base_kernel, haversine_rbf_kernel_factory(length_scale_km=scale)])

    return kernel_grid


def build_search_grids(grid_profile: str = "full") -> Dict[str, Any]:
    """
    Build the hyperparameter search spaces for Experiment 1.

    Args:
        grid_profile: One of ``"full"``, ``"reduced"``, or ``"smoke"``.

    Returns:
        A mapping from model identifier to scikit-learn compatible search spaces.

    Raises:
        ValueError: If ``grid_profile`` is unsupported.
    """
    geo_kp_lists: List[List[Callable[..., np.ndarray]]]
    if grid_profile == "smoke":
        geo_kp_lists = build_geo_kernel_grid([10.0])

        return {
            "knn": {
                "n_neighbors": [11],
                "weights": ["distance"],
                "p": [2],
            },
            "lpr": [
                {
                    "size_neighborhood": [11],
                    "degree": [0],
                    "kp": [tricube_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "rsklpr": [
                {
                    "size_neighborhood": [11],
                    "degree": [1],
                    "kp": [laplacian_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["joint"],
                }
            ],
            "gclpr_geo": [
                {
                    "size_neighborhood": [11],
                    "degree": [1],
                    "kp": geo_kp_lists,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "grclpr_geo": [
                {
                    "size_neighborhood": [11],
                    "degree": [1],
                    "kp": geo_kp_lists,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["conden"],
                }
            ],
        }

    if grid_profile == "reduced":
        geo_kp_lists = build_geo_kernel_grid([5.0, 10.0, 15.0, 25.0])

        return {
            "knn": {
                "n_neighbors": [7, 11, 21, 31, 41, 51],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            "lpr": [
                {
                    "size_neighborhood": [57, 67, 77, 87, 97],
                    "degree": [0, 1],
                    "kp": [laplacian_normalized_metric, tricube_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["none"],
                }
            ],
            "rsklpr": [
                {
                    "size_neighborhood": [27, 37, 47, 57],
                    "degree": [1],
                    "kp": [laplacian_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["joint", "conden"],
                }
            ],
            "gclpr_geo": [
                {
                    "size_neighborhood": [107, 117, 127, 137],
                    "degree": [1],
                    "kp": geo_kp_lists,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["none"],
                }
            ],
            "grclpr_geo": [
                {
                    "size_neighborhood": [117, 127, 137, 147],
                    "degree": [1],
                    "kp": geo_kp_lists,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["joint", "conden"],
                }
            ],
        }

    if grid_profile == "full":
        geo_kp_lists = build_geo_kernel_grid([3.0, 5.0, 7.5, 10.0, 15.0, 25.0])

        return {
            "knn": {
                "n_neighbors": list(range(3, 50, 2)),
                "weights": ["uniform", "distance"],
                "p": [1, 2],
                "leaf_size": [15, 30, 60],
            },
            "lpr": [
                {
                    "size_neighborhood": list(range(7, 151, 10)),
                    "degree": [0, 1],
                    "kp": [laplacian_normalized_metric, tricube_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["none"],
                },
                {
                    "size_neighborhood": list(range(7, 151, 10)),
                    "degree": [0, 1],
                    "kp": [laplacian_normalized_metric, tricube_normalized_metric],
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["none"],
                },
            ],
            "rsklpr": [
                {
                    "size_neighborhood": list(range(7, 151, 10)),
                    "degree": [1],
                    "kp": [laplacian_normalized_metric, tricube_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["conden", "joint"],
                },
                {
                    "size_neighborhood": list(range(7, 151, 10)),
                    "degree": [1],
                    "kp": [laplacian_normalized_metric, tricube_normalized_metric],
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["conden", "joint"],
                },
            ],
            "gclpr_geo": [
                {
                    "size_neighborhood": list(range(7, 151, 10)),
                    "degree": [1],
                    "kp": geo_kp_lists,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["none"],
                },
                {
                    "size_neighborhood": list(range(7, 151, 10)),
                    "degree": [1],
                    "kp": geo_kp_lists,
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["none"],
                },
            ],
            "grclpr_geo": [
                {
                    "size_neighborhood": list(range(7, 151, 10)),
                    "degree": [1],
                    "kp": geo_kp_lists,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["conden", "joint"],
                },
                {
                    "size_neighborhood": list(range(7, 151, 10)),
                    "degree": [1],
                    "kp": geo_kp_lists,
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["conden", "joint"],
                },
            ],
        }

    raise ValueError("grid_profile must be one of {'full', 'reduced', 'smoke'}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute the reported regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values. All values must be finite.

    Returns:
        The RMSE, MAE, and R2 metrics.

    Raises:
        ValueError: If ``y_pred`` contains non-finite values.
    """
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("compute_metrics requires all predictions to be finite")

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def safe_clone(estimator: BaseEstimator) -> BaseEstimator:
    """
    Clone an estimator, falling back to ``deepcopy`` when necessary.

    Args:
        estimator: Estimator to duplicate.

    Returns:
        A cloned estimator.
    """
    try:
        return clone(estimator)
    except Exception:
        return copy.deepcopy(estimator)


def strict_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute RMSE, treating any non-finite prediction as an invalid run.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        The RMSE when all predictions are finite, else ``inf``.
    """
    if not np.all(np.isfinite(y_pred)):
        return float("inf")

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def prediction_validity(y_pred: np.ndarray) -> Tuple[bool, float]:
    """Summarize whether a prediction vector is fully finite.

    Args:
        y_pred: Predicted target values.

    Returns:
        A tuple ``(is_valid, non_finite_fraction)``.
    """
    finite_mask: np.ndarray = np.isfinite(y_pred)
    valid: bool = bool(np.all(finite_mask))
    non_finite_fraction: float = float(1.0 - np.mean(finite_mask))
    return valid, non_finite_fraction


def resolve_scoring(scoring: str) -> Any:
    """
    Resolve the configured scoring string into a scorer object.

    Args:
        scoring: Scoring identifier from the experiment configuration.

    Returns:
        A scikit-learn compatible scoring object or scoring string.
    """
    if scoring == "strict_rmse":
        return make_scorer(strict_rmse, greater_is_better=False)

    if scoring in {"neg_root_mean_squared_error", "neg_mean_squared_error"}:
        return scoring

    raise ValueError(
        "Unsupported scoring value. Expected one of "
        "{'strict_rmse', 'neg_root_mean_squared_error', 'neg_mean_squared_error'}."
    )


def run_search(
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    search_grids: Dict[str, Any],
    config: ExperimentConfig,
) -> Any:
    """
    Run hyperparameter search for one model family.

    Args:
        model_key: Model identifier from ``MODEL_ORDER``.
        x_train: Training feature matrix.
        y_train: Training targets.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        The fitted search object.
    """
    search_fn: Callable[..., Any] = _search_dispatch[model_key]

    return search_fn(
        x_train,
        y_train,
        search_grids[model_key],
        cv=config.inner_folds,
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        search_mode=config.search_mode,
        n_iter=config.random_search_n_iter,
        random_state=config.random_state,
        scoring=resolve_scoring(config.scoring),
    )


def pilot_tune_models(
    x: np.ndarray,
    y: np.ndarray,
    search_grids: Dict[str, Any],
    config: ExperimentConfig,
) -> Tuple[Dict[str, BaseEstimator], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Tune one model configuration per family on a pilot subset.

    Args:
        x: Full feature matrix.
        y: Full target vector.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        Best estimators, best parameter mappings, and pilot-tuning audit rows.
    """
    pilot_idx: np.ndarray

    pilot_idx, _ = train_test_split(
        np.arange(len(y)),
        train_size=config.pilot_fraction,
        shuffle=True,
        random_state=config.random_state,
    )

    x_pilot: np.ndarray = x[pilot_idx]
    y_pilot: np.ndarray = y[pilot_idx]
    x_pilot_scaled: np.ndarray
    x_pilot_scaled, _ = scale_non_geo_features(x_train=x_pilot, x_test=x_pilot)

    best_estimators: Dict[str, BaseEstimator] = {}
    selected_params: Dict[str, Dict[str, Any]] = {}
    pilot_rows: List[Dict[str, Any]] = []
    model_key: str

    for model_key in _model_order:
        label: str = _model_labels[model_key]

        search = run_search(
            model_key=model_key, x_train=x_pilot_scaled, y_train=y_pilot, search_grids=search_grids, config=config
        )

        best_estimators[model_key] = search.best_estimator_
        selected_params[model_key] = search.best_params_

        pilot_rows.append(
            {
                "Model": label,
                "Pilot_Samples": len(pilot_idx),
                "Best_Score": search.best_score_,
                "Best_Params": str(search.best_params_),
            }
        )

        print(f"✅ Pilot tuning complete for {label}: {search.best_params_}")

    return best_estimators, selected_params, pilot_rows


def evaluate_fixed_estimators(
    x: np.ndarray,
    y: np.ndarray,
    best_estimators: Dict[str, BaseEstimator],
    selected_params: Dict[str, Dict[str, Any]],
    config: ExperimentConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Evaluate fixed tuned estimators across outer folds.

    Args:
        x: Full feature matrix.
        y: Full target vector.
        best_estimators: Frozen estimators chosen on the pilot subset.
        selected_params: Hyperparameters selected on the pilot subset.
        config: Experiment configuration.

    Returns:
        Fold-level result rows and reconstructed full-length predictions.
    """
    outer_cv: KFold = KFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    fold_records: List[Dict[str, Any]] = []
    full_predictions: Dict[str, np.ndarray] = {key: np.full_like(y, np.nan, dtype=float) for key in _model_order}
    fold_i: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(x), start=1):
        print(f"\n--- Evaluating fixed-parameter fold {fold_i}/{config.outer_folds} ---")
        x_train_scaled: np.ndarray
        x_test_scaled: np.ndarray
        x_train_scaled, x_test_scaled = scale_non_geo_features(x[train_idx], x[test_idx])
        y_train: np.ndarray = y[train_idx]
        y_test: np.ndarray = y[test_idx]
        model_key: str

        for model_key in _model_order:
            label: str = _model_labels[model_key]
            estimator: BaseEstimator = safe_clone(best_estimators[model_key])
            estimator.fit(x_train_scaled, y_train)
            y_pred: np.ndarray = np.asarray(estimator.predict(x_test_scaled), dtype=float).ravel()
            full_predictions[model_key][test_idx] = y_pred
            valid: bool
            non_finite_fraction: float
            valid, non_finite_fraction = prediction_validity(y_pred=y_pred)

            if valid:
                metrics = compute_metrics(y_test, y_pred)
            else:
                metrics = {
                    "RMSE": float("nan"),
                    "MAE": float("nan"),
                    "R2": float("nan"),
                }

            fold_records.append(
                {
                    "Fold": fold_i,
                    "Model": label,
                    "Protocol": config.protocol,
                    "Valid": valid,
                    "Non_Finite_Pct": 100.0 * non_finite_fraction,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "R2": metrics["R2"],
                    "Selected_Params": str(selected_params[model_key]),
                }
            )

            if valid:
                print(
                    f"  > {label}: RMSE={metrics['RMSE']:.4f} | " f"MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f}"
                )
            else:
                print(f"  > {label}: invalid predictions " f"({100.0 * non_finite_fraction:.1f}% non-finite)")

    return fold_records, full_predictions


def evaluate_nested_cv(
    x: np.ndarray,
    y: np.ndarray,
    search_grids: Dict[str, Any],
    config: ExperimentConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Evaluate all models with nested cross-validation.

    Args:
        x: Full feature matrix.
        y: Full target vector.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        Fold-level result rows and reconstructed full-length predictions.
    """
    outer_cv: KFold = KFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    fold_records: List[Dict[str, Any]] = []
    full_predictions: Dict[str, np.ndarray] = {key: np.full_like(y, np.nan, dtype=float) for key in _model_order}
    fold_i: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(x), start=1):
        print(f"\n--- Nested CV fold {fold_i}/{config.outer_folds} ---")
        x_train_scaled: np.ndarray
        x_test_scaled: np.ndarray
        x_train_scaled, x_test_scaled = scale_non_geo_features(x[train_idx], x[test_idx])
        y_train: np.ndarray = y[train_idx]
        y_test: np.ndarray = y[test_idx]
        model_key: str

        for model_key in _model_order:
            label: str = _model_labels[model_key]

            search: Any = run_search(
                model_key=model_key, x_train=x_train_scaled, y_train=y_train, search_grids=search_grids, config=config
            )

            estimator: BaseEstimator = search.best_estimator_
            y_pred: np.ndarray = np.asarray(estimator.predict(x_test_scaled), dtype=float).ravel()
            full_predictions[model_key][test_idx] = y_pred
            valid: bool
            non_finite_fraction: float
            valid, non_finite_fraction = prediction_validity(y_pred=y_pred)
            metrics: Dict[str, float]

            if valid:
                metrics = compute_metrics(y_true=y_test, y_pred=y_pred)
            else:
                metrics = {
                    "RMSE": float("nan"),
                    "MAE": float("nan"),
                    "R2": float("nan"),
                }

            fold_records.append(
                {
                    "Fold": fold_i,
                    "Model": label,
                    "Protocol": config.protocol,
                    "Valid": valid,
                    "Non_Finite_Pct": 100.0 * non_finite_fraction,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "R2": metrics["R2"],
                    "Selected_Params": str(search.best_params_),
                }
            )

            if valid:
                print(
                    f"  > {label}: RMSE={metrics['RMSE']:.4f} | " f"MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f}"
                )
            else:
                print(f"  > {label}: invalid predictions " f"({100.0 * non_finite_fraction:.1f}% non-finite)")

    return fold_records, full_predictions


def evaluate_holdout_cv(
    x: np.ndarray,
    y: np.ndarray,
    search_grids: Dict[str, Any],
    config: ExperimentConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Evaluate all models on one or more held-out test splits.

    Args:
        x: Full feature matrix.
        y: Full target vector.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        Result rows and averaged full-length predictions on the rows that
        appeared in at least one held-out test split.
    """
    fold_records: List[Dict[str, Any]] = []
    prediction_sums: Dict[str, np.ndarray] = {key: np.zeros_like(y, dtype=float) for key in _model_order}
    prediction_counts: Dict[str, np.ndarray] = {key: np.zeros_like(y, dtype=int) for key in _model_order}

    repeat_i: int
    for repeat_i in range(config.holdout_repeats):
        train_idx: np.ndarray
        test_idx: np.ndarray

        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=config.holdout_fraction,
            shuffle=True,
            random_state=config.random_state + repeat_i,
        )

        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        print(
            "\n--- Holdout evaluation "
            f"{repeat_i + 1}/{config.holdout_repeats} "
            f"(train={1.0 - config.holdout_fraction:.0%}, test={config.holdout_fraction:.0%}) ---"
        )

        x_train_scaled: np.ndarray
        x_test_scaled: np.ndarray
        x_train_scaled, x_test_scaled = scale_non_geo_features(x[train_idx], x[test_idx])
        y_train: np.ndarray = y[train_idx]
        y_test: np.ndarray = y[test_idx]
        model_key: str

        for model_key in _model_order:
            label: str = _model_labels[model_key]
            search: Any = run_search(
                model_key=model_key,
                x_train=x_train_scaled,
                y_train=y_train,
                search_grids=search_grids,
                config=config,
            )

            estimator: BaseEstimator = search.best_estimator_
            y_pred: np.ndarray = np.asarray(estimator.predict(x_test_scaled), dtype=float).ravel()
            finite_mask: np.ndarray = np.isfinite(y_pred)
            prediction_sums[model_key][test_idx[finite_mask]] += y_pred[finite_mask]
            prediction_counts[model_key][test_idx[finite_mask]] += 1

            valid: bool
            non_finite_fraction: float
            valid, non_finite_fraction = prediction_validity(y_pred=y_pred)
            metrics: Dict[str, float]

            if valid:
                metrics = compute_metrics(y_true=y_test, y_pred=y_pred)
            else:
                metrics = {
                    "RMSE": float("nan"),
                    "MAE": float("nan"),
                    "R2": float("nan"),
                }

            fold_records.append(
                {
                    "Fold": repeat_i + 1,
                    "Model": label,
                    "Protocol": config.protocol,
                    "Valid": valid,
                    "Non_Finite_Pct": 100.0 * non_finite_fraction,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "R2": metrics["R2"],
                    "Selected_Params": str(search.best_params_),
                }
            )

            if valid:
                print(
                    f"  > {label}: RMSE={metrics['RMSE']:.4f} | " f"MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f}"
                )
            else:
                print(f"  > {label}: invalid predictions ({100.0 * non_finite_fraction:.1f}% non-finite)")

    full_predictions: Dict[str, np.ndarray] = {}
    model_key = ""
    for model_key in _model_order:
        averaged: np.ndarray = np.full_like(y, np.nan, dtype=float)
        nonzero_mask: np.ndarray = prediction_counts[model_key] > 0
        averaged[nonzero_mask] = prediction_sums[model_key][nonzero_mask] / prediction_counts[model_key][nonzero_mask]
        full_predictions[model_key] = averaged

    return fold_records, full_predictions


def save_outputs(
    x: np.ndarray,
    y: np.ndarray,
    fold_records: List[Dict[str, Any]],
    full_predictions: Dict[str, np.ndarray],
    config: ExperimentConfig,
    pilot_rows: List[Dict[str, Any]] | None = None,
) -> None:
    """
    Persist result tables, metadata, and figures.

    Args:
        x: Full feature matrix.
        y: Full target vector.
        fold_records: Fold-level metric rows.
        full_predictions: Reconstructed full-length predictions keyed by model.
        config: Experiment configuration.
        pilot_rows: Optional pilot-tuning audit rows.
    """
    output_dir: Path = _out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path: Path = output_dir / f"{config.output_prefix}_raw.csv"
    summary_path: Path = output_dir / f"{config.output_prefix}_summary.csv"
    metadata_path: Path = output_dir / f"{config.output_prefix}_metadata.json"
    pilot_path: Path = output_dir / f"{config.output_prefix}_pilot_selection.csv"

    df_results: pd.DataFrame = pd.DataFrame(data=fold_records)
    df_results.to_csv(raw_path, index=False)
    valid_results: pd.DataFrame = (
        df_results.loc[df_results["Valid"]].copy() if "Valid" in df_results.columns else df_results.copy()
    )

    summary: pd.DataFrame = valid_results.groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    summary.to_csv(summary_path)

    metadata: Dict[str, Any] = {
        "config": asdict(config),
        "metrics_reported": ["RMSE", "MAE", "R2"],
        "primary_metric": "RMSE",
        "models": [_model_labels[key] for key in _model_order],
        "reporting_uses_valid_runs_only": True,
    }

    metadata_path.write_text(data=json.dumps(metadata, indent=2))

    if pilot_rows is not None:
        pd.DataFrame(pilot_rows).to_csv(pilot_path, index=False)

    if not config.generate_plots:
        return

    evaluated_mask: np.ndarray = np.zeros_like(y, dtype=bool)
    y_pred_full: np.ndarray
    for y_pred_full in full_predictions.values():
        evaluated_mask |= np.isfinite(y_pred_full)

    if not np.any(evaluated_mask):
        return

    x_plot: np.ndarray = x[evaluated_mask]
    y_plot: np.ndarray = y[evaluated_mask]
    predictions_plot: Dict[str, np.ndarray] = {
        key: np.asarray(y_pred_full[evaluated_mask], dtype=float) for key, y_pred_full in full_predictions.items()
    }

    agg_results_for_plots: Dict[str, Dict[str, float]] = {}
    model_key: str
    y_pred_plot: np.ndarray

    for model_key, y_pred_plot in predictions_plot.items():
        valid: bool
        valid, _ = prediction_validity(y_pred_plot)

        if valid:
            agg_results_for_plots[model_key] = {
                "RMSE": float(np.sqrt(mean_squared_error(y_plot, y_pred_plot))),
                "MAE": float(mean_absolute_error(y_plot, y_pred_plot)),
                "R²": float(r2_score(y_plot, y_pred_plot)),
            }
        else:
            agg_results_for_plots[model_key] = {
                "RMSE": float("nan"),
                "MAE": float("nan"),
                "R²": float("nan"),
            }

    plot_prediction_scatter_grid(
        predictions=predictions_plot,
        results=agg_results_for_plots,
        y_test=y_plot,
        save_plots=True,
        output_dir=output_dir,
    )

    plot_geospatial_error_maps(
        predictions=predictions_plot,
        results=agg_results_for_plots,
        x_test=x_plot,
        y_test=y_plot,
        save_plots=True,
        output_dir=output_dir,
    )

    if config.protocol != "holdout_cv" or config.holdout_repeats > 1:
        save_rmse_fold_plot(
            df_results=valid_results,
            output_dir=output_dir,
            output_prefix=config.output_prefix,
        )


def save_rmse_fold_plot(
    df_results: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """
    Save the fold-wise RMSE summary figure.

    Args:
        df_results: Fold-level results table.
        output_dir: Output directory for the figure.
        output_prefix: Filename prefix for the saved figure.
    """
    if df_results.empty:
        return

    order: List[Any] = df_results.groupby("Model")["RMSE"].mean().sort_values().index.tolist()
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    data: List[np.ndarray] = [df_results.loc[df_results["Model"] == model, "RMSE"].to_numpy() for model in order]

    ax.boxplot(
        data,
        tick_labels=order,
        patch_artist=True,
        boxprops={"facecolor": "#d9e6f2", "edgecolor": "#4a4a4a"},
        medianprops={"color": "#c0392b", "linewidth": 2},
        whiskerprops={"color": "#4a4a4a"},
        capprops={"color": "#4a4a4a"},
    )

    index: int
    model: Any

    for index, model in enumerate(order, start=1):
        values: np.ndarray = df_results.loc[df_results["Model"] == model, "RMSE"].to_numpy()
        jitter: np.ndarray = np.linspace(-0.08, 0.08, num=len(values)) if len(values) > 1 else np.array([0.0])

        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            color="#1f5a91",
            alpha=0.75,
            s=20,
            zorder=3,
        )

    ax.set_title("Experiment 1: Fold-wise RMSE Distribution")
    ax.set_ylabel("RMSE")
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{output_prefix}_rmse_boxplot.png", dpi=200)
    plt.close(fig)


def normalize_config(config: ExperimentConfig) -> ExperimentConfig:
    """
    Normalize and validate the experiment configuration.

    Args:
        config: Raw experiment configuration.

    Returns:
        The normalized configuration.

    Raises:
        ValueError: If ``protocol`` or ``search_mode`` is unsupported.
    """
    if config.protocol not in {"pilot_tune_then_refit", "nested_cv", "holdout_cv"}:
        raise ValueError("protocol must be 'pilot_tune_then_refit', 'nested_cv', or 'holdout_cv'")

    if config.search_mode not in {"grid", "random"}:
        raise ValueError("search_mode must be 'grid' or 'random'")

    if not 0.0 < config.pilot_fraction < 1.0:
        raise ValueError("pilot_fraction must be strictly between 0 and 1")

    if not 0.0 < config.holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")

    if config.holdout_repeats < 1:
        raise ValueError("holdout_repeats must be at least 1")

    if config.smoke:
        config = replace(
            config,
            grid_profile="smoke",
            outer_folds=2,
            inner_folds=2,
            pilot_fraction=min(config.pilot_fraction, 0.10),
            holdout_fraction=min(config.holdout_fraction, 0.20),
            holdout_repeats=1,
            generate_plots=False,
            max_samples=config.max_samples or 800,
            output_prefix=f"{config.output_prefix}_smoke",
        )

    resolve_scoring(scoring=config.scoring)
    return config


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run California Housing Experiment 1.

    Args:
        config: Experiment configuration.

    Returns:
        Fold-level results as a DataFrame.
    """
    config = normalize_config(config)
    x: np.ndarray
    y: np.ndarray
    x, y = load_raw_data(max_samples=config.max_samples, seed=config.random_state)
    search_grids: Dict[str, Any] = build_search_grids(grid_profile=config.grid_profile)

    best_estimators: Dict[str, BaseEstimator]
    selected_params: Dict[str, Dict[str, Any]]
    pilot_rows: List[Dict[str, Any]] | None = None
    fold_records: List[Dict[str, Any]]
    full_predictions: Dict[str, np.ndarray]

    if config.protocol == "pilot_tune_then_refit":
        best_estimators, selected_params, pilot_rows = pilot_tune_models(
            x=x,
            y=y,
            search_grids=search_grids,
            config=config,
        )

        fold_records, full_predictions = evaluate_fixed_estimators(
            x=x,
            y=y,
            best_estimators=best_estimators,
            selected_params=selected_params,
            config=config,
        )
    elif config.protocol == "holdout_cv":
        fold_records, full_predictions = evaluate_holdout_cv(
            x=x,
            y=y,
            search_grids=search_grids,
            config=config,
        )
    else:
        fold_records, full_predictions = evaluate_nested_cv(
            x,
            y,
            search_grids,
            config,
        )

    save_outputs(
        x=x, y=y, fold_records=fold_records, full_predictions=full_predictions, config=config, pilot_rows=pilot_rows
    )

    return pd.DataFrame(fold_records)


def parse_args() -> ExperimentConfig:
    """
    Parse CLI arguments into an experiment configuration.

    Returns:
        The parsed experiment configuration.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run California Housing Experiment 1 with configurable CV protocol.",
    )

    parser.add_argument(
        "--protocol",
        choices=["pilot_tune_then_refit", "nested_cv", "holdout_cv"],
        default="holdout_cv",
    )

    parser.add_argument("--search-mode", choices=["grid", "random"], default="grid")

    parser.add_argument(
        "--grid-profile",
        choices=["full", "reduced", "smoke"],
        default="reduced",
    )

    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--pilot-fraction", type=float, default=0.30)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--random-search-n-iter", type=int, default=None)

    parser.add_argument(
        "--scoring",
        choices=[
            "strict_rmse",
            "neg_root_mean_squared_error",
            "neg_mean_squared_error",
        ],
        default="strict_rmse",
    )

    parser.add_argument("--output-prefix", type=str, default="california_exp1")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    return ExperimentConfig(
        protocol=args.protocol,
        search_mode=args.search_mode,
        grid_profile=args.grid_profile,
        scoring=args.scoring,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        pilot_fraction=args.pilot_fraction,
        holdout_fraction=args.holdout_fraction,
        holdout_repeats=args.holdout_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        random_search_n_iter=args.random_search_n_iter,
        generate_plots=not args.no_plots,
        output_prefix=args.output_prefix,
        smoke=args.smoke,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    dataframe: pd.DataFrame = run_experiment(config=parse_args())

    summary: pd.DataFrame = (
        dataframe.loc[dataframe["Valid"]].groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    )

    print("\nFinal summary:")
    print(summary.round(4))
