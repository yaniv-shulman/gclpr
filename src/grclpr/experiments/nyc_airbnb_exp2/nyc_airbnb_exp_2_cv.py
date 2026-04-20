"""Experiment 2 runner for the NYC Airbnb study."""

import argparse
import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, List, Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rsklpr.kernels import laplacian_normalized_metric, tricube_normalized_metric
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from grclpr.experiments.nyc_airbnb_exp2.nyc_airbnb import (
    calculate_unified_index_to_station_map,
    cv_knn,
    cv_rsklpr,
    cv_subway,
    load_subway_graph,
    plot_geospatial_error_maps,
    plot_prediction_scatter_grid,
)

_repo_dir = Path(__file__).resolve().parents[4]
_airbnb_data_path = _repo_dir / "data" / "air_bnb_nyc_listings" / "listings.csv"
_subway_graph_path = _repo_dir / "data" / "gtfs_subway" / "nyc_subway_graph.graphml"
_out_dir = _repo_dir / "out" / "experiment_2"

_model_order: list[str] = [
    "knn",
    "lpr",
    "rsklpr",
    "gclpr_subway",
    "grclpr_subway",
]

_model_labels: dict[str, str] = {
    "knn": "KNN",
    "lpr": "LPR",
    "rsklpr": "RSKLPR",
    "gclpr_subway": "GC-LPR (Subway)",
    "grclpr_subway": "GRC-LPR (Subway)",
}

_search_dispatch: dict[str, Callable[..., Any]] = {
    "knn": cv_knn,
    "lpr": cv_rsklpr,
    "rsklpr": cv_rsklpr,
    "gclpr_subway": cv_subway,
    "grclpr_subway": cv_subway,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for NYC Airbnb Experiment 2."""

    protocol: str = "pilot_tune_then_refit"
    search_mode: str = "grid"
    grid_profile: str = "full"
    scoring: str = "raw_price_rmse_from_log"
    outer_folds: int = 5
    inner_folds: int = 3
    pilot_fraction: float = 0.30
    holdout_fraction: float = 0.20
    holdout_repeats: int = 1
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0
    random_search_n_iter: int | None = None
    generate_plots: bool = True
    output_prefix: str = "nyc_airbnb_exp2"
    smoke: bool = False
    max_samples: int | None = None
    max_station_dist_km: float = 1.0


def load_raw_data(
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[int], list[str]]:
    """Load and prepare the NYC Airbnb dataset for Experiment 2.

    Args:
        max_samples: Optional number of observations to subsample.
        seed: Random seed used when subsampling.

    Returns:
        The feature matrix ``x``, the log-price target, the raw price target,
        the coordinate frame, the indices to scale, and the ordered feature names.
    """
    frame = pd.read_csv(_airbnb_data_path)

    selected_features = [
        "host_listings_count",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
    ]
    frame = frame[selected_features].copy()

    frame["price"] = frame["price"].replace({r"\$": "", ",": ""}, regex=True)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame.dropna(subset=["price"], inplace=True)
    frame = frame.loc[frame["price"] > 0].copy()

    cols_to_fill = [
        "host_listings_count",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
    ]
    frame[cols_to_fill] = frame[cols_to_fill].fillna(frame[cols_to_fill].median())

    frame["log_price"] = np.log1p(frame["price"])
    frame = pd.get_dummies(frame, columns=["room_type"], drop_first=True)

    main_features = [
        column for column in frame.columns if column not in {"price", "log_price", "latitude", "longitude"}
    ]
    geo_features = ["latitude", "longitude"]
    all_features = main_features + geo_features
    scale_columns = [
        "host_listings_count",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
    ]
    scale_indices = [all_features.index(column) for column in scale_columns if column in all_features]

    x = frame[all_features].to_numpy(copy=True, dtype=np.float32)
    y_log = frame["log_price"].to_numpy(copy=True, dtype=float)
    y_price = frame["price"].to_numpy(copy=True, dtype=float)
    coords = frame[geo_features].reset_index(drop=True).copy()

    if max_samples is not None and max_samples < len(y_price):
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(y_price), size=max_samples, replace=False)
        x = x[indices]
        y_log = y_log[indices]
        y_price = y_price[indices]
        coords = coords.iloc[indices].reset_index(drop=True)

    return x, y_log, y_price, coords, scale_indices, all_features


def scale_non_geo_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    scale_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Scale selected non-geospatial columns.

    Args:
        x_train: Training feature matrix.
        x_test: Test feature matrix.
        scale_indices: Column indices to standardize.

    Returns:
        The scaled training and test matrices.
    """
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()
    if not scale_indices:
        return x_train_scaled, x_test_scaled

    scaler = StandardScaler()
    x_train_scaled[:, scale_indices] = scaler.fit_transform(x_train[:, scale_indices])
    x_test_scaled[:, scale_indices] = scaler.transform(x_test[:, scale_indices])
    return x_train_scaled, x_test_scaled


def build_search_grids(grid_profile: str = "full") -> dict[str, Any]:
    """Build the hyperparameter search spaces for Experiment 2.

    Args:
        grid_profile: One of ``"full"``, ``"reduced"``, or ``"smoke"``.

    Returns:
        A mapping from model identifier to scikit-learn compatible search spaces.

    Raises:
        ValueError: If ``grid_profile`` is unsupported.
    """
    size_neighborhood_full = list(range(12, 21, 1))
    size_neighborhood_reduced = [12, 15, 17, 19]
    base_kernels = [tricube_normalized_metric, laplacian_normalized_metric]

    if grid_profile == "smoke":
        return {
            "knn": {
                "n_neighbors": [12],
                "weights": ["distance"],
                "p": [2],
            },
            "lpr": [
                {
                    "size_neighborhood": [12],
                    "degree": [0],
                    "kp": [tricube_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "rsklpr": [
                {
                    "size_neighborhood": [12],
                    "degree": [1],
                    "kp": [laplacian_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["joint"],
                }
            ],
            "gclpr_subway": [
                {
                    "size_neighborhood": [12],
                    "degree": [0],
                    "distance_scale": [1.0],
                    "kp": [tricube_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "grclpr_subway": [
                {
                    "size_neighborhood": [12],
                    "degree": [0],
                    "distance_scale": [1.0],
                    "kp": [laplacian_normalized_metric],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["conden"],
                }
            ],
        }

    if grid_profile == "reduced":
        size_neighborhood = size_neighborhood_reduced
        distance_scale = [1.0, 2.0]
        return {
            "knn": {
                "n_neighbors": list(range(3, 41, 2)),
                "weights": ["uniform", "distance"],
                "p": [1, 2],
                "leaf_size": [15, 30],
            },
            "lpr": [
                {
                    "size_neighborhood": size_neighborhood,
                    "degree": [0, 1],
                    "kp": base_kernels,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["none"],
                }
            ],
            "rsklpr": [
                {
                    "size_neighborhood": size_neighborhood,
                    "degree": [0, 1],
                    "kp": base_kernels,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["conden", "joint"],
                }
            ],
            "gclpr_subway": [
                {
                    "size_neighborhood": size_neighborhood,
                    "degree": [0, 1],
                    "distance_scale": distance_scale,
                    "kp": base_kernels,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["none"],
                }
            ],
            "grclpr_subway": [
                {
                    "size_neighborhood": size_neighborhood,
                    "degree": [0, 1],
                    "distance_scale": distance_scale,
                    "kp": base_kernels,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["conden", "joint"],
                }
            ],
        }

    if grid_profile == "full":
        distance_scale = [1.0, 1.5, 2.0]
        return {
            "knn": {
                "n_neighbors": list(range(3, 21, 1)),
                "weights": ["uniform", "distance"],
                "p": [1, 2],
                "leaf_size": [15, 30, 60],
            },
            "lpr": [
                {
                    "size_neighborhood": size_neighborhood_full,
                    "degree": [0, 1],
                    "kp": base_kernels,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["none"],
                },
                {
                    "size_neighborhood": size_neighborhood_full,
                    "degree": [0, 1],
                    "kp": base_kernels,
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["none"],
                },
            ],
            "rsklpr": [
                {
                    "size_neighborhood": size_neighborhood_full,
                    "degree": [0, 1],
                    "kp": base_kernels,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["conden", "joint"],
                },
                {
                    "size_neighborhood": size_neighborhood_full,
                    "degree": [0, 1],
                    "kp": base_kernels,
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["conden", "joint"],
                },
            ],
            "gclpr_subway": [
                {
                    "size_neighborhood": size_neighborhood_full,
                    "degree": [0, 1],
                    "distance_scale": distance_scale,
                    "kp": base_kernels,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["none"],
                },
                {
                    "size_neighborhood": size_neighborhood_full,
                    "degree": [0, 1],
                    "distance_scale": distance_scale,
                    "kp": base_kernels,
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["none"],
                },
            ],
            "grclpr_subway": [
                {
                    "size_neighborhood": size_neighborhood_full,
                    "degree": [0, 1],
                    "distance_scale": distance_scale,
                    "kp": base_kernels,
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}, {"p": 2}],
                    "kr": ["conden", "joint"],
                },
                {
                    "size_neighborhood": size_neighborhood_full,
                    "degree": [0, 1],
                    "distance_scale": distance_scale,
                    "kp": base_kernels,
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["conden", "joint"],
                },
            ],
        }

    raise ValueError("grid_profile must be one of {'full', 'reduced', 'smoke'}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics on raw price predictions.

    Args:
        y_true: Ground-truth raw prices.
        y_pred: Predicted raw prices. All values must be finite.

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
    """Clone an estimator, falling back to ``deepcopy`` when necessary.

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
    """Compute RMSE, treating any non-finite prediction as an invalid run.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        The RMSE when all predictions are finite, else ``inf``.
    """
    if not np.all(np.isfinite(y_pred)):
        return float("inf")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def raw_price_rmse_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Compute raw-price RMSE from log-price targets and predictions.

    Args:
        y_true_log: Ground-truth log-price targets.
        y_pred_log: Predicted log-prices.

    Returns:
        The raw-price RMSE when all predictions are finite, else ``inf``.
    """
    if not np.all(np.isfinite(y_pred_log)):
        return float("inf")

    y_true_price = inverse_transform_log_predictions(y_true_log)
    y_pred_price = inverse_transform_log_predictions(y_pred_log)
    return float(np.sqrt(mean_squared_error(y_true_price, y_pred_price)))


def prediction_validity(y_pred: np.ndarray) -> tuple[bool, float]:
    """Summarize whether a prediction vector is fully finite.

    Args:
        y_pred: Predicted target values.

    Returns:
        A tuple ``(is_valid, non_finite_fraction)``.
    """
    finite_mask = np.isfinite(y_pred)
    valid = bool(np.all(finite_mask))
    non_finite_fraction = float(1.0 - np.mean(finite_mask))
    return valid, non_finite_fraction


def resolve_scoring(scoring: str) -> Any:
    """Resolve the configured scoring string into a scorer object.

    Args:
        scoring: Scoring identifier from the experiment configuration.

    Returns:
        A scikit-learn compatible scoring object or scoring string.
    """
    if scoring == "raw_price_rmse_from_log":
        return make_scorer(raw_price_rmse_from_log, greater_is_better=False)
    if scoring == "strict_log_rmse":
        return make_scorer(strict_rmse, greater_is_better=False)
    if scoring in {"neg_root_mean_squared_error", "neg_mean_squared_error"}:
        return scoring
    raise ValueError(
        "Unsupported scoring value. Expected one of "
        "{'raw_price_rmse_from_log', 'strict_log_rmse', "
        "'neg_root_mean_squared_error', 'neg_mean_squared_error'}."
    )


def uses_subway_context(model_key: str) -> bool:
    """Return whether a model uses the subway-context wrapper."""
    return model_key in {"gclpr_subway", "grclpr_subway"}


def augment_with_original_indices(
    x: np.ndarray,
    original_indices: np.ndarray,
) -> np.ndarray:
    """Append original dataset indices to a feature matrix.

    Args:
        x: Feature matrix.
        original_indices: Original dataset row indices.

    Returns:
        The augmented feature matrix.
    """
    return np.hstack((x, original_indices.reshape(-1, 1)))


def inverse_transform_log_predictions(y_pred_log: np.ndarray) -> np.ndarray:
    """Convert log-price predictions back to raw prices.

    Args:
        y_pred_log: Predicted log-prices.

    Returns:
        Predicted raw prices, clipped below at zero.
    """
    y_pred_price = np.expm1(y_pred_log)
    return np.asarray(np.maximum(y_pred_price, 0.0), dtype=float)


def run_search(
    model_key: str,
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    train_indices: np.ndarray,
    subway_graph: nx.Graph,
    original_idx_to_station_map: dict[int, str | None],
    search_grids: dict[str, Any],
    config: ExperimentConfig,
) -> Any:
    """Run hyperparameter search for one model family.

    Args:
        model_key: Model identifier from ``_model_order``.
        x_train: Training feature matrix.
        y_train_log: Training log-price targets.
        train_indices: Original row indices for the training split.
        subway_graph: Subway graph.
        original_idx_to_station_map: Global listing-to-station mapping.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        The fitted search object.
    """
    search_fn = _search_dispatch[model_key]
    if uses_subway_context(model_key):
        return search_fn(
            x_train=x_train,
            y_train=y_train_log,
            train_indices=train_indices,
            subway_graph=subway_graph,
            original_idx_to_station_map=original_idx_to_station_map,
            param_grid=search_grids[model_key],
            cv=config.inner_folds,
            n_jobs=config.n_jobs,
            verbose=config.verbose,
            search_mode=config.search_mode,
            n_iter=config.random_search_n_iter,
            random_state=config.random_state,
            scoring=resolve_scoring(config.scoring),
        )

    return search_fn(
        x_train=x_train,
        y_train=y_train_log,
        param_grid=search_grids[model_key],
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
    y_log: np.ndarray,
    scale_indices: list[int],
    subway_graph: nx.Graph,
    original_idx_to_station_map: dict[int, str | None],
    search_grids: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Tune one model configuration per family on a pilot subset.

    Args:
        x: Full feature matrix.
        y_log: Full log-price target vector.
        scale_indices: Column indices to scale.
        subway_graph: Subway graph.
        original_idx_to_station_map: Global listing-to-station mapping.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        Best estimators, best parameter mappings, and pilot-tuning audit rows.
    """
    pilot_idx, _ = train_test_split(
        np.arange(len(y_log)),
        train_size=config.pilot_fraction,
        shuffle=True,
        random_state=config.random_state,
    )

    x_pilot = x[pilot_idx]
    y_pilot_log = y_log[pilot_idx]
    x_pilot_scaled, _ = scale_non_geo_features(
        x_train=x_pilot,
        x_test=x_pilot,
        scale_indices=scale_indices,
    )

    best_estimators: dict[str, BaseEstimator] = {}
    selected_params: dict[str, dict[str, Any]] = {}
    pilot_rows: list[dict[str, Any]] = []

    for model_key in _model_order:
        label = _model_labels[model_key]
        search = run_search(
            model_key=model_key,
            x_train=x_pilot_scaled,
            y_train_log=y_pilot_log,
            train_indices=pilot_idx,
            subway_graph=subway_graph,
            original_idx_to_station_map=original_idx_to_station_map,
            search_grids=search_grids,
            config=config,
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
    y_log: np.ndarray,
    y_price: np.ndarray,
    scale_indices: list[int],
    best_estimators: dict[str, BaseEstimator],
    selected_params: dict[str, dict[str, Any]],
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Evaluate fixed tuned estimators across outer folds.

    Args:
        x: Full feature matrix.
        y_log: Full log-price target vector.
        y_price: Full raw-price target vector.
        scale_indices: Column indices to scale.
        best_estimators: Frozen estimators chosen on the pilot subset.
        selected_params: Hyperparameters selected on the pilot subset.
        config: Experiment configuration.

    Returns:
        Fold-level result rows and reconstructed full-length price predictions.
    """
    outer_cv = KFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    fold_records: list[dict[str, Any]] = []
    full_predictions = {key: np.full_like(y_price, np.nan, dtype=float) for key in _model_order}

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(x), start=1):
        print(f"\n--- Evaluating fixed-parameter fold {fold_i}/{config.outer_folds} ---")
        x_train_scaled, x_test_scaled = scale_non_geo_features(
            x_train=x[train_idx],
            x_test=x[test_idx],
            scale_indices=scale_indices,
        )
        y_train_log = y_log[train_idx]
        y_test_price = y_price[test_idx]

        for model_key in _model_order:
            label = _model_labels[model_key]
            estimator = safe_clone(best_estimators[model_key])

            if uses_subway_context(model_key):
                x_train_input = augment_with_original_indices(x_train_scaled, train_idx)
                x_test_input = augment_with_original_indices(x_test_scaled, test_idx)
            else:
                x_train_input = x_train_scaled
                x_test_input = x_test_scaled

            estimator.fit(x_train_input, y_train_log)
            y_pred_log = np.asarray(estimator.predict(x_test_input), dtype=float).ravel()
            y_pred_price = inverse_transform_log_predictions(y_pred_log)
            full_predictions[model_key][test_idx] = y_pred_price

            valid, non_finite_fraction = prediction_validity(y_pred_price)
            if valid:
                metrics = compute_metrics(y_test_price, y_pred_price)
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
    y_log: np.ndarray,
    y_price: np.ndarray,
    scale_indices: list[int],
    subway_graph: nx.Graph,
    original_idx_to_station_map: dict[int, str | None],
    search_grids: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Evaluate all models with nested cross-validation.

    Args:
        x: Full feature matrix.
        y_log: Full log-price target vector.
        y_price: Full raw-price target vector.
        scale_indices: Column indices to scale.
        subway_graph: Subway graph.
        original_idx_to_station_map: Global listing-to-station mapping.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        Fold-level result rows and reconstructed full-length price predictions.
    """
    outer_cv = KFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    fold_records: list[dict[str, Any]] = []
    full_predictions = {key: np.full_like(y_price, np.nan, dtype=float) for key in _model_order}

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(x), start=1):
        print(f"\n--- Nested CV fold {fold_i}/{config.outer_folds} ---")
        x_train_scaled, x_test_scaled = scale_non_geo_features(
            x_train=x[train_idx],
            x_test=x[test_idx],
            scale_indices=scale_indices,
        )
        y_train_log = y_log[train_idx]
        y_test_price = y_price[test_idx]

        for model_key in _model_order:
            label = _model_labels[model_key]
            search = run_search(
                model_key=model_key,
                x_train=x_train_scaled,
                y_train_log=y_train_log,
                train_indices=train_idx,
                subway_graph=subway_graph,
                original_idx_to_station_map=original_idx_to_station_map,
                search_grids=search_grids,
                config=config,
            )

            estimator = search.best_estimator_
            if uses_subway_context(model_key):
                x_test_input = augment_with_original_indices(x_test_scaled, test_idx)
            else:
                x_test_input = x_test_scaled

            y_pred_log = np.asarray(estimator.predict(x_test_input), dtype=float).ravel()
            y_pred_price = inverse_transform_log_predictions(y_pred_log)
            full_predictions[model_key][test_idx] = y_pred_price

            valid, non_finite_fraction = prediction_validity(y_pred_price)
            if valid:
                metrics = compute_metrics(y_test_price, y_pred_price)
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
    y_log: np.ndarray,
    y_price: np.ndarray,
    scale_indices: List[int],
    subway_graph: nx.Graph,
    original_idx_to_station_map: Dict[int, str | None],
    search_grids: Dict[str, Any],
    config: ExperimentConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Evaluate all models on one or more held-out test splits.

    Args:
        x: Full feature matrix.
        y_log: Full log-price target vector.
        y_price: Full raw-price target vector.
        scale_indices: Column indices to scale.
        subway_graph: Subway graph.
        original_idx_to_station_map: Global listing-to-station mapping.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        Result rows and averaged full-length price predictions on the rows that
        appeared in at least one held-out test split.
    """
    fold_records: List[Dict[str, Any]] = []
    prediction_sums = {key: np.zeros_like(y_price, dtype=float) for key in _model_order}
    prediction_counts = {key: np.zeros_like(y_price, dtype=int) for key in _model_order}

    for repeat_i in range(config.holdout_repeats):
        train_idx: np.ndarray
        test_idx: np.ndarray

        train_idx, test_idx = train_test_split(
            np.arange(len(y_price)),
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

        x_train_scaled, x_test_scaled = scale_non_geo_features(
            x_train=x[train_idx],
            x_test=x[test_idx],
            scale_indices=scale_indices,
        )

        y_train_log = y_log[train_idx]
        y_test_price = y_price[test_idx]

        for model_key in _model_order:
            label = _model_labels[model_key]
            search = run_search(
                model_key=model_key,
                x_train=x_train_scaled,
                y_train_log=y_train_log,
                train_indices=train_idx,
                subway_graph=subway_graph,
                original_idx_to_station_map=original_idx_to_station_map,
                search_grids=search_grids,
                config=config,
            )
            estimator = search.best_estimator_

            if uses_subway_context(model_key):
                x_test_input = augment_with_original_indices(x_test_scaled, test_idx)
            else:
                x_test_input = x_test_scaled

            y_pred_log = np.asarray(estimator.predict(x_test_input), dtype=float).ravel()
            y_pred_price = inverse_transform_log_predictions(y_pred_log)
            finite_mask = np.isfinite(y_pred_price)
            prediction_sums[model_key][test_idx[finite_mask]] += y_pred_price[finite_mask]
            prediction_counts[model_key][test_idx[finite_mask]] += 1

            valid, non_finite_fraction = prediction_validity(y_pred_price)
            if valid:
                metrics = compute_metrics(y_test_price, y_pred_price)
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
                    f"  > {label}: RMSE={metrics['RMSE']:.4f} | "
                    f"MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f}"
                )
            else:
                print(f"  > {label}: invalid predictions ({100.0 * non_finite_fraction:.1f}% non-finite)")

    full_predictions: dict[str, np.ndarray] = {}

    for model_key in _model_order:
        averaged = np.full_like(y_price, np.nan, dtype=float)
        nonzero_mask = prediction_counts[model_key] > 0
        averaged[nonzero_mask] = prediction_sums[model_key][nonzero_mask] / prediction_counts[model_key][nonzero_mask]
        full_predictions[model_key] = averaged

    return fold_records, full_predictions


def save_outputs(
    coords: pd.DataFrame,
    y_price: np.ndarray,
    fold_records: list[dict[str, Any]],
    full_predictions: dict[str, np.ndarray],
    config: ExperimentConfig,
    pilot_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Persist result tables, metadata, and figures.

    Args:
        coords: Coordinate frame aligned with the full dataset.
        y_price: Full raw-price target vector.
        fold_records: Fold-level metric rows.
        full_predictions: Reconstructed full-length predictions keyed by model.
        config: Experiment configuration.
        pilot_rows: Optional pilot-tuning audit rows.
    """
    output_dir = _out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{config.output_prefix}_raw.csv"
    summary_path = output_dir / f"{config.output_prefix}_summary.csv"
    metadata_path = output_dir / f"{config.output_prefix}_metadata.json"
    pilot_path = output_dir / f"{config.output_prefix}_pilot_selection.csv"

    df_results = pd.DataFrame(fold_records)
    df_results.to_csv(raw_path, index=False)

    valid_results = df_results.loc[df_results["Valid"]].copy() if "Valid" in df_results.columns else df_results.copy()
    summary = valid_results.groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    summary.to_csv(summary_path)

    metadata = {
        "config": asdict(config),
        "metrics_reported": ["RMSE", "MAE", "R2"],
        "primary_metric": "RMSE",
        "models": [_model_labels[key] for key in _model_order],
        "reporting_target_space": "price",
        "tuning_target_space": "price (via inverse-transformed log-price predictions)",
        "reporting_uses_valid_runs_only": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if pilot_rows is not None:
        pd.DataFrame(pilot_rows).to_csv(pilot_path, index=False)

    if not config.generate_plots:
        return

    evaluated_mask = np.zeros_like(y_price, dtype=bool)
    for y_pred_full in full_predictions.values():
        evaluated_mask |= np.isfinite(y_pred_full)

    if not np.any(evaluated_mask):
        return

    coords_plot = coords.loc[evaluated_mask].reset_index(drop=True)
    y_price_plot = y_price[evaluated_mask]
    predictions_plot = {
        key: np.asarray(y_pred_full[evaluated_mask], dtype=float)
        for key, y_pred_full in full_predictions.items()
    }

    agg_results_for_plots: dict[str, dict[str, float]] = {}
    for model_key, y_pred_plot in predictions_plot.items():
        valid, _ = prediction_validity(y_pred_plot)
        if valid:
            agg_results_for_plots[model_key] = {
                "RMSE": float(np.sqrt(mean_squared_error(y_price_plot, y_pred_plot))),
                "MAE": float(mean_absolute_error(y_price_plot, y_pred_plot)),
                "R2": float(r2_score(y_price_plot, y_pred_plot)),
            }
        else:
            agg_results_for_plots[model_key] = {
                "RMSE": float("nan"),
                "MAE": float("nan"),
                "R2": float("nan"),
            }

    plot_prediction_scatter_grid(
        predictions=predictions_plot,
        results=agg_results_for_plots,
        y_true=y_price_plot,
        save_plots=True,
        output_dir=output_dir,
    )
    plot_geospatial_error_maps(
        predictions=predictions_plot,
        results=agg_results_for_plots,
        coords=coords_plot,
        y_true=y_price_plot,
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
    """Save the fold-wise RMSE summary figure.

    Args:
        df_results: Fold-level results table.
        output_dir: Output directory for the figure.
        output_prefix: Filename prefix for the saved figure.
    """
    if df_results.empty:
        return

    order = df_results.groupby("Model")["RMSE"].mean().sort_values().index.tolist()
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    data = [df_results.loc[df_results["Model"] == model, "RMSE"].to_numpy() for model in order]
    ax.boxplot(
        data,
        tick_labels=order,
        patch_artist=True,
        boxprops={"facecolor": "#d9e6f2", "edgecolor": "#4a4a4a"},
        medianprops={"color": "#c0392b", "linewidth": 2},
        whiskerprops={"color": "#4a4a4a"},
        capprops={"color": "#4a4a4a"},
    )

    for index, model in enumerate(order, start=1):
        values = df_results.loc[df_results["Model"] == model, "RMSE"].to_numpy()
        jitter = np.linspace(-0.08, 0.08, num=len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            color="#1f5a91",
            alpha=0.75,
            s=20,
            zorder=3,
        )

    ax.set_title("Experiment 2: Fold-wise RMSE Distribution")
    ax.set_ylabel("RMSE")
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{output_prefix}_rmse_boxplot.png", dpi=200)
    plt.close(fig)


def normalize_config(config: ExperimentConfig) -> ExperimentConfig:
    """Normalize and validate the experiment configuration.

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
            max_samples=config.max_samples or 1000,
            output_prefix=f"{config.output_prefix}_smoke",
        )

    resolve_scoring(config.scoring)
    return config


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run NYC Airbnb Experiment 2.

    Args:
        config: Experiment configuration.

    Returns:
        Fold-level results as a DataFrame.
    """
    config = normalize_config(config)
    subway_graph = load_subway_graph(_subway_graph_path)
    x, y_log, y_price, coords, scale_indices, feature_names = load_raw_data(
        max_samples=config.max_samples,
        seed=config.random_state,
    )
    del feature_names
    original_idx_to_station_map = calculate_unified_index_to_station_map(
        subway_graph=subway_graph,
        listing_coords=coords,
        max_station_dist_km=config.max_station_dist_km,
    )
    search_grids = build_search_grids(grid_profile=config.grid_profile)

    pilot_rows = None
    if config.protocol == "pilot_tune_then_refit":
        best_estimators, selected_params, pilot_rows = pilot_tune_models(
            x=x,
            y_log=y_log,
            scale_indices=scale_indices,
            subway_graph=subway_graph,
            original_idx_to_station_map=original_idx_to_station_map,
            search_grids=search_grids,
            config=config,
        )
        fold_records, full_predictions = evaluate_fixed_estimators(
            x=x,
            y_log=y_log,
            y_price=y_price,
            scale_indices=scale_indices,
            best_estimators=best_estimators,
            selected_params=selected_params,
            config=config,
        )
    elif config.protocol == "holdout_cv":
        fold_records, full_predictions = evaluate_holdout_cv(
            x=x,
            y_log=y_log,
            y_price=y_price,
            scale_indices=scale_indices,
            subway_graph=subway_graph,
            original_idx_to_station_map=original_idx_to_station_map,
            search_grids=search_grids,
            config=config,
        )
    else:
        fold_records, full_predictions = evaluate_nested_cv(
            x=x,
            y_log=y_log,
            y_price=y_price,
            scale_indices=scale_indices,
            subway_graph=subway_graph,
            original_idx_to_station_map=original_idx_to_station_map,
            search_grids=search_grids,
            config=config,
        )

    save_outputs(
        coords=coords,
        y_price=y_price,
        fold_records=fold_records,
        full_predictions=full_predictions,
        config=config,
        pilot_rows=pilot_rows,
    )
    return pd.DataFrame(fold_records)


def parse_args() -> ExperimentConfig:
    """Parse CLI arguments into an experiment configuration.

    Returns:
        The parsed experiment configuration.
    """
    parser = argparse.ArgumentParser(
        description="Run NYC Airbnb Experiment 2 with configurable CV protocol.",
    )
    parser.add_argument(
        "--protocol",
        choices=["pilot_tune_then_refit", "nested_cv", "holdout_cv"],
        default="pilot_tune_then_refit",
    )
    parser.add_argument("--search-mode", choices=["grid", "random"], default="grid")
    parser.add_argument(
        "--grid-profile",
        choices=["full", "reduced", "smoke"],
        default="full",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--pilot-fraction", type=float, default=0.30)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-repeats", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--random-search-n-iter", type=int, default=None)
    parser.add_argument(
        "--scoring",
        choices=[
            "raw_price_rmse_from_log",
            "strict_log_rmse",
            "neg_root_mean_squared_error",
            "neg_mean_squared_error",
        ],
        default="raw_price_rmse_from_log",
    )
    parser.add_argument("--output-prefix", type=str, default="nyc_airbnb_exp2")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-station-dist-km", type=float, default=1.0)
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
        max_station_dist_km=args.max_station_dist_km,
    )


if __name__ == "__main__":
    dataframe = run_experiment(parse_args())
    summary = dataframe.loc[dataframe["Valid"]].groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    print("\nFinal summary:")
    print(summary.round(4))
