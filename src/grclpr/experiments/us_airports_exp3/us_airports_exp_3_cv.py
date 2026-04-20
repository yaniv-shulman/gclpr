"""Experiment 3 runner for the US airport network study."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rsklpr.kernels import tricube_normalized_metric
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from grclpr.experiments.us_airports_exp3.us_airports_network import (
    build_original_idx_to_node_map,
    cv_graph,
    cv_knn,
    cv_rsklpr,
    load_airport_data_and_graph,
    plot_airport_network_map,
    plot_graph_kernel_similarity,
    plot_prediction_scatter_grid,
    scale_non_positional_features,
)

_repo_dir: Path = Path(__file__).resolve().parents[4]
_airports_csv_path: Path = _repo_dir / "data" / "airports" / "airports.csv"
_routes_csv_path: Path = _repo_dir / "data" / "airports" / "flights-airport.csv"
_out_dir: Path = _repo_dir / "out" / "experiment_3"

_model_order: List[str] = [
    "knn",
    "lpr",
    "rsklpr",
    "gclpr_graph",
    "grclpr_graph",
]

_model_labels: Dict[str, str] = {
    "knn": "KNN",
    "lpr": "LPR",
    "rsklpr": "RSKLPR",
    "gclpr_graph": "GC-LPR (Graph)",
    "grclpr_graph": "GRC-LPR (Graph)",
}

_search_dispatch: Dict[str, Callable[..., Any]] = {
    "knn": cv_knn,
    "lpr": cv_rsklpr,
    "rsklpr": cv_rsklpr,
    "gclpr_graph": cv_graph,
    "grclpr_graph": cv_graph,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for US airport network Experiment 3."""

    protocol: str = "holdout_cv"
    search_mode: str = "grid"
    grid_profile: str = "full"
    scoring: str = "strict_rmse"
    holdout_fraction: float = 0.20
    holdout_repeats: int = 5
    inner_folds: int = 4
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0
    random_search_n_iter: int | None = None
    generate_plots: bool = True
    output_prefix: str = "us_airports_exp3"
    smoke: bool = False
    max_samples: int | None = None
    count_weighted_delay: bool = True


def load_raw_data(
    max_samples: int | None = None,
    seed: int = 42,
    count_weighted_delay: bool = True,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray, pd.DataFrame, Dict[int, str], List[str]]:
    """
    Load the airport graph dataset and align features, target, and coordinates.

    Args:
        max_samples: Optional subsample size for smoke tests or debugging.
        seed: Random seed used when subsampling.
        count_weighted_delay: Whether to synthesize the delay target with
            count-weighted propagation.

    Returns:
        The airport graph, feature matrix, target vector, coordinate frame,
        row-index to node-ID mapping, and ordered feature names.
    """
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    airport_network: nx.Graph
    df_features: pd.DataFrame
    x: np.ndarray
    y: np.ndarray

    airport_network, df_features, x, y = load_airport_data_and_graph(
        airports_csv_path=_airports_csv_path,
        routes_csv_path=_routes_csv_path,
        random_generator=rng,
        count_weighted_delay=count_weighted_delay,
    )

    if max_samples is not None and max_samples < len(y):
        indices = np.sort(rng.choice(len(y), size=max_samples, replace=False))
        df_features = df_features.iloc[indices].copy()
        x = x[indices]
        y = y[indices]
        airport_network = airport_network.subgraph(df_features.index).copy()

        nx.set_node_attributes(
            airport_network,
            values=dict(zip(df_features.index, y)),
            name="delay",
        )

    coords: pd.DataFrame = df_features[["latitude", "longitude"]].reset_index(drop=True).copy()
    original_idx_to_node_map: Dict[int, str] = build_original_idx_to_node_map(df_features=df_features)
    feature_names: List[str] = list(df_features.columns)
    return airport_network, x, y, coords, original_idx_to_node_map, feature_names


def build_search_grids(grid_profile: str = "full") -> Dict[str, Any]:
    """
    Build the hyperparameter search spaces for Experiment 3.

    Args:
        grid_profile: One of ``"full"``, ``"reduced"``, or ``"smoke"``.

    Returns:
        A mapping from model identifier to scikit-learn compatible search spaces.

    Raises:
        ValueError: If ``grid_profile`` is unsupported.
    """
    # Experiment 3 behaves differently from Experiments 1 and 2. The feature-only
    # local smoothers tend to prefer much broader neighborhoods, while the
    # graph-context models benefit from a smaller graph hop-distance scale. The
    # reduced profile is an exploratory coarse grid; the full profile is a
    # denser final grid centered on the same regime.
    size_neighborhood_feature_reduced: List[int] = [31, 63, 95, 127, 159]
    size_neighborhood_feature_full: List[int] = [79, 87, 95, 103, 111]
    size_neighborhood_graph_reduced: List[int] = [63, 95, 127, 159]
    size_neighborhood_graph_full: List[int] = [111, 119, 127, 135, 143, 159]
    graph_distance_reduced: List[float] = [0.2, 0.35, 0.5, 0.75, 1.0]
    graph_distance_full: List[float] = [0.25, 0.3, 0.35, 0.4, 0.5, 0.75]
    feature_kernel = tricube_normalized_metric

    if grid_profile == "smoke":
        return {
            "knn": {
                "n_neighbors": [5],
                "weights": ["distance"],
                "p": [1],
                "leaf_size": [30],
            },
            "lpr": [
                {
                    "size_neighborhood": [31],
                    "degree": [1],
                    "kp": [feature_kernel],
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["none"],
                }
            ],
            "rsklpr": [
                {
                    "size_neighborhood": [31],
                    "degree": [1],
                    "kp": [feature_kernel],
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["joint"],
                }
            ],
            "gclpr_graph": [
                {
                    "size_neighborhood": [31],
                    "degree": [1],
                    "distance_scale": [1.0],
                    "kp": [feature_kernel],
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["none"],
                }
            ],
            "grclpr_graph": [
                {
                    "size_neighborhood": [31],
                    "degree": [1],
                    "distance_scale": [1.0],
                    "kp": [feature_kernel],
                    "metric_x": ["mahalanobis"],
                    "metric_x_params": [None],
                    "kr": ["conden"],
                }
            ],
        }

    if grid_profile == "reduced":
        knn_neighbors = [1, 2, 3, 4, 5, 7, 9]
        feature_neighborhood = size_neighborhood_feature_reduced
        graph_neighborhood = size_neighborhood_graph_reduced
        graph_distance = graph_distance_reduced
    elif grid_profile == "full":
        knn_neighbors = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 41]
        feature_neighborhood = size_neighborhood_feature_full
        graph_neighborhood = size_neighborhood_graph_full
        graph_distance = graph_distance_full
    else:
        raise ValueError(f"Unsupported grid_profile={grid_profile!r}")

    return {
        "knn": {
            "n_neighbors": knn_neighbors,
            "weights": ["distance"],
            "p": [1],
            "leaf_size": [30],
        },
        "lpr": [
            {
                "size_neighborhood": feature_neighborhood,
                "degree": [1],
                "kp": [feature_kernel],
                "metric_x": ["mahalanobis"],
                "metric_x_params": [None],
                "kr": ["none"],
            }
        ],
        "rsklpr": [
            {
                "size_neighborhood": feature_neighborhood,
                "degree": [1],
                "kp": [feature_kernel],
                "metric_x": ["mahalanobis"],
                "metric_x_params": [None],
                "kr": ["joint"],
            }
        ],
        "gclpr_graph": [
            {
                "size_neighborhood": graph_neighborhood,
                "degree": [1],
                "distance_scale": graph_distance,
                "kp": [feature_kernel],
                "metric_x": ["mahalanobis"],
                "metric_x_params": [None],
                "kr": ["none"],
            }
        ],
        "grclpr_graph": [
            {
                "size_neighborhood": graph_neighborhood,
                "degree": [1],
                "distance_scale": graph_distance,
                "kp": [feature_kernel],
                "metric_x": ["mahalanobis"],
                "metric_x_params": [None],
                "kr": ["conden"],
            }
        ],
    }


def strict_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute RMSE, treating any non-finite prediction as invalid.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        The RMSE if all predictions are finite, else ``inf``.
    """
    if not np.all(np.isfinite(y_pred)):
        return float("inf")

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² for a valid prediction vector.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        A metric mapping with keys ``RMSE``, ``MAE``, and ``R2``.

    Raises:
        ValueError: If ``y_pred`` contains non-finite values.
    """
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("compute_metrics requires all predictions to be finite.")
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def prediction_validity(y_pred: np.ndarray) -> Tuple[bool, float]:
    """
    Determine whether a prediction vector is fully finite.

    Args:
        y_pred: Predicted values.

    Returns:
        A boolean validity flag and the fraction of non-finite predictions.
    """
    finite_mask: np.ndarray = np.isfinite(y_pred)
    non_finite_fraction: float = 1.0 - float(np.mean(finite_mask))
    return bool(np.all(finite_mask)), non_finite_fraction


def resolve_scoring(scoring: str) -> Any:
    """
    Resolve a configured scoring string to a scikit-learn scoring object.

    Args:
        scoring: Configured scoring name.

    Returns:
        A scorer object or native sklearn scoring string.

    Raises:
        ValueError: If the scoring name is unsupported.
    """
    if scoring == "strict_rmse":
        return make_scorer(strict_rmse, greater_is_better=False)

    if scoring in {"neg_mean_squared_error", "r2"}:
        return scoring

    raise ValueError(f"Unsupported scoring={scoring!r}.")


def uses_graph_context(model_key: str) -> bool:
    """Return whether a model uses the graph-context kernel.

    Args:
        model_key: Internal model identifier.

    Returns:
        ``True`` when the model includes the airport-graph context kernel.
    """
    return model_key in {"gclpr_graph", "grclpr_graph"}


def augment_with_original_indices(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Append original row indices for graph-aware estimators.

    Args:
        x: Feature matrix without row-index metadata.
        indices: Original row indices aligned with ``x``.

    Returns:
        The feature matrix with ``indices`` appended as the last column.
    """
    return np.hstack((x, indices.reshape(-1, 1)))


def run_search(
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: np.ndarray,
    airport_network: nx.Graph,
    original_idx_to_node_map: Mapping[int, str | None],
    search_grids: Dict[str, Any],
    config: ExperimentConfig,
) -> Any:
    """
    Dispatch model search to the appropriate helper.

    Args:
        model_key: Internal model identifier.
        x_train: Scaled training feature matrix.
        y_train: Training targets.
        train_indices: Original row indices for ``x_train``.
        airport_network: Airport graph.
        original_idx_to_node_map: Global row-index to node-ID mapping.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        The fitted search object.
    """
    search_fn: Callable[..., Any] = _search_dispatch[model_key]
    scoring: Any = resolve_scoring(scoring=config.scoring)

    if uses_graph_context(model_key):
        return search_fn(
            x_train=x_train,
            y_train=y_train,
            train_indices=train_indices,
            airport_network=airport_network,
            original_idx_to_node_map=original_idx_to_node_map,
            param_grid=search_grids[model_key],
            cv=config.inner_folds,
            n_jobs=config.n_jobs,
            verbose=config.verbose,
            search_mode=config.search_mode,
            n_iter=config.random_search_n_iter,
            random_state=config.random_state,
            scoring=scoring,
        )

    return search_fn(
        x_train=x_train,
        y_train=y_train,
        param_grid=search_grids[model_key],
        cv=config.inner_folds,
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        search_mode=config.search_mode,
        n_iter=config.random_search_n_iter,
        random_state=config.random_state,
        scoring=scoring,
    )


def evaluate_holdout_cv(
    x: np.ndarray,
    y: np.ndarray,
    airport_network: nx.Graph,
    original_idx_to_node_map: Mapping[int, str | None],
    search_grids: Dict[str, Any],
    config: ExperimentConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Evaluate all models on repeated held-out test splits.

    Args:
        x: Full feature matrix.
        y: Full target vector.
        airport_network: Airport graph.
        original_idx_to_node_map: Global row-index to node-ID mapping.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        Result rows and averaged full-length predictions on rows that appeared in
        at least one held-out split.
    """
    fold_records: List[Dict[str, Any]] = []

    prediction_sums: Dict[str, np.ndarray] = {key: np.zeros_like(y, dtype=float) for key in _model_order}

    prediction_counts: Dict[str, np.ndarray] = {key: np.zeros_like(y, dtype=int) for key in _model_order}

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

        x_train_scaled, x_test_scaled = scale_non_positional_features(
            x_train=x[train_idx],
            x_test=x[test_idx],
        )

        y_train: np.ndarray = y[train_idx]
        y_test: np.ndarray = y[test_idx]
        model_key: str

        for model_key in _model_order:
            label: str = _model_labels[model_key]

            search: Any = run_search(
                model_key=model_key,
                x_train=x_train_scaled,
                y_train=y_train,
                train_indices=train_idx,
                airport_network=airport_network,
                original_idx_to_node_map=original_idx_to_node_map,
                search_grids=search_grids,
                config=config,
            )

            estimator: Any = search.best_estimator_
            x_test_input: np.ndarray

            if uses_graph_context(model_key):
                x_test_input = augment_with_original_indices(x=x_test_scaled, indices=test_idx)
            else:
                x_test_input = x_test_scaled

            y_pred: np.ndarray = np.asarray(estimator.predict(x_test_input), dtype=float).ravel()
            finite_mask: np.ndarray = np.isfinite(y_pred)
            prediction_sums[model_key][test_idx[finite_mask]] += y_pred[finite_mask]
            prediction_counts[model_key][test_idx[finite_mask]] += 1

            valid: bool
            non_finite_fraction: float
            valid, non_finite_fraction = prediction_validity(y_pred=y_pred)
            metrics: Dict[str, float]
            if valid:
                metrics = compute_metrics(y_test, y_pred)
            else:
                metrics = {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}

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

    for model_key in _model_order:
        averaged: np.ndarray = np.full_like(y, np.nan, dtype=float)
        nonzero_mask: np.ndarray = prediction_counts[model_key] > 0
        averaged[nonzero_mask] = prediction_sums[model_key][nonzero_mask] / prediction_counts[model_key][nonzero_mask]
        full_predictions[model_key] = averaged

    return fold_records, full_predictions


def save_rmse_fold_plot(
    df_results: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """
    Save the repeat-wise RMSE summary figure.

    Args:
        df_results: Repeat-level results table.
        output_dir: Output directory for the figure.
        output_prefix: Filename prefix for the saved figure.
    """
    if df_results.empty:
        return

    order: List[str] = df_results.groupby("Model")["RMSE"].mean().sort_values().index.tolist()
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
    model: str

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

    ax.set_title("Experiment 3: Held-out RMSE Distribution")
    ax.set_ylabel("RMSE")
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{output_prefix}_rmse_boxplot.png", dpi=200)
    plt.close(fig)


def save_outputs(
    airport_network: nx.Graph,
    y: np.ndarray,
    fold_records: List[Dict[str, Any]],
    full_predictions: Dict[str, np.ndarray],
    config: ExperimentConfig,
) -> None:
    """
    Persist result tables, metadata, and figures.

    Args:
        airport_network: Airport graph used in the experiment.
        y: Full target vector.
        fold_records: Repeat-level metric rows.
        full_predictions: Reconstructed full-length predictions keyed by model.
        config: Experiment configuration.
    """
    output_dir: Path = _out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path: Path = output_dir / f"{config.output_prefix}_raw.csv"
    summary_path: Path = output_dir / f"{config.output_prefix}_summary.csv"
    metadata_path: Path = output_dir / f"{config.output_prefix}_metadata.json"

    df_results: pd.DataFrame = pd.DataFrame(fold_records)
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
        "reporting_target_space": "synthetic delay",
        "reporting_uses_valid_runs_only": True,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))

    if not config.generate_plots:
        return

    evaluated_mask: np.ndarray = np.zeros_like(y, dtype=bool)
    y_pred_full: np.ndarray

    for y_pred_full in full_predictions.values():
        evaluated_mask |= np.isfinite(y_pred_full)

    if not np.any(evaluated_mask):
        return

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
                "R2": float(r2_score(y_plot, y_pred_plot)),
            }
        else:
            agg_results_for_plots[model_key] = {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}

    plot_prediction_scatter_grid(
        predictions=predictions_plot,
        results=agg_results_for_plots,
        y_true=y_plot,
        save_plots=True,
        output_dir=output_dir,
    )

    plot_airport_network_map(
        airport_network=airport_network,
        save_plot=True,
        output_dir=output_dir,
    )

    plot_graph_kernel_similarity(
        airport_network=airport_network,
        reference_node_id="ATL",
        distance_scale=1.0,
        save_plot=True,
        output_dir=output_dir,
    )

    plot_graph_kernel_similarity(
        airport_network=airport_network,
        reference_node_id="HOU",
        distance_scale=1.0,
        save_plot=True,
        output_dir=output_dir,
    )

    save_rmse_fold_plot(
        df_results=valid_results,
        output_dir=output_dir,
        output_prefix=config.output_prefix,
    )


def normalize_config(config: ExperimentConfig) -> ExperimentConfig:
    """
    Normalize and validate the experiment configuration.

    Args:
        config: Raw experiment configuration.

    Returns:
        The normalized configuration.

    Raises:
        ValueError: If a configuration field is unsupported or invalid.
    """
    if config.protocol != "holdout_cv":
        raise ValueError("protocol must be 'holdout_cv'")

    if config.search_mode not in {"grid", "random"}:
        raise ValueError("search_mode must be 'grid' or 'random'")

    if not 0.0 < config.holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")

    if config.holdout_repeats < 1:
        raise ValueError("holdout_repeats must be at least 1")

    if config.smoke:
        config = replace(
            config,
            grid_profile="smoke",
            holdout_fraction=min(config.holdout_fraction, 0.20),
            holdout_repeats=2,
            inner_folds=2,
            generate_plots=False,
            max_samples=config.max_samples or 80,
            output_prefix=f"{config.output_prefix}_smoke",
        )

    resolve_scoring(config.scoring)
    return config


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run US airport network Experiment 3.

    Args:
        config: Experiment configuration.

    Returns:
        Repeat-level results as a DataFrame.
    """
    config = normalize_config(config)
    airport_network: nx.Graph
    x: np.ndarray
    y: np.ndarray
    original_idx_to_node_map: Dict[int, str]
    feature_names: List[str]

    airport_network, x, y, _, original_idx_to_node_map, feature_names = load_raw_data(
        max_samples=config.max_samples,
        seed=config.random_state,
        count_weighted_delay=config.count_weighted_delay,
    )

    del feature_names

    search_grids: Dict[str, Any] = build_search_grids(grid_profile=config.grid_profile)
    fold_records: List[Dict[str, Any]]
    full_predictions: Dict[str, np.ndarray]

    fold_records, full_predictions = evaluate_holdout_cv(
        x=x,
        y=y,
        airport_network=airport_network,
        original_idx_to_node_map=original_idx_to_node_map,
        search_grids=search_grids,
        config=config,
    )

    save_outputs(
        airport_network=airport_network,
        y=y,
        fold_records=fold_records,
        full_predictions=full_predictions,
        config=config,
    )

    return pd.DataFrame(fold_records)


def parse_args() -> ExperimentConfig:
    """Parse CLI arguments into an experiment configuration.

    Returns:
        The parsed experiment configuration.
    """
    parser = argparse.ArgumentParser(
        description="Run the US airport network Experiment 3 with repeated holdout evaluation.",
    )

    parser.add_argument(
        "--protocol",
        choices=["holdout_cv"],
        default="holdout_cv",
    )

    parser.add_argument("--search-mode", choices=["grid", "random"], default="grid")

    parser.add_argument(
        "--grid-profile",
        choices=["full", "reduced", "smoke"],
        default="full",
    )

    parser.add_argument(
        "--scoring",
        choices=["strict_rmse", "neg_mean_squared_error", "r2"],
        default="strict_rmse",
    )

    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--random-search-n-iter", type=int, default=None)
    parser.add_argument("--output-prefix", type=str, default="us_airports_exp3")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--unweighted-delay", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args: argparse.Namespace = parser.parse_args()

    return ExperimentConfig(
        protocol=args.protocol,
        search_mode=args.search_mode,
        grid_profile=args.grid_profile,
        scoring=args.scoring,
        holdout_fraction=args.holdout_fraction,
        holdout_repeats=args.holdout_repeats,
        inner_folds=args.inner_folds,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        random_search_n_iter=args.random_search_n_iter,
        generate_plots=not args.no_plots,
        output_prefix=args.output_prefix,
        smoke=args.smoke,
        max_samples=args.max_samples,
        count_weighted_delay=not args.unweighted_delay,
    )


if __name__ == "__main__":
    results_df: pd.DataFrame = run_experiment(parse_args())

    summary: pd.DataFrame = (
        results_df.loc[results_df["Valid"]].groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    )

    print("\nFinal summary:")
    print(summary.round(4))
