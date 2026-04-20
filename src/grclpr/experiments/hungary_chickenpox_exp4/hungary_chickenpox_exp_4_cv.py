"""Experiment 4 runner for the Hungary chickenpox graph-signal study."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rsklpr.kernels import tricube_normalized_metric
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from grclpr.experiments.hungary_chickenpox_exp4.hungary_chickenpox import (
    build_sample_idx_to_node_id_map,
    cv_graph,
    cv_knn,
    cv_rsklpr,
    load_chickenpox_data_and_graph,
    plot_graph_error_maps,
    plot_prediction_scatter_grid,
    scale_features,
)

_repo_dir: Path = Path(__file__).resolve().parents[4]
_raw_json_path: Path = _repo_dir / "data" / "hungary_chickenpox" / "chickenpox.json"
_out_dir: Path = _repo_dir / "out" / "experiment_4"

_model_order: list[str] = [
    "knn",
    "lpr",
    "rsklpr",
    "gclpr_graph",
    "grclpr_graph",
]

_model_labels: dict[str, str] = {
    "knn": "KNN",
    "lpr": "LPR",
    "rsklpr": "RSKLPR",
    "gclpr_graph": "GC-LPR (Graph)",
    "grclpr_graph": "GRC-LPR (Graph)",
}

_search_dispatch: dict[str, Callable[..., Any]] = {
    "knn": cv_knn,
    "lpr": cv_rsklpr,
    "rsklpr": cv_rsklpr,
    "gclpr_graph": cv_graph,
    "grclpr_graph": cv_graph,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for Hungary chickenpox Experiment 4."""

    protocol: str = "rolling_origin_cv"
    search_mode: str = "grid"
    grid_profile: str = "full"
    scoring: str = "strict_rmse"
    outer_splits: int = 5
    inner_folds: int = 4
    lags: int = 4
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0
    random_search_n_iter: int | None = None
    generate_plots: bool = True
    output_prefix: str = "hungary_chickenpox_exp4"
    smoke: bool = False
    max_snapshots: int | None = None


def load_raw_data(
    lags: int = 4,
    max_snapshots: int | None = None,
) -> tuple[nx.Graph, np.ndarray, np.ndarray, pd.DataFrame, dict[int, int], list[str]]:
    """Load and prepare the Hungary chickenpox dataset for Experiment 4.

    Args:
        lags: Number of lagged weekly counts used as predictors.
        max_snapshots: Optional cap on prediction snapshots after lag
            construction.

    Returns:
        The county graph, feature matrix, target vector, flattened sample
        metadata, row-index to county mapping, and ordered feature names.
    """
    county_graph, x, y, metadata = load_chickenpox_data_and_graph(
        raw_json_path=_raw_json_path,
        lags=lags,
        max_snapshots=max_snapshots,
    )
    sample_idx_to_node_id = build_sample_idx_to_node_id_map(metadata=metadata)
    feature_names = [f"lag_{lags - offset}" for offset in range(lags)]
    return county_graph, x, y, metadata, sample_idx_to_node_id, feature_names


def build_search_grids(grid_profile: str = "full") -> dict[str, Any]:
    """Build the hyperparameter search spaces for Experiment 4.

    Args:
        grid_profile: One of ``"full"``, ``"reduced"``, or ``"smoke"``.

    Returns:
        A mapping from model identifier to scikit-learn compatible search spaces.

    Raises:
        ValueError: If ``grid_profile`` is unsupported.
    """
    feature_kernel = tricube_normalized_metric

    graph_param_smoke = [1.0]
    graph_param_reduced = [0.5, 1.0, 1.5, 2.0]
    graph_param_full = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0]
    graph_neighborhood_smoke = [21]
    graph_neighborhood_reduced = [21, 31, 41, 61, 81]
    graph_neighborhood_full = [121, 141, 161, 181, 221, 281, 361]

    if grid_profile == "smoke":
        return {
            "knn": {
                "n_neighbors": [3],
                "weights": ["distance"],
                "p": [1],
                "leaf_size": [30],
            },
            "lpr": [
                {
                    "size_neighborhood": [11],
                    "degree": [1],
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "rsklpr": [
                {
                    "size_neighborhood": [11],
                    "degree": [1],
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["joint"],
                }
            ],
            "gclpr_graph": [
                {
                    "size_neighborhood": graph_neighborhood_smoke,
                    "degree": [1],
                    "distance_scale": graph_param_smoke,
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "grclpr_graph": [
                {
                    "size_neighborhood": graph_neighborhood_smoke,
                    "degree": [1],
                    "distance_scale": graph_param_smoke,
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["conden"],
                }
            ],
        }

    if grid_profile == "reduced":
        return {
            "knn": {
                "n_neighbors": [3, 5, 7, 11, 21],
                "weights": ["distance"],
                "p": [1, 2],
                "leaf_size": [30],
            },
            "lpr": [
                {
                    "size_neighborhood": [11, 21, 31, 41, 61],
                    "degree": [1],
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "rsklpr": [
                {
                    "size_neighborhood": [11, 21, 31, 41, 61],
                    "degree": [1],
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["joint"],
                }
            ],
            "gclpr_graph": [
                {
                    "size_neighborhood": graph_neighborhood_reduced,
                    "degree": [1],
                    "distance_scale": graph_param_reduced,
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "grclpr_graph": [
                {
                    "size_neighborhood": graph_neighborhood_reduced,
                    "degree": [1],
                    "distance_scale": graph_param_reduced,
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["conden"],
                }
            ],
        }

    if grid_profile == "full":
        return {
            "knn": {
                "n_neighbors": [11, 15, 21, 31, 41, 51, 61],
                "weights": ["distance"],
                "p": [1, 2],
                "leaf_size": [30],
            },
            "lpr": [
                {
                    "size_neighborhood": [41, 51, 61, 71, 81, 101, 121],
                    "degree": [1],
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "rsklpr": [
                {
                    "size_neighborhood": [41, 51, 61, 71, 81, 101, 121],
                    "degree": [1],
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["joint"],
                }
            ],
            "gclpr_graph": [
                {
                    "size_neighborhood": graph_neighborhood_full,
                    "degree": [1],
                    "distance_scale": graph_param_full,
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["none"],
                }
            ],
            "grclpr_graph": [
                {
                    "size_neighborhood": graph_neighborhood_full,
                    "degree": [1],
                    "distance_scale": graph_param_full,
                    "kp": [feature_kernel],
                    "metric_x": ["minkowski"],
                    "metric_x_params": [{"p": 1}],
                    "kr": ["conden"],
                }
            ],
        }

    raise ValueError(f"Unsupported grid_profile={grid_profile!r}")


def strict_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE, treating any non-finite prediction as invalid.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        The RMSE if all predictions are finite, else ``inf``.
    """
    if not np.all(np.isfinite(y_pred)):
        return float("inf")

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for a valid prediction vector.

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


def prediction_validity(y_pred: np.ndarray) -> tuple[bool, float]:
    """Determine whether a prediction vector is fully finite.

    Args:
        y_pred: Predicted values.

    Returns:
        A boolean validity flag and the fraction of non-finite predictions.
    """
    finite_mask = np.isfinite(y_pred)
    non_finite_fraction = 1.0 - float(np.mean(finite_mask))
    return bool(np.all(finite_mask)), non_finite_fraction


def resolve_scoring(scoring: str) -> Any:
    """Resolve a configured scoring string to a scikit-learn scoring object.

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
    """Return whether a model uses the county-graph context kernel.

    Args:
        model_key: Internal model identifier.

    Returns:
        ``True`` when the model includes the county-graph context kernel.
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


def build_time_series_sample_splits(
    sample_time_indices: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build rolling-origin train/test splits over flattened sample rows.

    Args:
        sample_time_indices: Time index for each flattened sample row.
        n_splits: Number of rolling-origin splits.

    Returns:
        Sample-row train/test splits compatible with scikit-learn search CV.

    Raises:
        ValueError: If there are not enough unique time steps.
    """
    unique_times = np.sort(np.unique(sample_time_indices))
    if len(unique_times) <= n_splits:
        raise ValueError("n_splits must be smaller than the number of unique forecast times.")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_time_positions, test_time_positions in splitter.split(unique_times):
        train_times = unique_times[train_time_positions]
        test_times = unique_times[test_time_positions]
        train_idx = np.flatnonzero(np.isin(sample_time_indices, train_times))
        test_idx = np.flatnonzero(np.isin(sample_time_indices, test_times))
        splits.append((train_idx, test_idx))

    return splits


def run_search(
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: np.ndarray,
    train_time_indices: np.ndarray,
    county_graph: nx.Graph,
    sample_idx_to_node_id: Mapping[int, int],
    search_grids: dict[str, Any],
    config: ExperimentConfig,
) -> Any:
    """Dispatch model search to the appropriate helper.

    Args:
        model_key: Internal model identifier.
        x_train: Scaled training feature matrix.
        y_train: Training targets.
        train_indices: Original row indices for ``x_train``.
        train_time_indices: Forecast-time index for each training row.
        county_graph: County graph.
        sample_idx_to_node_id: Global row-index to county-node mapping.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        The fitted search object.
    """
    search_fn = _search_dispatch[model_key]
    scoring = resolve_scoring(scoring=config.scoring)
    inner_splits = build_time_series_sample_splits(
        sample_time_indices=train_time_indices,
        n_splits=config.inner_folds,
    )

    if uses_graph_context(model_key):
        return search_fn(
            x_train=x_train,
            y_train=y_train,
            train_indices=train_indices,
            county_graph=county_graph,
            sample_idx_to_node_id=dict(sample_idx_to_node_id),
            param_grid=search_grids[model_key],
            cv=inner_splits,
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
        cv=inner_splits,
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        search_mode=config.search_mode,
        n_iter=config.random_search_n_iter,
        random_state=config.random_state,
        scoring=scoring,
    )


def evaluate_rolling_origin_cv(
    x: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    county_graph: nx.Graph,
    sample_idx_to_node_id: Mapping[int, int],
    search_grids: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Evaluate all models on rolling-origin temporal splits.

    Args:
        x: Full feature matrix.
        y: Full target vector.
        metadata: Full sample metadata.
        county_graph: County graph.
        sample_idx_to_node_id: Global row-index to county-node mapping.
        search_grids: Search spaces keyed by model identifier.
        config: Experiment configuration.

    Returns:
        Result rows and aggregated full-length predictions on rows that appeared
        in at least one held-out split.
    """
    fold_records: list[dict[str, Any]] = []
    sample_time_indices = metadata["time_idx"].to_numpy(dtype=int)
    outer_splits = build_time_series_sample_splits(
        sample_time_indices=sample_time_indices,
        n_splits=config.outer_splits,
    )

    prediction_sums = {key: np.zeros_like(y, dtype=float) for key in _model_order}
    prediction_counts = {key: np.zeros_like(y, dtype=int) for key in _model_order}

    for fold_i, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        print(
            "\n--- Rolling-origin evaluation "
            f"{fold_i}/{config.outer_splits} "
            f"(train_times={sample_time_indices[train_idx].min()}-{sample_time_indices[train_idx].max()}, "
            f"test_times={sample_time_indices[test_idx].min()}-{sample_time_indices[test_idx].max()}) ---"
        )

        x_train_scaled, x_test_scaled = scale_features(x_train=x[train_idx], x_test=x[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        train_time_indices = sample_time_indices[train_idx]

        for model_key in _model_order:
            label = _model_labels[model_key]
            search = run_search(
                model_key=model_key,
                x_train=x_train_scaled,
                y_train=y_train,
                train_indices=train_idx,
                train_time_indices=train_time_indices,
                county_graph=county_graph,
                sample_idx_to_node_id=sample_idx_to_node_id,
                search_grids=search_grids,
                config=config,
            )

            estimator = search.best_estimator_
            if uses_graph_context(model_key):
                x_test_input = augment_with_original_indices(x=x_test_scaled, indices=test_idx)
            else:
                x_test_input = x_test_scaled

            y_pred = np.asarray(estimator.predict(x_test_input), dtype=float).ravel()
            finite_mask = np.isfinite(y_pred)
            prediction_sums[model_key][test_idx[finite_mask]] += y_pred[finite_mask]
            prediction_counts[model_key][test_idx[finite_mask]] += 1

            valid, non_finite_fraction = prediction_validity(y_pred=y_pred)
            if valid:
                metrics = compute_metrics(y_test, y_pred)
            else:
                metrics = {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}

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
                    "Train_Time_Max": int(sample_time_indices[train_idx].max()),
                    "Test_Time_Min": int(sample_time_indices[test_idx].min()),
                    "Test_Time_Max": int(sample_time_indices[test_idx].max()),
                    "Selected_Params": str(search.best_params_),
                }
            )

            if valid:
                print(f"  > {label}: RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f}")
            else:
                print(f"  > {label}: invalid predictions ({100.0 * non_finite_fraction:.1f}% non-finite)")

    full_predictions: dict[str, np.ndarray] = {}
    for model_key in _model_order:
        averaged = np.full_like(y, np.nan, dtype=float)
        nonzero_mask = prediction_counts[model_key] > 0
        averaged[nonzero_mask] = prediction_sums[model_key][nonzero_mask] / prediction_counts[model_key][nonzero_mask]
        full_predictions[model_key] = averaged

    return fold_records, full_predictions


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

    ax.set_title("Experiment 4: Rolling-Origin RMSE Distribution")
    ax.set_ylabel("RMSE")
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{output_prefix}_rmse_boxplot.png", dpi=200)
    plt.close(fig)


def save_outputs(
    county_graph: nx.Graph,
    metadata: pd.DataFrame,
    y: np.ndarray,
    fold_records: list[dict[str, Any]],
    full_predictions: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> None:
    """Persist result tables, metadata, and figures.

    Args:
        county_graph: County graph used in the experiment.
        metadata: Full sample metadata.
        y: Full target vector.
        fold_records: Fold-level metric rows.
        full_predictions: Reconstructed full-length predictions keyed by model.
        config: Experiment configuration.
    """
    output_dir = _out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{config.output_prefix}_raw.csv"
    summary_path = output_dir / f"{config.output_prefix}_summary.csv"
    metadata_path = output_dir / f"{config.output_prefix}_metadata.json"

    df_results = pd.DataFrame(fold_records)
    df_results.to_csv(raw_path, index=False)

    valid_results = df_results.loc[df_results["Valid"]].copy() if "Valid" in df_results.columns else df_results.copy()
    summary = valid_results.groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    summary.to_csv(summary_path)

    metadata_payload = {
        "config": asdict(config),
        "metrics_reported": ["RMSE", "MAE", "R2"],
        "primary_metric": "RMSE",
        "models": [_model_labels[key] for key in _model_order],
        "reporting_target_space": "weekly chickenpox case counts",
        "reporting_uses_valid_runs_only": True,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))

    if not config.generate_plots:
        return

    evaluated_mask = np.zeros_like(y, dtype=bool)
    for y_pred_full in full_predictions.values():
        evaluated_mask |= np.isfinite(y_pred_full)

    if not np.any(evaluated_mask):
        return

    y_plot = y[evaluated_mask]
    metadata_plot = metadata.loc[evaluated_mask].reset_index(drop=True)
    predictions_plot = {
        key: np.asarray(y_pred_full[evaluated_mask], dtype=float) for key, y_pred_full in full_predictions.items()
    }

    agg_results_for_plots: dict[str, dict[str, float]] = {}
    for model_key, y_pred_plot in predictions_plot.items():
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

    plot_graph_error_maps(
        graph=county_graph,
        metadata=metadata_plot,
        y_true=y_plot,
        predictions=predictions_plot,
        save_plot=True,
        output_dir=output_dir,
    )

    save_rmse_fold_plot(
        df_results=valid_results,
        output_dir=output_dir,
        output_prefix=config.output_prefix,
    )


def normalize_config(config: ExperimentConfig) -> ExperimentConfig:
    """Normalize and validate the experiment configuration.

    Args:
        config: Raw experiment configuration.

    Returns:
        The normalized configuration.

    Raises:
        ValueError: If a configuration field is unsupported or invalid.
    """
    if config.protocol != "rolling_origin_cv":
        raise ValueError("protocol must be 'rolling_origin_cv'")

    if config.search_mode not in {"grid", "random"}:
        raise ValueError("search_mode must be 'grid' or 'random'")

    if config.outer_splits < 2:
        raise ValueError("outer_splits must be at least 2")

    if config.inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")

    if config.lags < 1:
        raise ValueError("lags must be at least 1")

    if config.smoke:
        config = replace(
            config,
            grid_profile="smoke",
            outer_splits=3,
            inner_folds=2,
            lags=min(config.lags, 3),
            generate_plots=False,
            max_snapshots=config.max_snapshots or 24,
            output_prefix=f"{config.output_prefix}_smoke",
        )

    resolve_scoring(config.scoring)
    return config


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run Hungary chickenpox Experiment 4.

    Args:
        config: Experiment configuration.

    Returns:
        Fold-level results as a DataFrame.
    """
    config = normalize_config(config)
    county_graph, x, y, metadata, sample_idx_to_node_id, feature_names = load_raw_data(
        lags=config.lags,
        max_snapshots=config.max_snapshots,
    )
    del feature_names

    search_grids = build_search_grids(grid_profile=config.grid_profile)
    fold_records, full_predictions = evaluate_rolling_origin_cv(
        x=x,
        y=y,
        metadata=metadata,
        county_graph=county_graph,
        sample_idx_to_node_id=sample_idx_to_node_id,
        search_grids=search_grids,
        config=config,
    )

    save_outputs(
        county_graph=county_graph,
        metadata=metadata,
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
        description="Run the Hungary chickenpox Experiment 4 with rolling-origin evaluation.",
    )
    parser.add_argument(
        "--protocol",
        choices=["rolling_origin_cv"],
        default="rolling_origin_cv",
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
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--lags", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--random-search-n-iter", type=int, default=None)
    parser.add_argument("--output-prefix", type=str, default="hungary_chickenpox_exp4")
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    return ExperimentConfig(
        protocol=args.protocol,
        search_mode=args.search_mode,
        grid_profile=args.grid_profile,
        scoring=args.scoring,
        outer_splits=args.outer_splits,
        inner_folds=args.inner_folds,
        lags=args.lags,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        random_search_n_iter=args.random_search_n_iter,
        generate_plots=not args.no_plots,
        output_prefix=args.output_prefix,
        smoke=args.smoke,
        max_snapshots=args.max_snapshots,
    )


if __name__ == "__main__":
    results_df = run_experiment(parse_args())

    summary = results_df.loc[results_df["Valid"]].groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])

    print("\nFinal summary:")
    print(summary.round(4))
