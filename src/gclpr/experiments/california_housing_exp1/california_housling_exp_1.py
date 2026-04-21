"""
Shared utilities for California Housing Experiment 1.

This module contains the reusable pieces behind the Experiment 1 runner:

- a Haversine-based geospatial kernel factory,
- a scikit-learn compatible wrapper around ``rsklpr.Rsklpr``,
- search helpers for the surviving model family,
- plotting helpers for the paper figures.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, TypeAlias, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rsklpr.kernels import laplacian_normalized_metric
from rsklpr.rsklpr import Rsklpr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor

KernelFn: TypeAlias = Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]

SearchCV: TypeAlias = GridSearchCV | RandomizedSearchCV

_lat_idx: int = 2
_lon_idx: int = 3


def haversine_rbf_kernel_factory(
    length_scale_km: float,
    radius_km: float = 6371.0088,
    lat_idx: int = _lat_idx,
    lon_idx: int = _lon_idx,
) -> KernelFn:
    """
    Build an RBF kernel based on Haversine distance.

    Args:
        length_scale_km: Kernel length scale in kilometers.
        radius_km: Earth radius in kilometers.
        lat_idx: Column index for latitude in the feature matrix.
        lon_idx: Column index for longitude in the feature matrix.

    Returns:
        A kernel function compatible with the ``Rsklpr`` ``kp`` interface.

    Raises:
        ValueError: If ``length_scale_km`` is not positive.
    """
    if length_scale_km <= 0:
        raise ValueError("length_scale_km must be positive")

    length_scale_sq: float = float(length_scale_km) ** 2

    def _to_radians(values: np.ndarray) -> np.ndarray:
        return np.asarray(np.deg2rad(values), dtype=float)

    def _haversine(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
        lat0: float = float(_to_radians(source[0, lat_idx]))
        lon0: float = float(_to_radians(source[0, lon_idx]))
        lat1: np.ndarray = _to_radians(targets[:, lat_idx])
        lon1: np.ndarray = _to_radians(targets[:, lon_idx])

        dlat: np.ndarray = lat1 - lat0
        dlon: np.ndarray = lon1 - lon0

        h: np.ndarray = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2.0) ** 2

        return np.asarray(
            2.0 * radius_km * np.arcsin(np.minimum(1.0, np.sqrt(h))),
            dtype=float,
        )

    def _kernel(
        x_0: np.ndarray,
        x_neighbors: np.ndarray,
        _: np.ndarray,
        __: int,
        ___: np.ndarray,
    ) -> np.ndarray:
        distances: np.ndarray = _haversine(source=x_0, targets=x_neighbors)
        return np.asarray(np.exp(-0.5 * (distances**2) / length_scale_sq), dtype=float)

    return lambda x0, xn, dists, index, indices: _kernel(x0, xn, dists, index, indices).reshape(1, -1)


class RsklprWrapper(BaseEstimator, RegressorMixin):
    """Wrap ``Rsklpr`` in a scikit-learn compatible regressor API."""

    def __init__(
        self,
        size_neighborhood: int = 50,
        degree: int = 1,
        kp: List[KernelFn] | KernelFn | None = None,
        kr: str = "none",
        metric_x: str = "minkowski",
        metric_x_params: Dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the wrapper.

        Args:
            size_neighborhood: Number of neighbors used by ``Rsklpr``.
            degree: Local polynomial degree.
            kp: Predictor kernel or list of predictor kernels.
            kr: Response kernel mode.
            metric_x: Distance metric used for neighbor search.
            metric_x_params: Optional parameters for ``metric_x``.
        """
        self.size_neighborhood = size_neighborhood
        self.degree = degree
        self.kp = kp if kp is not None else [laplacian_normalized_metric]
        self.kr = kr
        self.metric_x = metric_x
        self.metric_x_params = metric_x_params

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RsklprWrapper":
        """
        Fit the wrapped ``Rsklpr`` model.

        Args:
            x: Training features.
            y: Training targets.

        Returns:
            The fitted wrapper instance.
        """
        self.model_ = Rsklpr(
            size_neighborhood=self.size_neighborhood,
            degree=self.degree,
            kp=self.kp,
            kr=self.kr,
            metric_x=self.metric_x,
            metric_x_params=self.metric_x_params,
            suppress_warnings=True,
        )
        self.model_.fit(x=x, y=y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict with the wrapped ``Rsklpr`` model.

        Args:
            x: Features to score.

        Returns:
            The predicted response values.
        """
        return np.asarray(self.model_.predict(x=x), dtype=float)


def _search_space_size(param_grid: Any) -> int:
    """
    Count the size of a discrete scikit-learn parameter grid.

    Args:
        param_grid: Parameter grid accepted by scikit-learn search classes.

    Returns:
        The number of parameter combinations if the grid is discrete, else ``0``.
    """
    try:
        return len(ParameterGrid(param_grid))
    except Exception:
        return 0


def _run_search_cv(
    model: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Any,
    cv: int,
    n_jobs: int,
    verbose: int = 0,
    search_mode: str = "grid",
    n_iter: int | None = None,
    random_state: int = 42,
    scoring: Any = "neg_mean_squared_error",
) -> SearchCV:
    """Run either grid search or randomized search for a model.

    Args:
        model: Estimator to tune.
        x_train: Training features.
        y_train: Training targets.
        param_grid: Search space accepted by scikit-learn.
        cv: Number of cross-validation folds.
        n_jobs: Parallel job count.
        verbose: Search verbosity.
        search_mode: Either ``"grid"`` or ``"random"``.
        n_iter: Number of randomized-search samples when ``search_mode="random"``.
        random_state: Random seed for randomized search.
        scoring: Scoring object or scoring string.

    Returns:
        The fitted search object.

    Raises:
        ValueError: If ``search_mode`` is unsupported.
    """
    search: SearchCV

    if search_mode == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=verbose,
        )
    elif search_mode == "random":
        total_candidates: int = _search_space_size(param_grid)

        effective_n_iter: int = n_iter if n_iter is not None else min(24, total_candidates or 24)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=effective_n_iter,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown search_mode={search_mode!r}. Expected 'grid' or 'random'.")

    search.fit(x_train, y_train)
    return search


def cv_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 0,
    search_mode: str = "grid",
    n_iter: int | None = None,
    random_state: int = 42,
    scoring: Any = "neg_mean_squared_error",
) -> SearchCV:
    """
    une a KNN regressor by cross-validation.

    Args:
        x_train: Training features.
        y_train: Training targets.
        param_grid: Hyperparameter grid.
        cv: Number of cross-validation folds.
        n_jobs: Parallel job count.
        verbose: Search verbosity.
        search_mode: Either ``"grid"`` or ``"random"``.
        n_iter: Randomized-search budget when enabled.
        random_state: Random seed for randomized search.
        scoring: Scoring object or scoring string.

    Returns:
        The fitted search object.
    """
    print("🔎 Grid searching K-Nearest Neighbors... (CV)")

    return _run_search_cv(
        model=KNeighborsRegressor(),
        x_train=x_train,
        y_train=y_train,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        search_mode=search_mode,
        n_iter=n_iter,
        random_state=random_state,
        scoring=scoring,
    )


def cv_rsklpr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Any,
    cv: int,
    n_jobs: int,
    verbose: int = 0,
    search_mode: str = "grid",
    n_iter: int | None = None,
    random_state: int = 42,
    scoring: Any = "neg_mean_squared_error",
) -> SearchCV:
    """Tune a standard LPR or RSKLPR model by cross-validation.

    Args:
        x_train: Training features.
        y_train: Training targets.
        param_grid: Hyperparameter grid.
        cv: Number of cross-validation folds.
        n_jobs: Parallel job count.
        verbose: Search verbosity.
        search_mode: Either ``"grid"`` or ``"random"``.
        n_iter: Randomized-search budget when enabled.
        random_state: Random seed for randomized search.
        scoring: Scoring object or scoring string.

    Returns:
        The fitted search object.
    """
    print("🔎 Grid searching LPR (Standard)... (CV)")

    return _run_search_cv(
        model=RsklprWrapper(),
        x_train=x_train,
        y_train=y_train,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        search_mode=search_mode,
        n_iter=n_iter,
        random_state=random_state,
        scoring=scoring,
    )


def cv_gclpr_geo(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Any,
    cv: int,
    n_jobs: int,
    verbose: int = 0,
    search_mode: str = "grid",
    n_iter: int | None = None,
    random_state: int = 42,
    scoring: Any = "neg_mean_squared_error",
) -> SearchCV:
    """Tune a geospatial GC-LPR or GRC-LPR model by cross-validation.

    Args:
        x_train: Training features.
        y_train: Training targets.
        param_grid: Hyperparameter grid.
        cv: Number of cross-validation folds.
        n_jobs: Parallel job count.
        verbose: Search verbosity.
        search_mode: Either ``"grid"`` or ``"random"``.
        n_iter: Randomized-search budget when enabled.
        random_state: Random seed for randomized search.
        scoring: Scoring object or scoring string.

    Returns:
        The fitted search object.
    """
    print("🔎 Grid searching GRC-LPR (Geospatial)... (CV)")

    return _run_search_cv(
        model=RsklprWrapper(),
        x_train=x_train,
        y_train=y_train,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        search_mode=search_mode,
        n_iter=n_iter,
        random_state=random_state,
        scoring=scoring,
    )


def plot_prediction_scatter_grid(
    predictions: Dict[str, np.ndarray],
    results: Dict[str, Dict[str, float]],
    y_test: np.ndarray,
    save_plots: bool = False,
    output_dir: Path | None = None,
) -> None:
    """
    Plot a grid of actual-vs-predicted scatter plots for Experiment 1.

    Args:
        predictions: Predicted values keyed by model identifier.
        results: Metric summary keyed by model identifier.
        y_test: True target values aligned with the prediction arrays.
        save_plots: Whether to save the figure to disk.
        output_dir: Optional output directory for saved figures.
    """
    print("📈 Plotting scatter results...")
    num_models: int = len(predictions)
    ncols: int = 3
    nrows: int = int(np.ceil(num_models / ncols))
    fig: Figure
    axes: Any

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    fig.suptitle("Comparison of Model Predictions: Actual vs. Predicted", fontsize=18)
    axes_flat: List[Axes]

    if num_models == 1:
        axes_flat = [cast(Axes, axes)]
    else:
        axes_flat = cast(List[Axes], np.asarray(axes, dtype=object).ravel().tolist())

    plot_min: float = float(y_test.min())
    plot_max: float = float(np.percentile(y_test, 99.5))

    name_to_title: Dict[str, str] = {
        "knn": "K-Nearest Neighbors",
        "lpr": "LPR",
        "rsklpr": "RSKLPR (Standard)",
        "gclpr_geo": "GC-LPR (Geospatial)",
        "grclpr_geo": "GRC-LPR (Geospatial)",
    }
    index: int
    name: str

    for index, name in enumerate(predictions.keys()):
        ax: Axes = axes_flat[index]
        y_pred: np.ndarray = predictions[name]
        mask: np.ndarray = np.isfinite(y_pred)
        y_test_filtered: np.ndarray = y_test[mask]
        y_pred_filtered: np.ndarray = y_pred[mask]

        ax.scatter(y_test_filtered, y_pred_filtered, alpha=0.1, s=10)
        ax.plot([plot_min, plot_max], [plot_min, plot_max], "r--", lw=2)
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect("equal", "box")
        ax.set_title(f"{name_to_title[name]}\n" f"RMSE: {results[name]['RMSE']:.4f} | R²: {results[name]['R²']:.4f}")

        if index >= (num_models - ncols):
            ax.set_xlabel("Actual Value")
        if index % ncols == 0:
            ax.set_ylabel("Predicted Value")

    for index in range(num_models, len(axes_flat)):
        axes_flat[index].axis("off")

    if save_plots:
        target_dir: Path = output_dir or Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / "california_housling_exp_1_scatter_grid.png")
    else:
        plt.show()

    plt.close(fig)


def plot_geospatial_error_maps(
    predictions: Dict[str, np.ndarray],
    results: Dict[str, Dict[str, float]],
    x_test: np.ndarray,
    y_test: np.ndarray,
    save_plots: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Plot the 2x2 geospatial error-map figure for Experiment 1.

    Args:
        predictions: Predicted values keyed by model identifier.
        results: Metric summary keyed by model identifier.
        x_test: Feature matrix containing longitude and latitude columns.
        y_test: True target values aligned with the prediction arrays.
        save_plots: Whether to save the figure to disk.
        output_dir: Optional output directory for saved figures.
    """
    print("🗺️ Plotting geospatial error maps...")
    models_to_plot: list[str] = ["lpr", "rsklpr", "gclpr_geo", "grclpr_geo"]

    if not all(model_name in predictions for model_name in models_to_plot):
        print("Warning: Skipping geospatial plot. Missing one or more required " f"models: {models_to_plot}")
        return

    error_lpr: np.ndarray = y_test - predictions["lpr"]
    error_rsklpr: np.ndarray = y_test - predictions["rsklpr"]
    error_gclpr: np.ndarray = y_test - predictions["gclpr_geo"]
    error_grclpr: np.ndarray = y_test - predictions["grclpr_geo"]

    mask_lpr: np.ndarray = np.isfinite(error_lpr)
    mask_rsklpr: np.ndarray = np.isfinite(error_rsklpr)
    mask_gclpr: np.ndarray = np.isfinite(error_gclpr)
    mask_grclpr: np.ndarray = np.isfinite(error_grclpr)

    test_lon: np.ndarray = x_test[:, _lon_idx]
    test_lat: np.ndarray = x_test[:, _lat_idx]

    all_errors: np.ndarray = np.concatenate(
        [
            error_lpr[mask_lpr],
            error_rsklpr[mask_rsklpr],
            error_gclpr[mask_gclpr],
            error_grclpr[mask_grclpr],
        ]
    )

    vmax: float = float(np.nanpercentile(np.abs(all_errors), 99))
    vmin: float = -vmax
    fig: Figure
    ax: Axes

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    fig.suptitle(
        "Geospatial Analysis of Model Performance (Test Set Errors)",
        fontsize=18,
    )
    axes_flat: List[Axes] = list(axes.flatten())

    panels = [
        ("lpr", error_lpr, mask_lpr, "LPR Errors"),
        ("rsklpr", error_rsklpr, mask_rsklpr, "RSKLPR Errors"),
        ("gclpr_geo", error_gclpr, mask_gclpr, "GC-LPR Errors"),
        ("grclpr_geo", error_grclpr, mask_grclpr, "GRC-LPR Errors"),
    ]

    for ax, (name, errors, mask, title) in zip(axes_flat, panels):
        scatter = ax.scatter(
            test_lon[mask],
            test_lat[mask],
            c=errors[mask],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=5,
            alpha=0.5,
        )

        ax.set_title(f"{title} (RMSE: {results[name]['RMSE']:.3f})")
        fig.colorbar(scatter, ax=ax, label="Error (Actual - Pred)")

    axes_flat[2].set_xlabel("Longitude")
    axes_flat[3].set_xlabel("Longitude")
    axes_flat[0].set_ylabel("Latitude")
    axes_flat[2].set_ylabel("Latitude")

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    if save_plots:
        target_dir = output_dir or Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / "california_housling_exp_1_geospatial_errors.png")
    else:
        plt.show()

    plt.close(fig)
