"""Shared utilities for NYC Airbnb Experiment 2."""

import math
import warnings
from pathlib import Path
from typing import Any, Callable, TypeAlias, cast

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rsklpr.kernels import laplacian_normalized_metric
from rsklpr.rsklpr import Rsklpr
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor

KernelFn: TypeAlias = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, np.ndarray],
    np.ndarray,
]
SearchCV: TypeAlias = GridSearchCV | RandomizedSearchCV

_latitude_column = "latitude"
_longitude_column = "longitude"
_epsilon_similarity = 0.01


def load_subway_graph(graph_path: Path) -> nx.Graph:
    """Load the NYC subway graph from GraphML.

    Args:
        graph_path: GraphML path for the subway network.

    Returns:
        The loaded subway graph.
    """
    print("🚇 Loading subway graph...")
    return nx.read_graphml(graph_path)


def haversine(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
    radius_km: float = 6371.0088,
) -> np.ndarray:
    """Compute Haversine distances in kilometers.

    Args:
        lat1: Latitude of the first point or points.
        lon1: Longitude of the first point or points.
        lat2: Latitude of the second point or points.
        lon2: Longitude of the second point or points.
        radius_km: Sphere radius in kilometers.

    Returns:
        The Haversine distance array in kilometers.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad

    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return np.asarray(radius_km * c, dtype=float)


def calculate_unified_index_to_station_map(
    subway_graph: nx.Graph,
    listing_coords: pd.DataFrame,
    max_station_dist_km: float = 1.0,
) -> dict[int, str | None]:
    """Map each listing row index to its nearest subway station.

    Args:
        subway_graph: Subway graph with node latitude and longitude attributes.
        listing_coords: Listing coordinates with ``latitude`` and ``longitude`` columns.
        max_station_dist_km: Maximum allowed listing-to-station distance.

    Returns:
        A mapping from row index to the nearest station node ID, or ``None``.
    """
    station_data: list[tuple[float, float, str]] = [
        (
            float(data["lat"]),
            float(data["lon"]),
            str(station_id),
        )
        for station_id, data in subway_graph.nodes(data=True)
        if "lat" in data and "lon" in data
    ]

    station_coords = np.array([[lat, lon] for lat, lon, _ in station_data], dtype=float)
    station_ids = np.array([station_id for _, _, station_id in station_data], dtype=object)

    print(f"Extracted coordinates for {len(station_ids)} stations.")
    print("Pre-calculating unified index-to-station map for all listings...")

    station_tree = KDTree(station_coords)
    original_idx_to_station_map: dict[int, str | None] = {}
    approx_radius_degrees = max_station_dist_km / 111.0
    listing_values = listing_coords[[_latitude_column, _longitude_column]].to_numpy(copy=True)
    nearby_station_indices = station_tree.query_ball_point(
        listing_values,
        r=approx_radius_degrees,
    )

    for row_index, listing_coord in enumerate(listing_values):
        candidate_indices = np.asarray(nearby_station_indices[row_index], dtype=int)
        nearest_station_id: str | None = None

        if candidate_indices.size > 0:
            candidate_coords = station_coords[candidate_indices]
            distances_km = haversine(
                listing_coord[0],
                listing_coord[1],
                candidate_coords[:, 0],
                candidate_coords[:, 1],
            )
            closest_candidate = int(np.argmin(distances_km))
            if float(distances_km[closest_candidate]) <= max_station_dist_km:
                nearest_station_id = cast(str, station_ids[candidate_indices[closest_candidate]])

        original_idx_to_station_map[row_index] = nearest_station_id

    print("Unified station map created.")
    return original_idx_to_station_map


def subway_shortest_path_kernel_factory(
    graph: nx.Graph,
    original_idx_to_station_map: dict[int, str | None],
    train_set_original_indices: np.ndarray,
    predict_set_original_indices: np.ndarray,
    distance_scale: float = 1.0,
) -> KernelFn:
    """Build the subway shortest-path similarity kernel.

    Args:
        graph: Subway graph.
        original_idx_to_station_map: Mapping from dataset row index to station ID.
        train_set_original_indices: Original row indices for the fitted data.
        predict_set_original_indices: Original row indices for the prediction data.
        distance_scale: Exponential decay scale for graph distance.

    Returns:
        A kernel compatible with ``Rsklpr``.

    Raises:
        TypeError: If the train or predict indices are not 1D NumPy arrays.
        ValueError: If ``distance_scale`` is not positive.
    """
    if distance_scale <= 0:
        raise ValueError("distance_scale must be positive")

    if not isinstance(train_set_original_indices, np.ndarray) or train_set_original_indices.ndim != 1:
        raise TypeError("train_set_original_indices must be a 1D NumPy array")

    if not isinstance(predict_set_original_indices, np.ndarray) or predict_set_original_indices.ndim != 1:
        raise TypeError("predict_set_original_indices must be a 1D NumPy array")

    def _kernel(
        x_0: np.ndarray,
        x_neighbors: np.ndarray,
        dist_x_neighbors: np.ndarray,
        index_x_0: int,
        indices_neighbors: np.ndarray,
    ) -> np.ndarray:
        del x_0, x_neighbors, dist_x_neighbors

        similarity_scores = np.full(
            np.asarray(indices_neighbors).ravel().shape[0],
            fill_value=_epsilon_similarity,
            dtype=float,
        )

        try:
            target_original_idx = int(predict_set_original_indices[index_x_0])
        except IndexError:
            warnings.warn(
                f"index_x_0 ({index_x_0}) is out of bounds for predict_set_original_indices.",
                stacklevel=2,
            )
            return similarity_scores.reshape(1, -1)

        target_station = original_idx_to_station_map.get(target_original_idx)
        if target_station is None or target_station not in graph:
            return similarity_scores.reshape(1, -1)

        neighbor_relative_indices = np.asarray(indices_neighbors).ravel()
        for score_index, neighbor_relative_idx in enumerate(neighbor_relative_indices):
            try:
                neighbor_original_idx = int(train_set_original_indices[int(neighbor_relative_idx)])
            except IndexError:
                warnings.warn(
                    (
                        "neighbor_relative_idx "
                        f"({neighbor_relative_idx}) is out of bounds for train_set_original_indices."
                    ),
                    stacklevel=2,
                )
                continue

            neighbor_station = original_idx_to_station_map.get(neighbor_original_idx)
            if neighbor_station is None or neighbor_station not in graph:
                continue

            if neighbor_station == target_station:
                similarity_scores[score_index] = 1.0
                continue

            try:
                path_length = nx.shortest_path_length(
                    graph,
                    source=target_station,
                    target=neighbor_station,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            similarity_scores[score_index] = math.exp(-float(path_length) / distance_scale)

        return similarity_scores.reshape(1, -1)

    return _kernel


def _resolve_station_node_id(
    subway_graph: nx.Graph,
    reference_station_id: str,
) -> str | None:
    """Resolve a station display name or node ID to a concrete graph node ID."""
    for node_id, data in subway_graph.nodes(data=True):
        if data.get("name") == reference_station_id:
            return str(node_id)

    if reference_station_id in subway_graph:
        return reference_station_id

    return None


def plot_subway_kernel_similarity(
    subway_graph: nx.Graph,
    reference_station_id: str,
    distance_scale: float,
    save_plot: bool = False,
    output_dir: Path | None = None,
    zoom_graph_radius: int | None = None,
) -> None:
    """Plot shortest-path kernel similarity on the subway graph.

    Args:
        subway_graph: Full subway graph.
        reference_station_id: Station name or node ID to use as the source.
        distance_scale: Exponential decay scale for graph distance.
        save_plot: Whether to save the figure.
        output_dir: Optional output directory for the saved figure.
        zoom_graph_radius: Optional hop radius for a zoomed neighborhood plot.
    """
    print(f"🎨 Visualizing kernel similarity from '{reference_station_id}'...")
    resolved_station_id = _resolve_station_node_id(subway_graph, reference_station_id)

    if resolved_station_id is None:
        print(f"Warning: Could not resolve station '{reference_station_id}'. Skipping plot.")
        return

    if zoom_graph_radius is None:
        subgraph = subway_graph
    else:
        neighborhood = nx.single_source_shortest_path_length(
            subway_graph,
            resolved_station_id,
            cutoff=zoom_graph_radius,
        )
        subgraph = subway_graph.subgraph(neighborhood.keys()).copy()

    positions: dict[str, tuple[float, float]] = {
        str(node_id): (float(data["lon"]), float(data["lat"]))
        for node_id, data in subgraph.nodes(data=True)
        if "lon" in data and "lat" in data
    }

    similarities: list[float] = []
    nodes_to_plot: list[str] = []
    for node_id in positions:
        nodes_to_plot.append(node_id)
        try:
            path_length = nx.shortest_path_length(
                subway_graph,
                source=resolved_station_id,
                target=node_id,
            )
            similarity = math.exp(-float(path_length) / distance_scale)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            similarity = _epsilon_similarity
        similarities.append(similarity)

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(subgraph, positions, width=0.5, edge_color="lightgray", ax=ax)
    collection = nx.draw_networkx_nodes(
        subgraph,
        positions,
        nodelist=nodes_to_plot,
        node_color=similarities,
        cmap="viridis",
        node_size=20 if zoom_graph_radius is None else 60,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        subgraph,
        positions,
        nodelist=[resolved_station_id],
        node_color="red",
        edgecolors="black",
        node_size=90 if zoom_graph_radius is None else 160,
        ax=ax,
    )
    fig.colorbar(collection, ax=ax, label="Kernel Similarity")
    ax.set_title(f"Subway Kernel Similarity from {reference_station_id}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axis("equal")

    if save_plot:
        target_dir = output_dir or Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"subway_kernel_similarity_zoomed_{zoom_graph_radius}.png"
            if zoom_graph_radius is not None
            else "subway_kernel_similarity.png"
        )
        fig.savefig(
            target_dir / filename,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    else:
        plt.show()

    plt.close(fig)


class SubwayRsklprWrapper(BaseEstimator, RegressorMixin):
    """Wrap ``Rsklpr`` with an index-aware subway kernel for scikit-learn search."""

    def __init__(
        self,
        size_neighborhood: int = 50,
        degree: int = 1,
        distance_scale: float = 1.0,
        kp: KernelFn = laplacian_normalized_metric,
        kr: str = "none",
        metric_x: str = "minkowski",
        metric_x_params: dict[str, Any] | None = None,
        subway_graph: nx.Graph | None = None,
        original_idx_to_station_map: dict[int, str | None] | None = None,
    ) -> None:
        """Initialize the subway-aware wrapper.

        Args:
            size_neighborhood: Number of nearest neighbors.
            degree: Local polynomial degree.
            distance_scale: Exponential decay scale for subway distance.
            kp: Base predictor kernel applied to feature space.
            kr: Response kernel mode.
            metric_x: Distance metric for nearest-neighbor search.
            metric_x_params: Optional metric parameters.
            subway_graph: Subway graph shared across fits.
            original_idx_to_station_map: Global row-index to station mapping.
        """
        self.size_neighborhood = size_neighborhood
        self.degree = degree
        self.distance_scale = distance_scale
        self.kp = kp
        self.kr = kr
        self.metric_x = metric_x
        self.metric_x_params = metric_x_params
        self.subway_graph = subway_graph
        self.original_idx_to_station_map = original_idx_to_station_map

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SubwayRsklprWrapper":
        """Store the feature matrix and original indices for later prediction.

        Args:
            x: Training features with original row indices in the last column.
            y: Training targets.

        Returns:
            The fitted wrapper instance.

        Raises:
            ValueError: If global subway data were not provided.
        """
        if self.subway_graph is None or self.original_idx_to_station_map is None:
            raise ValueError("SubwayRsklprWrapper requires subway_graph and original_idx_to_station_map.")

        self.train_original_indices_ = x[:, -1].astype(int)
        self.x_fit_ = x[:, :-1]
        self.y_fit_ = y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict with the subway-aware ``Rsklpr`` model.

        Args:
            x: Prediction features with original row indices in the last column.

        Returns:
            Predicted responses.
        """
        predict_original_indices = x[:, -1].astype(int)
        x_predict_features = x[:, :-1]
        original_idx_to_station_map = cast(
            dict[int, str | None],
            self.original_idx_to_station_map,
        )

        subway_kernel = subway_shortest_path_kernel_factory(
            graph=self.subway_graph,
            original_idx_to_station_map=original_idx_to_station_map,
            train_set_original_indices=self.train_original_indices_,
            predict_set_original_indices=predict_original_indices,
            distance_scale=self.distance_scale,
        )

        model = Rsklpr(
            size_neighborhood=self.size_neighborhood,
            degree=self.degree,
            kp=[self.kp, subway_kernel],
            kr=self.kr,
            metric_x=self.metric_x,
            metric_x_params=self.metric_x_params,
            suppress_warnings=True,
        )
        model.fit(x=self.x_fit_, y=self.y_fit_)
        return np.asarray(model.predict(x=x_predict_features), dtype=float)


class RsklprWrapper(BaseEstimator, RegressorMixin):
    """Wrap ``Rsklpr`` in a scikit-learn compatible regressor API."""

    def __init__(
        self,
        size_neighborhood: int = 50,
        degree: int = 1,
        kp: list[KernelFn] | KernelFn | None = None,
        kr: str = "none",
        metric_x: str = "minkowski",
        metric_x_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the wrapper.

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
        """Store the fitted training data.

        Args:
            x: Training features.
            y: Training targets.

        Returns:
            The fitted wrapper instance.
        """
        self.x_fit_ = x
        self.y_fit_ = y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict with the wrapped ``Rsklpr`` model.

        Args:
            x: Features to score.

        Returns:
            Predicted responses.
        """
        model = Rsklpr(
            size_neighborhood=self.size_neighborhood,
            degree=self.degree,
            kp=self.kp,
            kr=self.kr,
            metric_x=self.metric_x,
            metric_x_params=self.metric_x_params,
            suppress_warnings=True,
        )
        model.fit(x=self.x_fit_, y=self.y_fit_)
        return np.asarray(model.predict(x=x), dtype=float)


def _search_space_size(param_grid: Any) -> int:
    """Count the size of a discrete scikit-learn parameter grid.

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
        total_candidates = _search_space_size(param_grid)
        effective_n_iter = n_iter if n_iter is not None else min(24, total_candidates or 24)
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
    param_grid: dict[str, list[Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 0,
    search_mode: str = "grid",
    n_iter: int | None = None,
    random_state: int = 42,
    scoring: Any = "neg_mean_squared_error",
) -> SearchCV:
    """Tune a KNN regressor by cross-validation."""
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
    """Tune a standard LPR or RSKLPR model by cross-validation."""
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


def cv_subway(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: np.ndarray,
    subway_graph: nx.Graph,
    original_idx_to_station_map: dict[int, str | None],
    param_grid: Any,
    cv: int,
    n_jobs: int,
    verbose: int = 0,
    search_mode: str = "grid",
    n_iter: int | None = None,
    random_state: int = 42,
    scoring: Any = "neg_mean_squared_error",
) -> SearchCV:
    """Tune a subway-context GC-LPR or GRC-LPR model by cross-validation."""
    print("🔎 Grid searching GRC-LPR (Subway)... (CV)")
    x_train_with_indices = np.hstack((x_train, train_indices.reshape(-1, 1)))
    estimator = SubwayRsklprWrapper(
        subway_graph=subway_graph,
        original_idx_to_station_map=original_idx_to_station_map,
    )
    return _run_search_cv(
        model=estimator,
        x_train=x_train_with_indices,
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
    predictions: dict[str, np.ndarray],
    results: dict[str, dict[str, float]],
    y_true: np.ndarray,
    save_plots: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Plot a grid of actual-vs-predicted scatter plots for Experiment 2.

    Args:
        predictions: Predicted values keyed by model identifier.
        results: Metric summary keyed by model identifier.
        y_true: True target values aligned with the prediction arrays.
        save_plots: Whether to save the figure.
        output_dir: Optional output directory for saved figures.
    """
    print("📈 Plotting scatter results...")
    model_count = len(predictions)
    ncols = 3
    nrows = int(np.ceil(model_count / ncols))
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
    fig.suptitle("NYC Airbnb: Actual vs. Predicted Price", fontsize=18)

    if model_count == 1:
        axes_flat = [cast(Axes, axes)]
    else:
        axes_flat = cast(list[Axes], np.asarray(axes, dtype=object).ravel().tolist())

    plot_min = 0.0
    plot_max = float(np.percentile(y_true, 99.0))
    name_to_title = {
        "knn": "K-Nearest Neighbors",
        "lpr": "LPR",
        "rsklpr": "RSKLPR (Standard)",
        "gclpr_subway": "GC-LPR (Subway)",
        "grclpr_subway": "GRC-LPR (Subway)",
    }

    for index, name in enumerate(predictions):
        ax = axes_flat[index]
        y_pred = predictions[name]
        mask = np.isfinite(y_pred)
        ax.scatter(y_true[mask], y_pred[mask], alpha=0.12, s=8)
        ax.plot([plot_min, plot_max], [plot_min, plot_max], "r--", lw=2)
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect("equal", "box")
        ax.set_title(f"{name_to_title[name]}\n" f"RMSE: {results[name]['RMSE']:.2f} | R2: {results[name]['R2']:.3f}")
        if index >= (model_count - ncols):
            ax.set_xlabel("Actual Price")
        if index % ncols == 0:
            ax.set_ylabel("Predicted Price")

    for index in range(model_count, len(axes_flat)):
        axes_flat[index].axis("off")

    if save_plots:
        target_dir = output_dir or Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / "nyc_airbnb_exp_2_scatter_grid.png")
    else:
        plt.show()

    plt.close(fig)


def plot_geospatial_error_maps(
    predictions: dict[str, np.ndarray],
    results: dict[str, dict[str, float]],
    coords: pd.DataFrame,
    y_true: np.ndarray,
    save_plots: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Plot the 2x2 NYC geospatial error-map figure.

    Args:
        predictions: Predicted values keyed by model identifier.
        results: Metric summary keyed by model identifier.
        coords: Listing coordinates with longitude and latitude columns.
        y_true: True target values aligned with the prediction arrays.
        save_plots: Whether to save the figure.
        output_dir: Optional output directory for saved figures.
    """
    print("🗺️ Plotting geospatial error maps...")
    models_to_plot = ["lpr", "rsklpr", "gclpr_subway", "grclpr_subway"]
    if not all(model_name in predictions for model_name in models_to_plot):
        print("Warning: Skipping geospatial plot. Missing one or more required " f"models: {models_to_plot}")
        return

    errors = {
        "lpr": y_true - predictions["lpr"],
        "rsklpr": y_true - predictions["rsklpr"],
        "gclpr_subway": y_true - predictions["gclpr_subway"],
        "grclpr_subway": y_true - predictions["grclpr_subway"],
    }
    masks = {name: np.isfinite(values) for name, values in errors.items()}
    all_errors = np.concatenate([values[masks[name]] for name, values in errors.items()])
    vmax = float(np.nanpercentile(np.abs(all_errors), 99))
    vmin = -vmax

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle("NYC Airbnb: Geospatial Prediction Errors", fontsize=18)
    axes_flat = cast(list[Axes], np.asarray(axes, dtype=object).ravel().tolist())
    panels = [
        ("lpr", "LPR Errors"),
        ("rsklpr", "RSKLPR Errors"),
        ("gclpr_subway", "GC-LPR Errors"),
        ("grclpr_subway", "GRC-LPR Errors"),
    ]

    longitudes = coords[_longitude_column].to_numpy(copy=True)
    latitudes = coords[_latitude_column].to_numpy(copy=True)

    for ax, (name, title) in zip(axes_flat, panels, strict=True):
        scatter = ax.scatter(
            longitudes[masks[name]],
            latitudes[masks[name]],
            c=errors[name][masks[name]],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=5,
            alpha=0.5,
        )
        ax.set_title(f"{title} (RMSE: {results[name]['RMSE']:.1f})")
        fig.colorbar(scatter, ax=ax, label="Error (Actual - Pred)")

    axes_flat[2].set_xlabel("Longitude")
    axes_flat[3].set_xlabel("Longitude")
    axes_flat[0].set_ylabel("Latitude")
    axes_flat[2].set_ylabel("Latitude")
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    if save_plots:
        target_dir = output_dir or Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / "nyc_airbnb_exp_2_geospatial_errors.png")
    else:
        plt.show()

    plt.close(fig)
