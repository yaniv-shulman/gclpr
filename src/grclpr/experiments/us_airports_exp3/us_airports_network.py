"""Utilities for Experiment 3 on the US airport network."""
import math
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rsklpr.kernels import laplacian_normalized_metric
from rsklpr.rsklpr import Rsklpr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

KernelFn: TypeAlias = Callable[..., np.ndarray]
SearchCV: TypeAlias = GridSearchCV | RandomizedSearchCV

_epsilon_similarity = 1.0e-3


def load_airport_data_and_graph(
    airports_csv_path: Path,
    routes_csv_path: Path,
    random_generator: np.random.Generator | None = None,
    count_weighted_delay: bool = True,
) -> tuple[nx.Graph, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the airport graph, engineer node features, and synthesize the target.

    Args:
        airports_csv_path: Local path to the airport metadata CSV.
        routes_csv_path: Local path to the route edge-list CSV.
        random_generator: Optional random number generator for reproducibility.
        count_weighted_delay: Whether to use traffic counts during synthetic
            delay propagation.

    Returns:
        The airport graph, the engineered feature frame, the feature matrix, and
        the synthetic delay target.

    Raises:
        FileNotFoundError: If either CSV path does not exist.
    """
    if not airports_csv_path.exists():
        raise FileNotFoundError(f"Missing airport metadata CSV: {airports_csv_path}")

    if not routes_csv_path.exists():
        raise FileNotFoundError(f"Missing airport routes CSV: {routes_csv_path}")

    if random_generator is None:
        random_generator = np.random.default_rng(seed=42)

    airports_df = pd.read_csv(airports_csv_path).set_index("iata")
    routes_df = pd.read_csv(routes_csv_path)
    routes_df["inverse_count"] = 1.0 / routes_df["count"]

    airport_network = nx.from_pandas_edgelist(
        df=routes_df,
        source="origin",
        target="destination",
        edge_attr=["count", "inverse_count"],
    )

    valid_airports = list(airport_network.nodes())
    airports_df = airports_df.loc[airports_df.index.intersection(valid_airports)]
    airport_network = airport_network.subgraph(airports_df.index).copy()

    positions = {
        iata: (float(row["longitude"]), float(row["latitude"]))
        for iata, row in airports_df.iterrows()
    }

    nx.set_node_attributes(airport_network, positions, "pos")
    nx.set_node_attributes(airport_network, airports_df["name"].to_dict(), "name")
    nx.set_node_attributes(airport_network, airports_df["city"].to_dict(), "city")
    nx.set_node_attributes(airport_network, airports_df["state"].to_dict(), "state")

    df_features = pd.DataFrame(index=airport_network.nodes())
    df_features["pagerank"] = pd.Series(nx.pagerank(airport_network))
    df_features["betweenness"] = pd.Series(nx.betweenness_centrality(airport_network))
    df_features["degree"] = pd.Series(dict(airport_network.degree()))
    df_features = df_features.join(airports_df[["latitude", "longitude"]]).dropna()

    scaler = StandardScaler()

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_features),
        columns=df_features.columns,
        index=df_features.index,
    )

    base_delay = (
        0.3 * df_scaled["pagerank"]
        + 0.5 * df_scaled["betweenness"]
        + 0.2 * df_scaled["degree"]
        + 0.1 * random_generator.normal(size=len(df_scaled))
    )

    if count_weighted_delay:
        adj_matrix = nx.to_pandas_adjacency(
            airport_network,
            nodelist=df_features.index,
            weight="count",
        )
    else:
        adj_matrix = nx.to_pandas_adjacency(
            airport_network,
            nodelist=df_features.index,
        )

    adj_norm = adj_matrix.div(adj_matrix.sum(axis=1), axis=0).fillna(0.0)
    delay_t = base_delay.copy()

    for _ in range(7):
        neighbor_delay = adj_norm.dot(delay_t)
        delay_t = 0.7 * base_delay + 0.3 * neighbor_delay

    y_delay = delay_t.to_numpy(dtype=float)
    y_delay -= y_delay.min() - 0.1
    x_features = df_features.to_numpy(dtype=float, copy=True)

    airport_network = airport_network.subgraph(df_features.index).copy()
    nx.set_node_attributes(
        airport_network,
        values=dict(zip(df_features.index, y_delay)),
        name="delay",
    )
    return airport_network, df_features, x_features, y_delay


def build_original_idx_to_node_map(df_features: pd.DataFrame) -> dict[int, str]:
    """Build the row-index to airport-node mapping for kernel lookups.

    Args:
        df_features: Engineered feature frame indexed by airport node ID.

    Returns:
        A mapping from row index to airport node ID.
    """
    return dict(enumerate(df_features.index.to_list()))


def scale_non_positional_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale all non-geospatial columns while preserving latitude/longitude.

    Args:
        x_train: Training feature matrix.
        x_test: Test feature matrix.

    Returns:
        The scaled training and test matrices.
    """
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()
    if x_train.shape[1] <= 2:
        return x_train_scaled, x_test_scaled

    scaler = StandardScaler()
    x_train_scaled[:, :-2] = scaler.fit_transform(x_train[:, :-2])
    x_test_scaled[:, :-2] = scaler.transform(x_test[:, :-2])
    x_train_scaled[:, -2:] = x_train[:, -2:]
    x_test_scaled[:, -2:] = x_test[:, -2:]
    return x_train_scaled, x_test_scaled


def airport_network_kernel_factory(
    graph: nx.Graph,
    original_idx_to_node_map: Mapping[int, str | None],
    train_set_original_indices: np.ndarray,
    predict_set_original_indices: np.ndarray,
    distance_scale: float = 1.0,
) -> KernelFn:
    """Build an index-aware graph kernel based on unweighted shortest paths.

    Args:
        graph: Airport route graph.
        original_idx_to_node_map: Global row-index to node-ID mapping.
        train_set_original_indices: Original row indices for the training set.
        predict_set_original_indices: Original row indices for the prediction set.
        distance_scale: Exponential decay scale for hop distance.

    Returns:
        A kernel function compatible with ``Rsklpr``.
    """
    if train_set_original_indices.ndim != 1:
        raise TypeError("train_set_original_indices must be a 1D NumPy array.")
    if predict_set_original_indices.ndim != 1:
        raise TypeError("predict_set_original_indices must be a 1D NumPy array.")

    def _kernel(
        x_0: np.ndarray | None,
        x_neighbors: np.ndarray | None,
        dist_x_neighbors: np.ndarray | None,
        index_x_0: int,
        indices_neighbors: np.ndarray,
    ) -> np.ndarray:
        del x_0, x_neighbors, dist_x_neighbors

        try:
            target_original_idx = int(predict_set_original_indices[index_x_0])
        except IndexError:
            warnings.warn(f"index_x_0 ({index_x_0}) is out of bounds.", stacklevel=2)
            target_original_idx = -1

        target_node = original_idx_to_node_map.get(target_original_idx)
        neighbor_relative_indices = np.asarray(indices_neighbors[0], dtype=int)
        similarity_scores = np.full(
            shape=len(neighbor_relative_indices),
            fill_value=_epsilon_similarity,
            dtype=float,
        )

        if target_node is None or target_node not in graph:
            return similarity_scores.reshape(1, -1)

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

            neighbor_node = original_idx_to_node_map.get(neighbor_original_idx)
            if neighbor_node is None or neighbor_node not in graph:
                continue
            if neighbor_node == target_node:
                similarity_scores[score_index] = 1.0
                continue

            try:
                path_dist = nx.shortest_path_length(
                    graph,
                    source=target_node,
                    target=neighbor_node,
                    weight=None,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            if np.isfinite(path_dist) and path_dist >= 0.0:
                similarity_scores[score_index] = math.exp(-float(path_dist) / distance_scale)

        return similarity_scores.reshape(1, -1)

    return _kernel


class GraphRsklprWrapper(BaseEstimator, RegressorMixin):
    """Wrap ``Rsklpr`` with an index-aware airport-graph context kernel."""

    def __init__(
        self,
        size_neighborhood: int = 25,
        degree: int = 1,
        distance_scale: float = 1.0,
        kp: KernelFn = laplacian_normalized_metric,
        kr: str = "none",
        metric_x: str = "mahalanobis",
        metric_x_params: dict[str, Any] | None = None,
        airport_network: nx.Graph | None = None,
        original_idx_to_node_map: Mapping[int, str | None] | None = None,
    ) -> None:
        """Initialize the graph-aware wrapper.

        Args:
            size_neighborhood: Number of nearest neighbors.
            degree: Local polynomial degree.
            distance_scale: Exponential decay scale for graph hop distance.
            kp: Base predictor kernel.
            kr: Response kernel mode.
            metric_x: Distance metric for feature-space neighbor search.
            metric_x_params: Optional metric parameters.
            airport_network: Shared airport graph.
            original_idx_to_node_map: Shared row-index to node-ID mapping.
        """
        self.size_neighborhood = size_neighborhood
        self.degree = degree
        self.distance_scale = distance_scale
        self.kp = kp
        self.kr = kr
        self.metric_x = metric_x
        self.metric_x_params = metric_x_params
        self.airport_network = airport_network
        self.original_idx_to_node_map = original_idx_to_node_map

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GraphRsklprWrapper":
        """Store training data and global indices.

        Args:
            x: Training features with original row indices in the last column.
            y: Training targets.

        Returns:
            The fitted wrapper.

        Raises:
            ValueError: If the shared airport graph data were not supplied.
        """
        if self.airport_network is None or self.original_idx_to_node_map is None:
            raise ValueError("GraphRsklprWrapper requires airport_network and original_idx_to_node_map.")

        self.train_original_indices_ = x[:, -1].astype(int)
        self.x_fit_ = x[:, :-1]
        self.y_fit_ = y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict with the graph-aware ``Rsklpr`` model.

        Args:
            x: Prediction features with original row indices in the last column.

        Returns:
            Predicted responses.
        """
        predict_original_indices = x[:, -1].astype(int)
        x_predict_features = x[:, :-1]
        original_idx_to_node_map = cast(Mapping[int, str | None], self.original_idx_to_node_map)

        graph_kernel = airport_network_kernel_factory(
            graph=cast(nx.Graph, self.airport_network),
            original_idx_to_node_map=original_idx_to_node_map,
            train_set_original_indices=self.train_original_indices_,
            predict_set_original_indices=predict_original_indices,
            distance_scale=self.distance_scale,
        )
        model = Rsklpr(
            size_neighborhood=self.size_neighborhood,
            degree=self.degree,
            kp=[self.kp, graph_kernel],
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
        size_neighborhood: int = 25,
        degree: int = 1,
        kp: list[KernelFn] | KernelFn | None = None,
        kr: str = "none",
        metric_x: str = "mahalanobis",
        metric_x_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            size_neighborhood: Number of nearest neighbors.
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
        """Store training data.

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
    """Count the size of a discrete scikit-learn parameter grid."""
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
    """Tune a feature-only LPR or RSKLPR model by cross-validation."""
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


def cv_graph(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: np.ndarray,
    airport_network: nx.Graph,
    original_idx_to_node_map: dict[int, str | None],
    param_grid: Any,
    cv: int,
    n_jobs: int,
    verbose: int = 0,
    search_mode: str = "grid",
    n_iter: int | None = None,
    random_state: int = 42,
    scoring: Any = "neg_mean_squared_error",
) -> SearchCV:
    """Tune a graph-context GC-LPR or GRC-LPR model by cross-validation."""
    print("🔎 Grid searching GRC-LPR (Graph)... (CV)")
    x_train_with_indices = np.hstack((x_train, train_indices.reshape(-1, 1)))
    estimator = GraphRsklprWrapper(
        airport_network=airport_network,
        original_idx_to_node_map=original_idx_to_node_map,
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


def _setup_airport_map_ax() -> Any:
    """Create a standard Cartopy axes object for the contiguous US."""
    plt.figure(figsize=(15, 10))
    ax: Any = plt.axes(
        projection=ccrs.AlbersEqualArea(central_longitude=-96.0, central_latitude=37.5)
    )
    ax.set_extent([-125, -66.5, 20, 50], crs=ccrs.Geodetic())
    ax.add_feature(cfeature.LAND, color="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.STATES, linestyle=":", linewidth=0.3)
    return ax


def _draw_airport_edges(
    ax: Any,
    airport_network: nx.Graph,
    pos_latlon: dict[str, tuple[float, float]],
    alpha: float = 0.1,
    linewidth: float = 0.5,
) -> None:
    """Draw graph edges onto a Cartopy map."""
    for start_node, end_node in airport_network.edges():
        if start_node not in pos_latlon or end_node not in pos_latlon:
            continue
        lon1, lat1 = pos_latlon[start_node]
        lon2, lat2 = pos_latlon[end_node]
        ax.plot(
            [lon1, lon2],
            [lat1, lat2],
            color="gray",
            alpha=alpha,
            linewidth=linewidth,
            transform=ccrs.Geodetic(),
        )


def _project_node_positions(
    pos_latlon: dict[str, tuple[float, float]],
    ax: Any,
) -> dict[str, tuple[float, float]]:
    """Project longitude/latitude node positions into the map coordinates."""
    data_proj = ccrs.PlateCarree()
    map_proj = ax.projection
    pos_projected: dict[str, tuple[float, float]] = {}
    for node, (lon, lat) in pos_latlon.items():
        x_proj, y_proj = map_proj.transform_point(lon, lat, data_proj)
        pos_projected[node] = (x_proj, y_proj)
    return pos_projected


def plot_airport_network_map(
    airport_network: nx.Graph,
    save_plot: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Plot the airport network colored by the synthetic delay target."""
    print("🗺️ Plotting airport network graph on US map...")
    ax = _setup_airport_map_ax()
    pos_latlon = cast(
        dict[str, tuple[float, float]],
        nx.get_node_attributes(airport_network, "pos"),
    )
    delays = cast(dict[str, float], nx.get_node_attributes(airport_network, "delay"))
    degrees = dict(airport_network.degree())
    pos_latlon = {
        node: pos_latlon[node]
        for node in airport_network.nodes()
        if node in pos_latlon
    }
    nodes_list = list(pos_latlon.keys())
    node_colors = [delays.get(node, 0.0) for node in nodes_list]
    node_sizes = [degrees.get(node, 0) * 5 for node in nodes_list]
    _draw_airport_edges(ax, airport_network, pos_latlon, alpha=0.1, linewidth=0.5)
    pos_projected = _project_node_positions(pos_latlon, ax)
    nx.draw_networkx_nodes(
        airport_network,
        pos_projected,
        nodelist=nodes_list,
        node_color=node_colors,
        node_size=node_sizes,
        cmap="viridis",
        alpha=0.7,
        ax=ax,
    )
    scalar_map = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)),
    )
    scalar_map.set_array([])
    cbar = plt.colorbar(scalar_map, ax=ax, shrink=0.8, orientation="vertical", pad=0.02)
    cbar.set_label("Synthetic Delay (Higher is Worse)", rotation=270, labelpad=15, fontsize=10)
    plt.title("US Airport Network with Synthetic Delay Propagation", fontsize=16)

    if save_plot:
        target_dir = output_dir or Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(target_dir / "us_airport_delay_network.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_graph_kernel_similarity(
    airport_network: nx.Graph,
    reference_node_id: str,
    distance_scale: float,
    save_plot: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Visualize graph-kernel similarity from one reference airport."""
    if reference_node_id not in airport_network:
        raise ValueError(f"Reference node '{reference_node_id}' not in graph.")

    nodes_list = list(airport_network.nodes())
    node_to_index_map = {node_id: idx for idx, node_id in enumerate(nodes_list)}
    reference_node_index = node_to_index_map[reference_node_id]
    all_indices = np.arange(len(nodes_list))
    local_idx_to_node = {idx: node for idx, node in enumerate(nodes_list)}
    kernel_function = airport_network_kernel_factory(
        graph=airport_network,
        train_set_original_indices=all_indices,
        predict_set_original_indices=all_indices,
        original_idx_to_node_map=local_idx_to_node,
        distance_scale=distance_scale,
    )
    similarity_scores_array = kernel_function(
        x_0=None,
        x_neighbors=None,
        dist_x_neighbors=None,
        index_x_0=reference_node_index,
        indices_neighbors=all_indices.reshape(1, -1),
    )
    node_colors = similarity_scores_array.flatten()

    ax = _setup_airport_map_ax()
    pos_latlon = cast(
        dict[str, tuple[float, float]],
        nx.get_node_attributes(airport_network, "pos"),
    )
    degrees = dict(airport_network.degree())
    pos_latlon = {
        node: pos_latlon[node]
        for node in airport_network.nodes()
        if node in pos_latlon
    }
    node_sizes = [degrees.get(node, 0) * 5 for node in nodes_list]
    _draw_airport_edges(ax, airport_network, pos_latlon, alpha=0.05, linewidth=0.5)
    pos_projected = _project_node_positions(pos_latlon, ax)
    nx.draw_networkx_nodes(
        airport_network,
        pos_projected,
        nodelist=nodes_list,
        node_color=node_colors,
        node_size=node_sizes,
        cmap="viridis",
        alpha=0.9,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    nx.draw_networkx_nodes(
        airport_network,
        pos_projected,
        nodelist=[reference_node_id],
        node_color="red",
        node_size=100,
        edgecolors="black",
        ax=ax,
    )
    scalar_map = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=0.0, vmax=1.0),
    )
    scalar_map.set_array([])
    cbar = plt.colorbar(scalar_map, ax=ax, shrink=0.8, orientation="vertical", pad=0.02)
    cbar.set_label(
        f"Kernel Similarity (exp(-hop_distance / {distance_scale:.1f}))",
        rotation=270,
        labelpad=15,
        fontsize=10,
    )
    plt.title(f"Graph Kernel Similarity from {reference_node_id}", fontsize=16)

    if save_plot:
        target_dir = output_dir or Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"us_airport_delay_sim_{reference_node_id.lower()}.png"
        plt.savefig(target_dir / filename, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_prediction_scatter_grid(
    predictions: dict[str, np.ndarray],
    results: dict[str, dict[str, float]],
    y_true: np.ndarray,
    save_plots: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Plot actual-vs-predicted scatter plots for Experiment 3."""
    print("📈 Plotting scatter results...")
    model_count = len(predictions)
    ncols = 3
    nrows = int(np.ceil(model_count / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle("US Airport Network: Actual vs. Predicted Delay", fontsize=18)
    axes_flat = [cast(Any, axes)] if model_count == 1 else cast(
        list[Any],
        np.asarray(axes, dtype=object).ravel().tolist(),
    )
    plot_min = float(np.min(y_true))
    plot_max = float(np.max(y_true))
    name_to_title = {
        "knn": "K-Nearest Neighbors",
        "lpr": "LPR",
        "rsklpr": "RSKLPR (Standard)",
        "gclpr_graph": "GC-LPR (Graph)",
        "grclpr_graph": "GRC-LPR (Graph)",
    }

    for index, name in enumerate(predictions):
        ax = axes_flat[index]
        y_pred = predictions[name]
        mask = np.isfinite(y_pred)
        ax.scatter(y_true[mask], y_pred[mask], alpha=0.16, s=12)
        ax.plot([plot_min, plot_max], [plot_min, plot_max], "r--", lw=2)
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect("equal", "box")
        ax.set_title(
            f"{name_to_title[name]}\n"
            f"RMSE: {results[name]['RMSE']:.3f} | R2: {results[name]['R2']:.3f}"
        )
        if index >= (model_count - ncols):
            ax.set_xlabel("Actual Delay")
        if index % ncols == 0:
            ax.set_ylabel("Predicted Delay")

    for index in range(model_count, len(axes_flat)):
        axes_flat[index].axis("off")

    if save_plots:
        target_dir = output_dir or Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / "us_airports_exp_3_scatter_grid.png")
    else:
        plt.show()
    plt.close(fig)
