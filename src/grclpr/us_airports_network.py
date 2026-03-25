"""
Experiment 3: Generalized LPR on the US Airport Network

This module demonstrates the GRC-LPR framework on a non-geographical graph.
It uses the US domestic flight network, where:
- Nodes: Airports
- Edges: Direct flight routes
- X (Features): Graph centrality metrics (PageRank, etc.) & coordinates.
- Y (Target): A synthetically generated delay, created by propagating a base delay over the graph.

This simulation study is designed to create a target variable Y that is known to be dependent on both node features and
graph topology. The experiment then tests which model is best at recovering this relationship.
"""

import math
import time
import warnings
from collections.abc import Iterable
from typing import Dict, Optional, Callable, Tuple, List, Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rsklpr.kernels import laplacian_normalized_metric, tricube_normalized_metric
from rsklpr.rsklpr import Rsklpr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def load_airport_data_and_graph(
    random_generator: Optional[np.random.Generator] = None,
    count_weighted_delay: bool = False,
) -> Tuple[nx.Graph, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Loads airport and route data, builds the graph, creates features, and synthesizes a target variable (Y) for the
    experiment. Y (synthetic delay) is created from a combination of node features and graph-propagated noise, ensuring
    the graph structure is predictive.

    Args:
        random_generator: Optional NumPy random generator for reproducibility. If None, a default generator with
        seed=42 is used.

    Returns:
        airport_network: The airport route network graph.
        df_features: DataFrame of engineered node features.
        X_features: Feature matrix for ML.
        Y_delay: Synthesized target variable (delay).
    """
    print("✈️ Loading airport and route data...")

    if random_generator is None:
        print(" No random generator provided, using default with seed=42.")
        random_generator = np.random.default_rng(seed=42)

    # 1. Load Node Features (Airports)
    # Using the Vega-Lite dataset source
    airport_url: str = (
        "https://raw.githubusercontent.com/vega/vega-datasets/main/data/airports.csv"
    )
    airports_df: pd.DataFrame

    try:
        airports_df = pd.read_csv(airport_url)
        airports_df = airports_df.set_index("iata")
    except Exception as e:
        print(f"Error loading airport data: {e}")
        print("Please download 'airports.csv' from the Vega dataset repo.")
        raise

    print(f"Loaded {len(airports_df)} airports.")

    # 2. Load Edge List (Routes) and Build Graph
    routes_url: str = (
        "https://raw.githubusercontent.com/vega/vega-datasets/main/data/flights-airport.csv"
    )
    routes_df: pd.DataFrame

    try:
        routes_df = pd.read_csv(routes_url)
    except Exception as e:
        print(f"Error loading routes data: {e}")
        raise

    print(f"Loaded {len(routes_df)} routes.")
    routes_df["inverse_count"] = 1.0 / routes_df["count"]

    # Create graph from the edge list
    airport_network: nx.Graph = nx.from_pandas_edgelist(
        df=routes_df,
        source="origin",
        target="destination",
        edge_attr=["count", "inverse_count"],
    )

    adj_matrix: pd.DataFrame = nx.to_pandas_adjacency(airport_network)
    assert adj_matrix.shape[0] == airport_network.number_of_nodes()
    assert (adj_matrix == adj_matrix.T).all().all(), "Graph is not undirected!"
    assert (adj_matrix.values.diagonal() == 0).all(), "Graph has self-loops!"
    assert adj_matrix.isnull().sum().sum() == 0, "Adjacency matrix has NaNs!"
    unique_vals: np.ndarray = np.unique(adj_matrix.values)

    assert (
        unique_vals.size == 2
    ), f"Adjacency has {unique_vals.size} uniques: {unique_vals}"
    assert unique_vals.min() == 0 and unique_vals.max() == 1

    # The routes file only contains a subset of airports, so we filter
    # our main airport dataframe to match the nodes in the graph.
    valid_airports: List[str] = list(airport_network.nodes())
    airports_df = airports_df.loc[airports_df.index.intersection(valid_airports)]

    # Get a list of nodes that are in the graph AND have feature data
    nodes_with_data: List[str] = list(airports_df.index)

    # Create the subgraph *before* feature engineering
    airport_network = airport_network.subgraph(nodes_with_data)

    # Add node attributes (lat/lon) to the graph for plotting
    pos: Dict[str, Tuple[float, float]] = {
        iata: (row["longitude"], row["latitude"])
        for iata, row in airports_df.iterrows()
    }

    nx.set_node_attributes(airport_network, pos, "pos")
    nx.set_node_attributes(airport_network, airports_df["name"].to_dict(), "name")
    nx.set_node_attributes(airport_network, airports_df["city"].to_dict(), "city")
    nx.set_node_attributes(airport_network, airports_df["state"].to_dict(), "state")

    print(
        f"Graph built: {airport_network.number_of_nodes()} airports, {airport_network.number_of_edges()} routes."
    )

    # Engineer Features (X)
    print("Engineering features (X) from graph topology...")

    # Use centrality as our R^D features
    df_features: pd.DataFrame = pd.DataFrame(index=airport_network.nodes())
    df_features["pagerank"] = pd.Series(nx.pagerank(airport_network))
    df_features["betweenness"] = pd.Series(nx.betweenness_centrality(airport_network))
    df_features["degree"] = pd.Series(dict(airport_network.degree()))

    # Add coordinates as features to prove they aren't enough
    df_features = df_features.join(airports_df[["latitude", "longitude"]])
    print(f" Features engineered for {len(df_features)} airports.")

    # Drop any nodes we couldn't get features for
    df_features = df_features.dropna()
    print(f" After dropping NaNs, {len(df_features)} airports remain.")

    # Synthesize Target Variable (Y)
    print("Synthesizing target variable (y_delay)...")

    # Scale features for stable synthesis
    scaler: StandardScaler = StandardScaler()
    features_scaled: np.ndarray = scaler.fit_transform(df_features)

    df_features_scaled: pd.DataFrame = pd.DataFrame(
        features_scaled, columns=df_features.columns, index=df_features.index
    )

    # Base Delay: A combination of features + noise
    base_delay: pd.Series = (
        0.3 * df_features_scaled["pagerank"]
        + 0.5 * df_features_scaled["betweenness"]
        + 0.2 * df_features_scaled["degree"]
        + 0.1 * random_generator.normal(size=len(df_features_scaled))  # Random noise
    )

    # b) Propagate Delay: Simulate delay propagation through the network
    delay_t: pd.Series = base_delay.copy()

    if count_weighted_delay:
        print(" Using weighted edges for delay propagation.")
        # Create weighted adjacency matrix
        adj_matrix: pd.DataFrame = nx.to_pandas_adjacency(
            airport_network, nodelist=df_features.index, weight="count"
        )
    else:
        print(" Using unweighted edges for delay propagation.")
        adj_matrix: pd.DataFrame = nx.to_pandas_adjacency(
            airport_network,
            nodelist=df_features.index,
        )

    # Normalize adjacency matrix by row (sum to 1) for averaging
    adj_norm: pd.DataFrame = adj_matrix.div(adj_matrix.sum(axis=1), axis=0).fillna(0)
    print("Simulating delay propagation over 7 steps...")

    for _ in range(7):
        # Average neighbor delay
        neighbor_delay: pd.Series = adj_norm.dot(delay_t)

        # New delay is a mix of its own delay and neighbor delay
        delay_t = 0.7 * base_delay + 0.3 * neighbor_delay

    # Final Y is the propagated delay
    y_delay: np.ndarray = delay_t.values
    y_delay -= y_delay.min() - 0.1  # Shift to min=0.1

    # Final X is the feature matrix
    x_features: np.ndarray = df_features.values

    # Filter graph to only include nodes we have X and Y for (This is slightly redundant now but good for robustness)
    final_nodes: pd.Index = df_features.index
    airport_network = airport_network.subgraph(final_nodes)

    print("Data preparation complete.")

    # Create a dictionary mapping node_id -> delay
    delay_dict: Dict[str, float] = dict(zip(df_features.index, y_delay))

    # Add the final Y (delay) to the graph nodes, this is used later for plotting
    nx.set_node_attributes(airport_network, values=delay_dict, name="delay")
    return airport_network, df_features, x_features, y_delay


def prepare_ml_data(
    df_features: pd.DataFrame,
    x_features: np.ndarray,
    y_delay: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[int, str],
]:
    """
    Prepares the data for the experiment.

    1. Creates the index-to-node mapping.
    2. Splits data into train/test sets.
    3. Scales the X features.

    Args:
        df_features: DataFrame of node features.
        x_features: Feature matrix.
        y_delay: Target variable array.
        test_size: Proportion of data to use as test set.
        random_state: Random seed for reproducibility of the train/test split.

    Returns:
        x_train_scaled: Scaled training features.
        x_test_scaled: Scaled testing features.
        y_train: Training target variable.
        y_test: Testing target variable.
        train_indices: Original indices of training samples.
        test_indices: Original indices of testing samples.
        original_idx_to_node_map: Mapping from original index to node ID (airport code). This is used by the kernel for
            lookups.
    """

    # Create the mapping from original array index (0...N-1) to node ID
    print("Creating original index-to-node map...")
    original_indices: np.ndarray = np.arange(len(df_features))
    node_ids: np.ndarray = df_features.index.values  # This gets the IATA codes in order
    original_idx_to_node_map: Dict[int, str] = dict(zip(original_indices, node_ids))
    print(f"Index to node map created for {len(original_idx_to_node_map)} airports.")

    # Split the data into training and testing sets
    print("Splitting data into train/test sets...")

    # We need to split x_features, y_delay, and original_indices (!) all at once.
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray

    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        x_features,
        y_delay,
        original_indices,  # The array of original indices (0...N-1)
        test_size=test_size,
        random_state=random_state,
    )

    print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Scale features (fit on train, transform both train and test)
    print("Scaling non positional features...")
    x_train_scaled: np.ndarray = np.empty_like(x_train)
    x_test_scaled: np.ndarray = np.empty_like(x_test)

    scaler: StandardScaler = StandardScaler()
    x_train_scaled[:, :-2] = scaler.fit_transform(x_train[:, :-2])
    x_test_scaled[:, :-2] = scaler.transform(x_test[:, :-2])

    x_train_scaled[:, -2:] = x_train[:, -2:]  # Latitude and Longitude unscaled
    x_test_scaled[:, -2:] = x_test[:, -2:]

    return (
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        train_indices,
        test_indices,
        original_idx_to_node_map,
    )


# Plotting Helper Functions


def _setup_airport_map_ax() -> Any:
    """
    Sets up and returns a standard Cartopy axes object for plotting the US.

    Returns:
        ax: The Cartopy axes object.
    """
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
    pos_latlon: Dict[str, Tuple[float, float]],
    alpha: float = 0.1,
    linewidth: float = 0.5,
) -> None:
    """
    Draws the graph edges onto the map axes as great circle lines.

    Args:
        ax: The Cartopy axes to draw on.
        airport_network: The airport network graph.
        pos_latlon: Dictionary mapping node IDs to (longitude, latitude) tuples.
        alpha: Transparency of the edges.
        linewidth: Width of the edge lines.
    """
    print("Drawing edges...")
    start_node: str
    end_node: str

    for start_node, end_node in airport_network.edges():
        if start_node in pos_latlon and end_node in pos_latlon:
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
    pos_latlon: Dict[str, Tuple[float, float]], ax: Any
) -> Dict[str, Tuple[float, float]]:
    """
    Projects longitude/latitude positions to the map's coordinate system.

    Args:
        pos_latlon: Dictionary mapping node IDs to (longitude, latitude) tuples.
        ax: The Cartopy axes with the desired projection.

    Returns:
        pos_projected: Dictionary mapping node IDs to projected (x, y) tuples.
    """
    print("Projecting node coordinates...")
    data_proj: ccrs.PlateCarree = ccrs.PlateCarree()
    map_proj: Any = ax.projection
    pos_projected: Dict[str, Tuple[float, float]] = {}
    node: str
    lon: float
    lat: float

    for node, (lon, lat) in pos_latlon.items():
        x_proj: float
        y_proj: float
        x_proj, y_proj = map_proj.transform_point(lon, lat, data_proj)
        pos_projected[node] = (x_proj, y_proj)

    return pos_projected


def plot_airport_network_map(
    airport_network: nx.Graph, save_plot: bool = False
) -> None:
    """
    Plots the airport network on a map of the United States. Nodes are colored by their synthetic delay and sized by
    their degree in the network.

    Args:
        airport_network: The airport network graph.
        save_plot: Whether to save the plot as a PNG file.
    """
    print("🗺️ Plotting airport network graph on US map...")

    # Set up the map
    ax: Any = _setup_airport_map_ax()

    # Get data for plotting
    pos_latlon: Dict[str, Tuple[float, float]] = nx.get_node_attributes(
        airport_network, "pos"
    )

    delays: Dict[str, float] = nx.get_node_attributes(airport_network, "delay")
    degrees: Dict[str, int] = dict(airport_network.degree())

    pos_latlon: Dict[str, Tuple[float, float]] = {
        node: pos_latlon[node] for node in airport_network.nodes() if node in pos_latlon
    }

    nodes_list: List[str] = list(pos_latlon.keys())
    node_colors: List[float] = [delays.get(node, 0) for node in nodes_list]
    node_sizes: List[int] = [degrees.get(node, 0) * 5 for node in nodes_list]

    # Draw Edges
    _draw_airport_edges(ax, airport_network, pos_latlon, alpha=0.1, linewidth=0.5)

    # Project node positions
    pos_projected: Dict[str, Tuple[float, float]] = _project_node_positions(
        pos_latlon, ax
    )

    # Draw Nodes
    nx.draw_networkx_nodes(
        airport_network,
        pos_projected,
        nodelist=nodes_list,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.viridis,
        alpha=0.7,
        ax=ax,
    )

    sm: plt.cm.ScalarMappable = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)),
    )

    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, orientation="vertical", pad=0.02)

    cbar.set_label(
        "Synthetic Delay (Higher is Worse)", rotation=270, labelpad=15, fontsize=10
    )

    plt.title("US Airport Network with Synthetic Delay Propagation", fontsize=16)

    if save_plot:
        plt.savefig("airport_network_delay_map.png", dpi=300, bbox_inches="tight")
        print("Plot saved as 'airport_network_delay_map.png'.")

    plt.show()


def plot_graph_kernel_similarity(
    airport_network: nx.Graph,
    reference_node_id: str,
    graph_distance_scale: float,
    original_idx_to_node_map: Dict[int, str],
    save_plot: bool = False,
) -> None:
    """
    Visualizes the kernel similarity (based on shortest path) from a single reference airport to all other airports in
    the graph.

    Args:
        airport_network: The airport network graph.
        reference_node_id: The IATA code of the reference airport node.
        graph_distance_scale: The distance scale parameter for the kernel function.
        original_idx_to_node_map: Mapping from original index to node ID (airport code).
        save_plot: Whether to save the plot as a PNG file.
    """
    print(f"🎨 Visualizing kernel similarity from '{reference_node_id}'...")

    if reference_node_id not in airport_network:
        raise ValueError(f"Reference node '{reference_node_id}' not in graph.")

    nodes_list: List[str] = list(airport_network.nodes())

    if len(nodes_list) == 0:
        raise ValueError("Graph has no nodes to plot.")

    # Kernel-Specific Logic (build index map aligned with nodes_list)
    node_to_index_map: Dict[str, int] = {
        node_id: i for i, node_id in enumerate(nodes_list)
    }

    try:
        reference_node_index: int = node_to_index_map[reference_node_id]
    except KeyError:
        raise ValueError(
            f"Reference node '{reference_node_id}' not in node-to-index map."
        )

    # Use a local original-index mapping aligned to nodes_list to avoid permutation mismatch
    all_indices: np.ndarray = np.arange(len(nodes_list))
    local_idx_to_node: Dict[int, str] = {i: n for i, n in enumerate(nodes_list)}

    kernel_function: Callable[
        [np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray
    ] = airport_network_kernel_factory(
        airport_network=airport_network,
        train_set_original_indices=all_indices,
        predict_set_original_indices=all_indices,
        original_idx_to_node_map=local_idx_to_node,
        graph_distance_scale=graph_distance_scale,
    )

    print("Calling kernel function to get similarity scores...")

    similarity_scores_array = kernel_function(
        x_0=None,
        x_neighbors=None,
        dist_x_neighbors=None,
        index_x_0=reference_node_index,
        indices_neighbors=all_indices.reshape(1, -1),
    )

    node_colors: np.ndarray = similarity_scores_array.flatten()

    ax: Any = _setup_airport_map_ax()

    # Get data for plotting
    pos_latlon: Dict[str, Tuple[float, float]] = nx.get_node_attributes(
        airport_network, "pos"
    )

    degrees: Dict[str, int] = dict(airport_network.degree())

    pos_latlon: Dict[str, Tuple[float, float]] = {
        node: pos_latlon[node] for node in airport_network.nodes() if node in pos_latlon
    }

    node_sizes: List[int] = [degrees.get(node, 0) * 5 for node in nodes_list]

    _draw_airport_edges(ax, airport_network, pos_latlon, alpha=0.05, linewidth=0.5)

    pos_projected: Dict[str, Tuple[float, float]] = _project_node_positions(
        pos_latlon, ax
    )

    nx.draw_networkx_nodes(
        airport_network,
        pos_projected,
        nodelist=nodes_list,
        node_color=node_colors,  # Use colors from kernel function
        node_size=node_sizes,
        cmap=plt.cm.viridis,
        alpha=0.9,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )

    # Highlight the reference node
    nx.draw_networkx_nodes(
        airport_network,
        pos_projected,
        nodelist=[reference_node_id],
        node_color="red",
        node_size=100,
        edgecolors="black",
        ax=ax,
    )

    sm: plt.cm.ScalarMappable = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0.0, vmax=1.0)
    )

    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, orientation="vertical", pad=0.02)

    cbar.set_label(
        f"Kernel Similarity (exp(-path_length / {graph_distance_scale:.1f}))",
        rotation=270,
        labelpad=15,
        fontsize=10,
    )

    plt.title(f"Graph Kernel Similarity from {reference_node_id}", fontsize=16)

    ax.text(
        0.02,  # 2% from the left edge
        0.02,  # 2% from the bottom edge
        "Node Size = Airport Degree",
        transform=ax.transAxes,  # Use relative axes coordinates
        fontsize=10,
        verticalalignment="top",
        # Add a semi-transparent white box behind the text
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    if save_plot:
        plt.savefig(
            f"airport_kernel_similarity_{reference_node_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved as 'airport_kernel_similarity_{reference_node_id}.png'.")

    plt.show()


def airport_network_kernel_factory(
    airport_network: nx.Graph,
    train_set_original_indices: np.ndarray,
    predict_set_original_indices: np.ndarray,
    original_idx_to_node_map: Dict[int, Optional[str]],
    graph_distance_scale: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]:
    """
    Factory using original dataset index maps for robust node lookup. This kernel computes similarity based on
    shortest path lengths in the graph.

    Args:
        airport_network: The airport network NetworkX graph.
        train_set_original_indices: 1D NumPy array of original indices for the training set.
        predict_set_original_indices: 1D NumPy array that maps the *original dataset index* (0 to N-1) to a node ID.
        original_idx_to_node_map: Mapping from original index to node ID (airport code).
        graph_distance_scale: Scaling factor for the exponential decay based on path length.

    Returns:
        A kernel function that computes similarity scores based on shortest path lengths.
    """
    if (
        not isinstance(predict_set_original_indices, np.ndarray)
        or predict_set_original_indices.ndim != 1
    ):
        raise TypeError("predict_set_original_indices must be a 1D NumPy array.")
    if (
        not isinstance(train_set_original_indices, np.ndarray)
        or train_set_original_indices.ndim != 1
    ):
        raise TypeError("train_set_original_indices must be a 1D NumPy array.")

    _map: Dict[int, Optional[str]] = original_idx_to_node_map
    _pred_indices: np.ndarray = predict_set_original_indices
    _train_indices: np.ndarray = train_set_original_indices

    def _kernel(
        x_0: np.ndarray,
        x_neighbors: np.ndarray,
        dist_x_neighbors: np.ndarray,
        index_x_0: int,
        indices_neighbors: np.ndarray,
    ) -> np.ndarray:
        """
        The actual kernel function that computes similarity scores. This kernel uses the original dataset indices to look up
        the correct nodes in the graph. It calculates similarity based on the shortest path length between the target node
        and its neighbors in the predictors space.

        Args:
            x_0: Feature vector of the target sample (not used in this kernel).
            x_neighbors: Feature vectors of the neighbor samples (not used in this kernel).
            dist_x_neighbors: Distances to neighbor samples (not used in this kernel).
            index_x_0: Index of the target sample in the predict set.
            indices_neighbors: 2D array of neighbor indices in the train set.

        Returns:
            similarity_scores: 2D NumPy array of similarity scores between the target sample and its neighbors. If a path
            exists, the score is exp(-path_length / graph_distance_scale); otherwise, it is a small epsilon (0.001).
        """
        target_original_idx: int

        try:
            target_original_idx = int(_pred_indices[index_x_0])
        except IndexError:
            warnings.warn(f"index_x_0 ({index_x_0}) out of bounds.")
            target_original_idx = -1

        target_node: Optional[str] = _map.get(target_original_idx)
        neighbor_relative_indices: np.ndarray = indices_neighbors[0]

        eps: float = 0.001

        similarity_scores: np.ndarray = np.full(
            len(neighbor_relative_indices), fill_value=eps
        )

        if target_node is None or target_node not in airport_network:
            # If target node is invalid, return eps (0.01) for all neighbors
            return similarity_scores.reshape(1, -1)

        j: int

        for j, neighbor_relative_idx in enumerate(neighbor_relative_indices):
            neighbor_node: Optional[str]

            try:
                neighbor_original_idx: int = int(_train_indices[neighbor_relative_idx])
                neighbor_node = _map.get(neighbor_original_idx)
            except (IndexError, KeyError):
                neighbor_node = None

            if (
                neighbor_node is None
                or neighbor_node not in airport_network
                or neighbor_node == target_node
            ):
                if neighbor_node == target_node and target_node is not None:
                    similarity_scores[j] = 1.0
                continue

            try:
                path_dist: float = nx.shortest_path_length(
                    airport_network,
                    source=target_node,
                    target=neighbor_node,
                    weight="inverse_count",
                )

                if path_dist >= 0.0 and not (
                    np.isnan(path_dist) or np.isinf(path_dist) or path_dist is None
                ):
                    similarity_scores[j] = math.exp(-path_dist / graph_distance_scale)

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        return similarity_scores.reshape(1, -1)

    return _kernel


class IndexAwareGraphKernel(BaseEstimator):
    """
    A scikit-learn compatible wrapper for an index-aware kernel factory.

    This class is "stateful." It receives train/predict indices at runtime and becomes a callable kernel function that
    Rsklpr can use.

    GridSearchCV can tune its parameters (e.g., 'graph_distance_scale').
    """

    def __init__(
        self,
        airport_network: nx.Graph,
        original_idx_to_node_map: Dict[int, Optional[str]],
        graph_distance_scale: float = 1.0,
    ) -> None:
        """
        Initializes the IndexAwareGraphKernel.

        Args:
            airport_network: The airport network graph.
            original_idx_to_node_map: Mapping from original index to node ID (airport code).
            graph_distance_scale: Scaling factor for the exponential decay based on path length.
        """
        self.airport_network: nx.Graph = airport_network

        self.original_idx_to_node_map: Dict[int, Optional[str]] = (
            original_idx_to_node_map
        )

        self.graph_distance_scale: float = graph_distance_scale

        # These will be set at runtime by IndexAwareRsklprWrapper
        self.train_original_indices_: Optional[np.ndarray] = None
        self.predict_original_indices_: Optional[np.ndarray] = None

    def set_train_indices(self, train_indices: np.ndarray) -> None:
        """
        Called by the wrapper during fit()

        Args:
            train_indices: 1D NumPy array of original indices for the training set.

        Raises:
            TypeError: If train_indices is not a 1D NumPy array.
        """
        if not isinstance(train_indices, np.ndarray) or train_indices.ndim != 1:
            raise TypeError("train_indices must be a 1D NumPy array.")

        self.train_original_indices_ = train_indices

    def set_predict_indices(self, predict_indices: np.ndarray) -> None:
        """
        Called by the wrapper during predict()

        Args:
            predict_indices: 1D NumPy array of original indices for the prediction set.

        Raises:
            TypeError: If predict_indices is not a 1D NumPy array.
        """

        if not isinstance(predict_indices, np.ndarray) or predict_indices.ndim != 1:
            raise TypeError("predict_indices must be a 1D NumPy array.")

        self.predict_original_indices_ = predict_indices

    def __call__(
        self,
        x_0: np.ndarray,
        x_neighbors: np.ndarray,
        dist_x_neighbors: np.ndarray,
        index_x_0: int,
        indices_neighbors: np.ndarray,
    ) -> np.ndarray:
        """
        This makes the object callable, so Rsklpr sees it as a kernel function.

        Args:
            x_0: Feature vector of the target sample.
            x_neighbors: Feature vectors of the neighbor samples.
            dist_x_neighbors: Distances to neighbor samples.
            index_x_0: Index of the target sample in the predict set.
            indices_neighbors: 2D array of neighbor indices in the train set.

        Returns:
            similarity_scores: 2D NumPy array of similarity scores between the target sample and its neighbors.
        """
        if (
            self.train_original_indices_ is None
            or self.predict_original_indices_ is None
        ):
            raise RuntimeError("Indices not set. Use IndexAwareRsklprWrapper.")

        # Create the kernel function *just-in-time* with the correct indices
        kernel_func: Callable[
            [np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray
        ] = airport_network_kernel_factory(
            airport_network=self.airport_network,
            train_set_original_indices=self.train_original_indices_,
            predict_set_original_indices=self.predict_original_indices_,
            original_idx_to_node_map=self.original_idx_to_node_map,
            graph_distance_scale=self.graph_distance_scale,
        )

        # Call and return the real kernel's output
        return kernel_func(
            x_0, x_neighbors, dist_x_neighbors, index_x_0, indices_neighbors
        )

    # These are needed for BaseEstimator but we don't fit the kernel itself
    def fit(self, X, y=None) -> "IndexAwareGraphKernel":
        return self

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError("This is a kernel, not a model.")


class IndexAwareRsklprWrapper(BaseEstimator, RegressorMixin):
    """
    A generic Rsklpr wrapper that is "index-aware.". It assumes X has features in X[:, :-1] and original indices
    in X[:, -1]. It iterates through its `kp` list and injects indices into any kernel that supports it.

    It explicitly lists kernel-specific parameters (e.g., graph_distance_scale) in its __init__ to make them visible to
    GridSearchCV.
    """

    def __init__(
        self,
        size_neighborhood: int = 50,
        degree: int = 1,
        kp: Optional[List[Callable]] = None,
        kr: str = "none",
        metric_x: str = "minkowski",
        metric_x_params: Optional[Dict[str, Any]] = None,
        graph_distance_scale: float = 1.0,
    ) -> None:
        """
        Initializes the IndexAwareRsklprWrapper.

        Args:
            size_neighborhood: Number of neighbors to consider.
            degree: Degree of the polynomial kernel.
            kp: List of kernel functions (some may be index-aware). Defaults to [laplacian_normalized_metric].
            kr: Regression kernel type.
            metric_x: Metric for feature space.
            metric_x_params: Additional parameters for the feature space metric.
            graph_distance_scale: Graph distance scale for graph-based kernels.
        """
        self.size_neighborhood: int = size_neighborhood
        self.degree: int = degree

        self.kp: List[Callable] = (
            kp if kp is not None else [laplacian_normalized_metric]
        )
        self.kr: str = kr
        self.metric_x: str = metric_x
        self.metric_x_params: Optional[Dict[str, Any]] = metric_x_params

        # Store the kernel parameters explicitly
        self.graph_distance_scale: float = graph_distance_scale
        self.train_original_indices_: np.ndarray = np.array([])
        self.X_fit_: np.ndarray = np.array([])
        self.y_fit_: np.ndarray = np.array([])

    def _set_kernel_params(self) -> None:
        """
        Helper function to pass top-level parameters (like 'graph_distance_scale') down to the kernels in the 'kp' list
        that can accept them.
        """
        if len(self.kp) == 0:
            return

        kernel: Callable

        for kernel in self.kp:
            if hasattr(kernel, "get_params") and hasattr(kernel, "set_params"):
                # Get all params this kernel accepts (e.g., 'graph_distance_scale')
                valid_params: Iterable = kernel.get_params().keys()

                # Build kwargs dict from *self's* matching attributes
                kwargs_for_this_kernel: Dict[str, Any] = {}
                param_name: str

                for param_name in valid_params:
                    if hasattr(self, param_name):
                        # (e.g., param_name is 'graph_distance_scale')
                        # (self has 'self.graph_distance_scale')
                        kwargs_for_this_kernel[param_name] = getattr(self, param_name)

                if len(kwargs_for_this_kernel) > 0:
                    # Set the parameters (e.g., graph_distance_scale) on the
                    # IndexAwareGraphKernel object
                    kernel.set_params(**kwargs_for_this_kernel)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IndexAwareRsklprWrapper":
        """
        Stores training data and injects train_indices and kernel params.

        Args:
            X: Training feature matrix with original indices in the last column.
            y: Training target variable.

        Returns:
            self: Fitted IndexAwareRsklprWrapper instance.
        """
        self.train_original_indices_ = X[:, -1].astype(int)
        self.X_fit_ = X[:, :-1]  # Features
        self.y_fit_ = y

        # Inject train indices
        kernel: Callable

        for kernel in self.kp:
            if hasattr(kernel, "set_train_indices"):
                kernel.set_train_indices(self.train_original_indices_)

        # Pass down kernel params
        self._set_kernel_params()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts on new data X.

        Args:
            X: Prediction feature matrix with original indices in the last column.

        Returns:
            predictions: Predicted target variable.
        """
        predict_original_indices: np.ndarray = X[:, -1].astype(int)
        X_predict_features: np.ndarray = X[:, :-1]  # Features

        # Inject predict indices
        kernel: Callable

        for kernel in self.kp:
            if hasattr(kernel, "set_predict_indices"):
                kernel.set_predict_indices(predict_original_indices)

        # Pass down kernel params (GridSearchCV may have changed them)
        self._set_kernel_params()

        model: Rsklpr = Rsklpr(
            size_neighborhood=self.size_neighborhood,
            degree=self.degree,
            kp=self.kp,
            kr=self.kr,
            metric_x=self.metric_x,
            metric_x_params=self.metric_x_params,
            suppress_warnings=True,
        )

        model.fit(x=self.X_fit_, y=self.y_fit_)
        prediction = model.predict(x=X_predict_features)

        if np.isnan(prediction).any():

            warnings.warn(
                "NaN values found in predictions. Estimator params: "
                + str(self.get_params())
            )

        return prediction


# --- CV and Experiment Functions ---


def cv_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 1,
) -> GridSearchCV:
    """
    Runs GridSearchCV for Linear Regression.

    Args:
        x_train: Training feature matrix.
        y_train: Training target variable.
        param_grid: Parameter grid for GridSearchCV.
        cv: Number of cross-validation folds.
        n_jobs: Number of parallel jobs.
        verbose: Verbosity level for GridSearchCV.

    Returns:
        grid: Fitted GridSearchCV object.
    """
    print("🔎 Grid searching LinearRegression...")

    grid: GridSearchCV = GridSearchCV(
        LinearRegression(),
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )

    grid.fit(x_train, y_train)
    return grid


def cv_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 1,
) -> GridSearchCV:
    """
    Runs GridSearchCV for RandomForestRegressor.

    Args:
        x_train: Training feature matrix.
        y_train: Training target variable.
        param_grid: Parameter grid for GridSearchCV.
        cv: Number of cross-validation folds.
        n_jobs: Number of parallel jobs.
        verbose: Verbosity level for GridSearchCV.

    Returns:
        grid: Fitted GridSearchCV object.
    """
    print("🔎 Grid searching RandomForestRegressor...")
    rf: RandomForestRegressor = RandomForestRegressor(random_state=42, n_jobs=1)

    grid: GridSearchCV = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )

    grid.fit(x_train, y_train)
    return grid


def cv_lpr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: List[Dict[str, List[Any]]],
    cv: int,
    n_jobs: int,
    verbose: int = 1,
) -> GridSearchCV:
    """
    Runs GridSearchCV for LPR (kr='none').

    Args:
        x_train: Training feature matrix.
        y_train: Training target variable.
        param_grid: Parameter grid for GridSearchCV (where 'kr' must be ['none']).
        cv: Number of cross-validation folds.
        n_jobs: Number of parallel jobs.
        verbose: Verbosity level for GridSearchCV.

    Returns:
        grid: Fitted GridSearchCV object.

    Raises:
        ValueError: If param_grid['kr'] is not ['none'].
    """
    print("🔎 Grid searching LPR (using IndexAwareWrapper)...")

    for i in range(len(param_grid)):
        if not param_grid[i]["kr"] == ["none"]:
            raise ValueError("For cv_lpr, param_grid['kr'] must be ['none'].")

    # Pack dummy indices. They won't be used, but the wrapper expects them.
    dummy_indices: np.ndarray = np.arange(len(x_train))

    x_train_with_indices: np.ndarray = np.hstack(
        (x_train, dummy_indices.reshape(-1, 1))
    )

    estimator: IndexAwareRsklprWrapper = IndexAwareRsklprWrapper()

    grid: GridSearchCV = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )

    grid.fit(x_train_with_indices, y_train)
    return grid


def cv_rsklpr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 1,
) -> GridSearchCV:
    """
    Runs GridSearchCV for RSKLPR (kr != 'none').

    Args:
        x_train: Training feature matrix.
        y_train: Training target variable.
        param_grid: Parameter grid for GridSearchCV (where 'kr' must NOT be ['none']).
        cv: Number of cross-validation folds.
        n_jobs: Number of parallel jobs.
        verbose: Verbosity level for GridSearchCV.

    Returns:
        grid: Fitted GridSearchCV object.

    Raises:
        ValueError: If param_grid['kr'] contains 'none'.
    """
    print("🔎 Grid searching RSKLPR (using IndexAwareWrapper)...")

    if "none" in param_grid["kr"]:
        raise ValueError("For cv_rsklpr, param_grid['kr'] must NOT contain 'none'.")

    # Pack dummy indices. They won't be used, but the wrapper expects them.
    dummy_indices: np.ndarray = np.arange(len(x_train))
    x_train_with_indices = np.hstack((x_train, dummy_indices.reshape(-1, 1)))

    estimator: IndexAwareRsklprWrapper = IndexAwareRsklprWrapper()

    grid: GridSearchCV = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )

    grid.fit(x_train_with_indices, y_train)
    return grid


def cv_graph_lpr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: np.ndarray,
    airport_network: nx.Graph,
    original_idx_to_node_map: Dict[int, Optional[str]],
    param_grid: List[Dict[str, Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 1,
) -> GridSearchCV:
    """
    Runs GridSearchCV for Graph LPR (uses stateful graph kernel).

    Args:
        x_train: Training feature matrix.
        y_train: Training target variable.
        train_indices: 1D NumPy array of original indices for the training set.
        airport_network: The airport network graph.
        original_idx_to_node_map: Mapping from original index to node ID (airport code).
        param_grid: Parameter grid for GridSearchCV. The 'kp' key should contain lists where "GRAPH_KERNEL_PLACEHOLDER"
            indicates where to insert the graph kernel.
        cv: Number of cross-validation folds.
        n_jobs: Number of parallel jobs.
        verbose: Verbosity level for GridSearchCV.

    Returns:
        grid: Fitted GridSearchCV object.
    """
    print("🔎 Grid searching Graph LPR (using IndexAwareWrapper)...")

    # Pack the REAL indices
    x_train_with_indices: np.ndarray = np.hstack(
        (x_train, train_indices.reshape(-1, 1))
    )

    # Create the stateful graph kernel instance once. We pass a default graph_distance_scale, but GridSearchCV will
    # override it via the wrapper.
    graph_kernel_obj: IndexAwareGraphKernel = IndexAwareGraphKernel(
        airport_network=airport_network,
        original_idx_to_node_map=original_idx_to_node_map,
        graph_distance_scale=1.0,  # Default, will be overwritten
    )

    # Inject this instance into the param_grid's 'kp' list
    updated_param_grid = []
    params: Dict[str, Any]

    for params in param_grid:
        new_params: Dict[str, Any] = params.copy()

        # 'kp' in the input grid is a list of options for the kp_list
        # e.g., "kp": [ [k1, "PH"] ]
        kp_options: List[List[Callable | str]] = new_params.pop("kp")

        new_kp_list_of_lists = []
        kp_list_option: Callable

        for kp_list_option in kp_options:  # e.g., [k1, "PH"]
            new_kp_list: List[Callable] = []
            kernel: Callable | str

            for kernel in kp_list_option:  # e.g., k1, then "PH"
                if kernel == "GRAPH_KERNEL_PLACEHOLDER":
                    new_kp_list.append(graph_kernel_obj)
                else:
                    new_kp_list.append(kernel)

            new_kp_list_of_lists.append(new_kp_list)  # e.g., [[k1, obj]]

        new_params["kp"] = new_kp_list_of_lists  # e.g., "kp": [ [[k1, obj]] ]

        # 'graph_distance_scale' is already a top-level key. We leave it as-is. GridSearchCV will handle it.
        updated_param_grid.append(new_params)

    estimator: IndexAwareRsklprWrapper = IndexAwareRsklprWrapper()

    grid: GridSearchCV = GridSearchCV(
        estimator,
        updated_param_grid,  # This grid now has the kernel object
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )

    grid.fit(x_train_with_indices, y_train)
    return grid


def grid_search_multiple_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: np.ndarray,
    airport_network: nx.Graph,
    original_idx_to_node_map: Dict[int, Optional[str]],
    cv: int = 5,
    n_jobs: int = -1,
    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
) -> Dict[str, GridSearchCV]:
    results: Dict[str, GridSearchCV] = {}
    if param_grids is None:
        param_grids = {}

    results["linear"] = cv_linear(
        x_train=x_train,
        y_train=y_train,
        param_grid=param_grids.get("linear", {}),
        cv=cv,
        n_jobs=n_jobs,
    )

    results["random_forest"] = cv_random_forest(
        x_train=x_train,
        y_train=y_train,
        param_grid=param_grids.get("random_forest", {}),
        cv=cv,
        n_jobs=n_jobs,
    )

    results["lpr"] = cv_lpr(
        x_train=x_train,
        y_train=y_train,
        param_grid=[param_grids.get("lpr", {})],
        cv=cv,
        n_jobs=n_jobs,
    )

    results["graph_lpr"] = cv_graph_lpr(
        x_train=x_train,
        y_train=y_train,
        train_indices=train_indices,
        airport_network=airport_network,
        original_idx_to_node_map=original_idx_to_node_map,
        param_grid=[param_grids.get("graph_lpr", {})],
        cv=cv,
        n_jobs=n_jobs,
    )

    print("✅ Grid search complete for all models.")
    return results


def display_model_results(
    grid_search_results: Dict[str, GridSearchCV],
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """
    Calculates and plots R2, MAE, and RMSE for all models. Generates bar plots for comparison.
    Args:
        grid_search_results: Dictionary mapping model names to their fitted GridSearchCV objects.
        x_test: Test feature matrix.
        y_test: Test target variable.
        test_indices: 1D NumPy array of original indices for the test set.
    """
    print("\n📊 Displaying results for all evaluated models...")

    num_models: int = len(grid_search_results)

    if num_models == 0:
        print("No models to display.")
        raise ValueError("grid_search_results is empty.")

    model_metrics = []

    for model_name, grid_search in grid_search_results.items():
        print(f"\n--- Processing: {model_name} ---")
        print(f"Best CV Score (neg_mean_squared_error): {grid_search.best_score_:.4f}")

        best_estimator = grid_search.best_estimator_

        # Prepare test data
        if model_name == "graph_lpr" or model_name == "lpr":
            x_test_prepared = np.hstack((x_test, test_indices.reshape(-1, 1)))
        else:
            x_test_prepared = x_test

        y_pred = best_estimator.predict(x_test_prepared)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Test Set R²:   {r2:.4f}")
        print(f"Test Set MAE:  {mae:.4f}")
        print(f"Test Set RMSE: {rmse:.4f}")

        model_metrics.append(
            {"model_name": model_name, "r2": r2, "mae": mae, "rmse": rmse}
        )

    # Create Summary Bar Plots
    if not model_metrics:
        print("No metrics to plot in summary.")
        return

    metrics_df = pd.DataFrame(model_metrics).set_index("model_name")

    fig_bars, (ax_r2, ax_mae, ax_rmse) = plt.subplots(3, 1, figsize=(10, 15))
    fig_bars.suptitle(
        "Test Set Metric Comparison (Airport Delay Simulation)", fontsize=16, y=1.02
    )

    # R² (Higher is better)
    metrics_df["r2"].sort_values(ascending=False).plot(
        kind="barh", ax=ax_r2, colormap="summer", title="R² Score (Higher is Better)"
    )
    ax_r2.set_xlabel("R² Score")
    ax_r2.grid(axis="x", linestyle="--", alpha=0.7)

    # MAE (Lower is better)
    metrics_df["mae"].sort_values(ascending=True).plot(
        kind="barh",
        ax=ax_mae,
        colormap="autumn",
        title="Mean Absolute Error (Lower is Better)",
    )
    ax_mae.set_xlabel("Mean Absolute Error (Synthetic Delay)")
    ax_mae.grid(axis="x", linestyle="--", alpha=0.7)

    # RMSE (Lower is better)
    metrics_df["rmse"].sort_values(ascending=True).plot(
        kind="barh",
        ax=ax_rmse,
        colormap="winter",
        title="Root Mean Squared Error (Lower is Better)",
    )
    ax_rmse.set_xlabel("Root Mean Squared Error (Synthetic Delay)")
    ax_rmse.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("airport_model_comparison_barchart.png", dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    airport_network: nx.Graph
    df_features: pd.DataFrame
    x_full: np.ndarray
    y_full: np.ndarray
    airport_network, df_features, x_full, y_full = load_airport_data_and_graph(
        count_weighted_delay=True
    )

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    original_idx_to_node_map: Dict[int, Optional[str]]

    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_indices,
        test_indices,
        original_idx_to_node_map,
    ) = prepare_ml_data(
        df_features=df_features,
        x_features=x_full,
        y_delay=y_full,
        test_size=0.2,
        random_state=42,
    )

    plot_airport_network_map(airport_network=airport_network, save_plot=True)

    plot_graph_kernel_similarity(
        airport_network=airport_network,
        reference_node_id="ATL",
        graph_distance_scale=2.0,
        original_idx_to_node_map=original_idx_to_node_map,
        save_plot=True,
    )

    plot_graph_kernel_similarity(
        airport_network=airport_network,
        reference_node_id="HOU",
        graph_distance_scale=2.0,
        original_idx_to_node_map=original_idx_to_node_map,
        save_plot=True,
    )

    # param grids for each model
    cv: int = 10
    n_jobs = 1

    param_grids: Dict[str, Dict[str, Any]] = {
        "linear": {
            "fit_intercept": [True, False],
            "positive": [False, True],  # enforce non-negative coefficients
        },
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "max_features": [1.0, "sqrt"],
            "min_samples_leaf": [1, 2, 4],
        },
        "lpr": {
            "size_neighborhood": list(range(7, 105, 4)),
            "degree": [0, 1, 2],
            "kp": [
                [tricube_normalized_metric],
                [laplacian_normalized_metric],
            ],
            "kr": ["none"],
            "metric_x": ["mahalanobis"],
        },
        "graph_lpr": {
            "size_neighborhood": list(range(7, 35, 4)),
            "degree": [1],
            "kp": [
                [tricube_normalized_metric, "GRAPH_KERNEL_PLACEHOLDER"],
            ],
            "kr": ["conden"],
            "metric_x": ["mahalanobis"],
            "graph_distance_scale": np.linspace(0.1, 3.0, num=29),
        },
    }

    # --- 5. RUN EXPERIMENT ---
    start_time: float = time.time()

    all_grid_results = grid_search_multiple_models(
        x_train=x_train,
        y_train=y_train,
        train_indices=train_indices,
        airport_network=airport_network,
        original_idx_to_node_map=original_idx_to_node_map,
        cv=cv,
        n_jobs=n_jobs,
        param_grids=param_grids,
    )
    end_time: float = time.time()

    print(f"\n--- Total Grid Search Time: {end_time - start_time:.2f} seconds ---")

    display_model_results(
        grid_search_results=all_grid_results,
        x_test=x_test,
        y_test=y_test,
        test_indices=test_indices,
    )


if __name__ == "__main__":
    main()
