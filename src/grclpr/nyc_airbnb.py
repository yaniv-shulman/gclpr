import math
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple, List, Any

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rsklpr.kernels import tricube_normalized_metric, laplacian_normalized_metric
from rsklpr.rsklpr import Rsklpr
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def load_subway_graph() -> nx.Graph:
    print("🚇 Loading subway graph...")
    graph: nx.Graph = nx.read_graphml(Path("/home/yaniv/data/datasets/raw/gtfs_subway") / "nyc_subway_graph.graphml")
    return graph


def plot_subway_graph(subway_graph: nx.Graph) -> None:
    print("🎨 Visualizing subway graph...")
    start_viz_time = time.time()
    # Extract positions (longitude, latitude) from node attributes"
    # Note: Plotting uses (x, y), which typically corresponds to (lon, lat)"
    pos = {
        node: (data["lon"], data["lat"])
        for node, data in subway_graph.nodes(data=True)
        if "lon" in data and "lat" in data
    }

    # Create a figure"
    plt.figure(figsize=(8, 8))

    # Draw the graph using the extracted positions"
    # Make nodes small and remove labels for clarity on a large graph"
    nx.draw(subway_graph, pos, node_size=5, width=0.5, with_labels=False, node_color="blue", edge_color="gray")

    plt.title("NYC Subway Network Connectivity")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")  # Ensure aspect ratio is correct for geo-coordinates"
    plt.show()

    end_viz_time = time.time()
    print(f"Graph visualization displayed in {end_viz_time - start_viz_time:.2f} seconds.")


def haversine(lat1, lon1, lat2, lon2, radius=6371):
    """
    Calculate the great circle distance in kilometers between two points
    or arrays of points on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: Latitude and Longitude of first point(s).
        lat2, lon2: Latitude and Longitude of second point(s).
        radius: Radius of the sphere (default is Earth's mean radius in km).

    Returns:
        Distance(s) in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = radius * c
    return distance


def load_nyc_airbnb_dataset() -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]
):
    print("📥 Loading and preparing NYC Airbnb dataset...")
    airbnb_df = pd.read_csv(Path("/home/yaniv/data/datasets/raw/air_bnb_nyc_listings") / "listings.csv")

    # Basic Feature Selection and Cleaning
    features = [
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

    airbnb_df = airbnb_df[features].copy()

    # Clean price column (remove '$', ',', convert to numeric)
    airbnb_df["price"] = airbnb_df["price"].replace({"\\$": "", ",": ""}, regex=True)
    airbnb_df["price"] = pd.to_numeric(airbnb_df["price"], errors="coerce")

    # Handle missing values & filter unreasonable data
    airbnb_df.dropna(subset=["price"], inplace=True)
    airbnb_df = airbnb_df[airbnb_df["price"] > 0]  # Price must be positive

    # Fill other NaNs simply (can be improved)
    cols_to_fill = [
        "host_listings_count",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
    ]

    median_values = airbnb_df[cols_to_fill].median()
    airbnb_df.fillna(median_values, inplace=True)

    # Log transform price
    airbnb_df["log_price"] = np.log1p(airbnb_df["price"])
    y_full_log = airbnb_df["log_price"].values

    # One-hot encode categorical features
    airbnb_df = pd.get_dummies(airbnb_df, columns=["room_type"], drop_first=True)

    # Define final feature list (ensure lat/lon are last for consistency if needed)
    main_features = [col for col in airbnb_df.columns if col not in ["price", "log_price", "latitude", "longitude"]]
    geo_features = ["latitude", "longitude"]
    all_features = main_features + geo_features
    X_full = airbnb_df[all_features].values
    print(f"Features being used: {all_features}")
    num_main_features = len(main_features)

    # Store listing coordinates separately for kernel factory
    listing_coords_all: pd.Series = airbnb_df[geo_features]

    # Split data (using log price as target)
    x_train, x_test, y_train_log, y_test_log = train_test_split(X_full, y_full_log, test_size=0.2, random_state=42)

    # Store original indices of test set for mapping later if needed
    train_indices, test_indices = train_test_split(np.arange(len(X_full)), test_size=0.2, random_state=42)

    # Scale numerical features (excluding lat/lon and one-hot encoded)

    # Identify numerical columns to scale (adjust based on your actual columns after get_dummies)
    numerical_cols = [
        "host_listings_count",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
    ]

    # Find the indices of these columns in the final feature list
    numerical_indices = [all_features.index(col) for col in numerical_cols if col in all_features]

    scaler = StandardScaler()
    x_train[:, numerical_indices] = scaler.fit_transform(x_train[:, numerical_indices])
    x_test[:, numerical_indices] = scaler.transform(x_test[:, numerical_indices])

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    print(f"Data prepared: {len(y_train_log)} training samples, {len(y_test_log)} testing samples.")
    return x_train, x_test, y_train_log, y_test_log, train_indices, test_indices, listing_coords_all


def calculate_unified_index_to_station_map(
    subway_graph: nx.Graph, listing_coords_all: pd.Series, max_station_dist_km: float = 1.0
) -> Dict[int, Optional[str]]:
    station_data = [
        (data["lat"], data["lon"], station_id)
        for station_id, data in subway_graph.nodes(data=True)
        if "lat" in data and "lon" in data
    ]

    # Create NumPy array of coordinates [N_stations, 2]
    station_coords_np = np.array([[lat, lon] for lat, lon, _ in station_data])

    # Create list of corresponding station IDs
    station_ids_list = [sid for _, _, sid in station_data]

    print(f"Extracted coordinates for {len(station_ids_list)} stations.")
    print("Pre-calculating unified index-to-station map for ALL data...")

    station_tree = KDTree(station_coords_np)
    original_idx_to_station_map: Dict[int, Optional[str]] = {}
    station_id_array = np.array(station_ids_list)
    approx_radius_degrees = max_station_dist_km / 111.0
    indices_list = station_tree.query_ball_point(listing_coords_all.values, r=approx_radius_degrees)

    for i, listing_coord in enumerate(listing_coords_all.values):  # i is the original index
        nearby_indices = indices_list[i]
        nearest_station_id = None
        if nearby_indices:
            nearby_station_coords = station_coords_np[nearby_indices]
            distances_km = haversine(
                listing_coord[0], listing_coord[1], nearby_station_coords[:, 0], nearby_station_coords[:, 1]
            )
            min_dist_idx = np.argmin(distances_km)
            if distances_km[min_dist_idx] <= max_station_dist_km:
                nearest_station_original_idx = nearby_indices[min_dist_idx]
                nearest_station_id = station_id_array[nearest_station_original_idx]
        original_idx_to_station_map[i] = nearest_station_id

    print("Unified map created.")
    return original_idx_to_station_map


def subway_shortest_path_kernel_factory(
    graph: nx.Graph,
    # Map from ORIGINAL dataset index (0 to N-1) to nearest station ID
    original_idx_to_station_map: Dict[int, Optional[str]],
    # NEW: Map from X_train index (0...len(X_train)-1) back to original index
    train_set_original_indices: np.ndarray,
    # This array maps X_pred index (0...len(X_pred)-1) back to original index
    predict_set_original_indices: np.ndarray,
    distance_scale: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]:
    """
    Factory using original dataset index maps for robust station lookup,
    matching the Rsklpr kp interface.
    """
    if not isinstance(predict_set_original_indices, np.ndarray) or predict_set_original_indices.ndim != 1:
        raise TypeError("predict_set_original_indices must be a 1D NumPy array.")

    if not isinstance(train_set_original_indices, np.ndarray) or train_set_original_indices.ndim != 1:
        raise TypeError("train_set_original_indices must be a 1D NumPy array.")

    _map = original_idx_to_station_map
    _pred_indices = predict_set_original_indices
    _train_indices = train_set_original_indices

    def _kernel(
        x_0: np.ndarray,
        x_neighbors: np.ndarray,
        dist_x_neighbors: np.ndarray,
        index_x_0: int,  # Index of x_0 WITHIN the current PREDICTION batch
        indices_neighbors: np.ndarray,  # Indices of neighbors in the FITTED (training) data
    ) -> np.ndarray:

        # Map predict index back to original index
        try:
            target_original_idx = _pred_indices[index_x_0]
        except IndexError:
            warnings.warn(f"index_x_0 ({index_x_0}) out of bounds for predict_set_original_indices.")
            target_original_idx = -1  # Invalid index

        target_station = _map.get(target_original_idx)

        # Map neighbor indices back to original indices
        # indices_neighbors is (1, K), get the 1D array of relative indices
        neighbor_relative_indices = indices_neighbors[0]

        eps: float = 0.01
        similarity_scores = np.full(len(neighbor_relative_indices), fill_value=eps)

        if target_station is None or target_station not in graph:
            return similarity_scores.reshape(1, -1)

        # Iterate over the relative neighbor indices
        for j, neighbor_relative_idx in enumerate(neighbor_relative_indices):

            # Map relative train index back to original index
            try:
                neighbor_original_idx = _train_indices[neighbor_relative_idx]
                neighbor_station = _map.get(neighbor_original_idx)
            except IndexError:
                warnings.warn(
                    f"neighbor_relative_idx ({neighbor_relative_idx}) out of bounds for train_set_original_indices."
                )
                continue
            except KeyError:
                neighbor_station = None  # Handled by _map.get

            if neighbor_station is None or neighbor_station not in graph or neighbor_station == target_station:
                if neighbor_station == target_station and target_station is not None:
                    similarity_scores[j] = 1.0
                continue  # Leaves eps otherwise

            try:
                path_length = nx.shortest_path_length(graph, source=target_station, target=neighbor_station)
                if path_length >= 0:
                    similarity_scores[j] = math.exp(-path_length / distance_scale)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass  # Already initialized to eps

        return similarity_scores.reshape(1, -1)

    return _kernel


def _find_test_indices(test_indices, train_indices, station_map, graph, max_search=1000):
    """
    # --- Helper function to find test pairs ---
    Searches for pairs of (relative_test_idx, relative_train_idx)
    that match our test criteria.
    """
    found_same = None
    found_distance = None

    # Limit search to avoid hanging on large datasets
    search_limit_test = min(max_search, len(test_indices))
    search_limit_train = min(max_search, len(train_indices))

    print(f"Searching first {search_limit_test} test points against first {search_limit_train} train points...")

    for i in range(search_limit_test):
        if found_same and found_distance:
            break  # Found both

        orig_idx_test = test_indices[i]
        station_test = station_map.get(orig_idx_test)

        if not station_test or station_test not in graph:
            continue

        for j in range(search_limit_train):
            orig_idx_train = train_indices[j]
            station_train = station_map.get(orig_idx_train)

            if not station_train or station_train not in graph:
                continue

            # 1. Check for "same station"
            if not found_same and station_test == station_train:
                found_same = (i, j)  # (relative_test_idx, relative_train_idx)

            # 2. Check for "distance"
            if not found_distance and station_test != station_train:
                try:
                    dist = nx.shortest_path_length(graph, station_test, station_train)
                    if dist > 0:  # Found a valid path
                        found_distance = (i, j)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass  # No path, keep looking

            if found_same and found_distance:
                break  # Inner loop

    return {"same_station": found_same, "distance": found_distance}


def _run_kernel_test(test_name, kernel_func, test_idx_pair, test_indices, train_indices, station_map, graph, scale):
    """
    # --- Helper function to run a single test ---
    Args:
        test_name:
        kernel_func:
        test_idx_pair:
        test_indices:
        train_indices:
        station_map:
        graph:
        scale:

    Returns:

    """

    print(f"\n--- {test_name} ---")

    if test_idx_pair is None:
        print("SKIPPED: Could not find a suitable pair of points in the search limit.")
        return

    target_idx_in_test_set, neighbor_idx_in_train_set = test_idx_pair

    try:
        # 1. Manual Calculation
        target_original_idx = test_indices[target_idx_in_test_set]
        neighbor_original_idx = train_indices[neighbor_idx_in_train_set]
        station_1 = station_map.get(target_original_idx)
        station_2 = station_map.get(neighbor_original_idx)

        print(f"Target (Test Idx {target_idx_in_test_set}):   Orig Idx {target_original_idx} -> Station: {station_1}")
        print(
            f"Neighbor (Train Idx {neighbor_idx_in_train_set}): Orig Idx {neighbor_original_idx} -> Station: {station_2}"
        )

        expected_score = 0.01  # Default epsilon

        if station_1 and station_2 and station_1 in graph and station_2 in graph:
            if station_1 == station_2:
                expected_score = 1.0
                print("Stations are the same. Expected score: 1.0")
            else:
                dist = nx.shortest_path_length(graph, station_1, station_2)
                expected_score = math.exp(-dist / scale)
                print(f"Distance: {dist}. Expected score: exp(-{dist}/{scale}) = {expected_score:.4f}")
        else:
            print(f"One station is invalid or None. Expected score: {expected_score}")

        # 2. Kernel Call
        kernel_output = kernel_func(
            x_0=None,  # Ignored by kernel
            x_neighbors=None,  # Ignored by kernel
            dist_x_neighbors=None,  # Ignored by kernel
            index_x_0=target_idx_in_test_set,
            indices_neighbors=np.array([[neighbor_idx_in_train_set]]),
        )
        kernel_score = kernel_output[0, 0]
        print(f"Kernel Output Score: {kernel_score:.4f}")

        # 3. Verify
        assert np.isclose(kernel_score, expected_score), "Kernel score does not match manual calculation!"
        print("✅ Success: Kernel output matches manual calculation.")

    except Exception as e:
        print(f"❌ FAILED: An error occurred during the test: {e}")


def sanity_test_path_exists_and_kernel(
    subway_graph: nx.Graph,
    original_idx_to_station_map: Dict[int, str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    best_distance_scale: float = 2.0,
):

    try:
        print("🏭 Generating kernel for testing...")
        test_kernel = subway_shortest_path_kernel_factory(
            graph=subway_graph,
            original_idx_to_station_map=original_idx_to_station_map,
            train_set_original_indices=train_indices,
            predict_set_original_indices=test_indices,
            distance_scale=best_distance_scale,
        )

        # 2. Find test pairs
        print("🔍 Searching for test point pairs...")
        test_pairs = _find_test_indices(test_indices, train_indices, original_idx_to_station_map, subway_graph)

        # 3. Run Test 1: Same Station
        _run_kernel_test(
            "Test 1: Same Station",
            test_kernel,
            test_pairs["same_station"],
            test_indices,
            train_indices,
            original_idx_to_station_map,
            subway_graph,
            best_distance_scale,
        )

        # 4. Run Test 2: Some Distance

        _run_kernel_test(
            "Test 2: Some Distance",
            test_kernel,
            test_pairs["distance"],
            test_indices,
            train_indices,
            original_idx_to_station_map,
            subway_graph,
            best_distance_scale,
        )

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def plot_kernel_similarity_from_reference(
    subway_graph: nx.Graph,
    reference_station_id: str,
    distance_scale: float,
    target_nodes_list: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    save_plot: bool = False,
    zoom_graph_radius: Optional[int] = None,  # <-- NEW PARAMETER
) -> None:
    """
    Visualizes the kernel similarity (based on shortest path) from a single
    reference station to all other stations in the graph.

    This uses your existing loaded 'subway_graph'.

    Args:
        subway_graph: The full NYC subway graph (loaded from your function).
        reference_station_id: The 'id' (name) of the station to use as the source.
                                (e.g., "Times Sq-42 St")
        distance_scale: The distance_scale (sigma) used in your kernel
                          (exp(-path / scale)).
        target_nodes_list: Optional. A list of station IDs to plot.
                             If None, plots the entire graph.
        ax: Optional. A matplotlib axes object to plot on.
        save_plot: If True, saves the plot as 'subway_kernel_similarity.png'.
        zoom_graph_radius: Optional. If set (e.g., to 5), the plot will only
                           include nodes within this many stops (graph distance)
                           of the reference_station_id.
    """
    print(f"🎨 Visualizing kernel similarity from '{reference_station_id}'...")

    correct_station_id = None

    # Find the node ID for the given station name
    for node_id, data in subway_graph.nodes(data=True):
        if data.get("name") == reference_station_id:
            correct_station_id = node_id
            break  # Found it

    if correct_station_id:
        print(f"Found it! The ID for '{reference_station_id}' is: '{correct_station_id}'")
    else:
        print(f"❌ Error: Could not find station with name '{reference_station_id}'.")
        # Check if the ID itself was passed
        if reference_station_id in subway_graph:
            print(f"Using '{reference_station_id}' as a direct node ID.")
            correct_station_id = reference_station_id
        else:
            print("Hint: Node names might be station IDs. Check a name by running: print(list(subway_graph.nodes(data='name'))[0])")
            return

    # 2. Validate reference station
    if correct_station_id not in subway_graph:
        print(f"❌ Error: Reference station '{correct_station_id}' not found in the full graph.")
        return

    # 3. Calculate *all* shortest path lengths from the reference node
    #    This is now required *first* to enable the zoom feature.
    try:
        path_lengths_map = nx.shortest_path_length(subway_graph, source=correct_station_id)
    except nx.NodeNotFound:
        print(f"Error: Reference station '{correct_station_id}' not found in graph during path calculation.")
        return

    # --- NEW/MODIFIED: Node Selection Logic ---
    
    # 1. Determine which nodes to plot
    if zoom_graph_radius is not None:
        print(f"Applying zoom: graph radius <= {zoom_graph_radius} stops.")
        # Get all nodes within the radius from our pre-calculated map
        plot_node_ids = [
            node_id for node_id, path_length in path_lengths_map.items() 
            if path_length <= zoom_graph_radius
        ]
        # Ensure the reference station itself is included (path_length=0)
        if correct_station_id not in plot_node_ids:
             plot_node_ids.append(correct_station_id)
        print(f"Found {len(plot_node_ids)} nodes within radius.")
    
    elif target_nodes_list:
        # User provided a specific list, but no zoom
        plot_node_ids = list(set(target_nodes_list + [correct_station_id]))
    
    else:
        # Default: plot all nodes
        plot_node_ids = list(subway_graph.nodes())

    # 2. Now, create the subgraph based on the chosen node list
    try:
        valid_plot_nodes = [n for n in plot_node_ids if n in subway_graph]
        plot_graph = subway_graph.subgraph(valid_plot_nodes)
    except nx.NetworkXError as e:
        print(f"Error creating subgraph. Are all target nodes in the main graph? {e}")
        return
    # --- End New/Modified Block ---


    # 3. Get positions for the nodes we intend to plot
    #    This uses the 'lon' and 'lat' attributes from your loaded graph
    pos = {
        node: (data["lon"], data["lat"])
        for node, data in plot_graph.nodes(data=True) # This now uses the new subgraph
        if "lon" in data and "lat" in data
    }

    # Filter node list to only those we have positions for
    nodes_with_pos = list(pos.keys())
    if not nodes_with_pos:
        print("Error: No nodes in the target set have (lon, lat) data.")
        return

    if correct_station_id not in pos:
        print(f"Warning: Reference station '{correct_station_id}' has no (lon, lat) data and will not be visible.")

    # Get data limits for a tight axis
    lons = [coord[0] for coord in pos.values()]
    lats = [coord[1] for coord in pos.values()]
    
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    
    # Add a small buffer (e.g., 2% or 5% if zoomed)
    buffer_pct = 0.05 if zoom_graph_radius is not None else 0.02
    lon_buffer = (lon_max - lon_min) * buffer_pct
    lat_buffer = (lat_max - lat_min) * buffer_pct
    
    lim_lon = (lon_min - lon_buffer, lon_max + lon_buffer)
    lim_lat = (lat_min - lat_buffer, lat_max + lat_buffer)


    # 5. Calculate kernel similarity scores for each node we are plotting
    node_scores = []
    eps = 0.0  # Use 0.0 for unreachable nodes for clearer visualization

    for node in nodes_with_pos:
        if node == correct_station_id:
            score = 1.0
        else:
            # Get path length, default to -1 if no path exists
            path_length = path_lengths_map.get(node, -1) # Use the full map

            if path_length == -1:  # No path
                score = eps
            else:
                # This is your kernel similarity formula
                score = math.exp(-path_length / distance_scale)

        node_scores.append(score)

    # 6. Setup plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # 7. Draw the graph

    # Draw all edges in the subgraph
    nx.draw_networkx_edges(plot_graph, pos, ax=ax, edge_color="#AAAAAA", width=0.5)

    # Draw the nodes, colored by their similarity score
    # --- MODIFIED: Adjust node size if zoomed ---
    node_size = 50 if zoom_graph_radius is not None else 15
    
    nodes_collection = nx.draw_networkx_nodes(
        plot_graph,
        pos,
        ax=ax,
        nodelist=nodes_with_pos,
        node_color=node_scores,
        node_size=node_size,
        cmap=plt.cm.viridis,
        vmin=0.0,
        vmax=1.0,
    )

    # Highlight the reference node
    if correct_station_id in pos:
        # --- MODIFIED: Adjust node size if zoomed ---
        ref_node_size = 100 if zoom_graph_radius is not None else 50

        nx.draw_networkx_nodes(
            plot_graph,
            pos,
            ax=ax,
            nodelist=[correct_station_id],
            node_color="red",  # Highlight color
            node_size=ref_node_size,
            edgecolors="black",  # Border for visibility
        )

    # 8. Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, orientation="vertical", pad=0.02)
    
    cbar.set_label(
        f"Kernel Similarity (exp(-path_length / {distance_scale:.1f}))", 
        rotation=270, 
        labelpad=15,
        fontsize=10
    )
    cbar.ax.tick_params(labelsize=8)

    # Clean up aesthetics for paper
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)

    # Set tight limits *before* setting axis to equal
    ax.set_xlim(lim_lon)
    ax.set_ylim(lim_lat)

    ax.axis("equal")
    
    ax.tick_params(axis='both', which='major', labelsize=8)

    if save_plot:
        # Add bbox_inches='tight' for paper-ready output
        filename = f"subway_kernel_similarity_zoomed_{zoom_graph_radius}.png" if zoom_graph_radius is not None else "subway_kernel_similarity.png"
        plt.savefig(
            filename, 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0.05
        )
        print(f"Plot saved as '{filename}'.")
    
    plt.show()


class GrcLprSubwayWrapper(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for Rsklpr that uses an index-dependent subway kernel.
    This wrapper expects the *last column* of X to be the original dataset index.
    """

    def __init__(
        self,
        size_neighborhood: int = 50,
        degree: int = 1,
        distance_scale: float = 1.0,
        kp: Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray] = tricube_normalized_metric,        metric_x: str = "minkowski",
        metric_x_params: Optional[Dict[str, Any]] = None,
        kr: str = "none",
        # --- Global data (must be passed during init) ---
        subway_graph: nx.Graph = None,
        original_idx_to_station_map: Dict[int, Optional[str]] = None,
    ) -> None:
        """
        Initializes the wrapper with both tunable hyperparameters and global data.

        Args:
            size_neighborhood: The size of the neighborhood for Rsklpr.
            degree: The degree parameter for Rsklpr.
            distance_scale: The distance scale for the subway kernel (tunable).
            kp: The kernel function for predictors in Rsklpr.
            metric_x: The metric for Rsklpr.
            kr: The kr parameter for Rsklpr.
            subway_graph: The subway graph (networkx Graph).
            original_idx_to_station_map: Mapping from original dataset indices to station IDs.
        """
        self.size_neighborhood: int = size_neighborhood
        self.degree: int = degree
        self.distance_scale: float = distance_scale  # Tunable
        self.kp = kp
        self.metric_x: str = metric_x
        self.metric_x_params: Optional[Dict[str, Any]] = metric_x_params
        self.kr: str = kr

        # Store global data
        self.subway_graph = subway_graph
        self.original_idx_to_station_map = original_idx_to_station_map

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GrcLprSubwayWrapper":
        """
        Stores the training data and its original indices. X is assumed to have features in X[:, :-1] and
        original indices in X[:, -1].

        Args:
            X: Training data with original indices in X[:, -1].
            y: Target values.

        Returns:
            self
        """
        # Check that global data was provided
        if self.subway_graph is None or self.original_idx_to_station_map is None:
            raise ValueError("Wrapper must be initialized with subway_graph and original_idx_to_station_map")

        # Separate features from indices
        self.train_original_indices_ = X[:, -1].astype(int)
        self.X_fit_ = X[:, :-1]  # Features
        self.y_fit_ = y

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Creates the kernel, instantiates the model, fits it to the stored training data, and predicts on the new X.
        The kernel needs to know the original indices of both training and prediction sets, this is a bit more involved
        due to how GridSearchCV shuffles data.

        Args:
            X: New data to predict on, with original indices in X[:, -1].

        Returns:
            Predictions as a NumPy array.
        """
        # Get original indices of the prediction set (e.g., validation fold)
        predict_original_indices: np.ndarray = X[:, -1].astype(int)
        X_predict_features: np.ndarray = X[:, :-1]  # Features

        # Create the custom subway kernel
        subway_kernel: Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray] = (
            subway_shortest_path_kernel_factory(
                graph=self.subway_graph,
                original_idx_to_station_map=self.original_idx_to_station_map,
                train_set_original_indices=self.train_original_indices_,  # From fit()
                predict_set_original_indices=predict_original_indices,  # From predict()
                distance_scale=self.distance_scale,
            )
        )

        # Create the full kernel list
        kp_list: List[Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]] = [
            self.kp,
            subway_kernel,
        ]

        # Create, fit, and predict using the *real* Rsklpr model
        # This is inefficient, but required by how GridSearchCV works
        # with this stateful kernel.
        model: Rsklpr = Rsklpr(
            size_neighborhood=self.size_neighborhood,
            degree=self.degree,
            kp=kp_list,
            kr=self.kr,
            metric_x=self.metric_x,
            metric_x_params=self.metric_x_params,
            suppress_warnings=True,
        )

        # Fit on the stored training data
        model.fit(x=self.X_fit_, y=self.y_fit_)

        # Predict on the new data
        return model.predict(x=X_predict_features)


class RsklprWrapper(BaseEstimator, RegressorMixin):
    """
    A generic scikit-learn wrapper for Rsklpr to make it compatible
    with GridSearchCV, allowing hyperparameters (like the `kp` kernel list)
    to be searched.
    """

    def __init__(
        self,
        size_neighborhood: int = 50,
        degree: int = 1,
        kp: List[
            Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]
        ] = [laplacian_normalized_metric],
        kr: str = "none",
        metric_x: str = "minkowski",
        metric_x_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        This wrapper is needed to make Rsklpr compatible with scikit-learn's GridSearchCV

        Args:
            size_neighborhood: Number of neighbors to consider.
            degree: Degree of the local polynomial.
            kp: List of kernel functions for the predictors.
            kr: Kernel function for the responses.
            metric_x: Distance metric for finding neighbors.
        """
        self.size_neighborhood = size_neighborhood
        self.degree = degree
        self.kp = kp
        self.kr = kr
        self.metric_x = metric_x
        self.metric_x_params = metric_x_params

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRsklprWrapper":
        # store training features & target
        self.x_fit_ = X
        self.y_fit_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Build Rsklpr with only the base kernel

        model: Rsklpr = Rsklpr(
            size_neighborhood=self.size_neighborhood,
            degree=self.degree,
            kp=self.kp,
            kr=self.kr,
            metric_x=self.metric_x,
            metric_x_params=self.metric_x_params,
            suppress_warnings=True,  # Add warning suppression
        )

        model.fit(x=self.x_fit_, y=self.y_fit_)
        return model.predict(x=X)


def cv_knn(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose=0,
) -> GridSearchCV:
    """Runs GridSearchCV for KNeighborsRegressor."""
    print("🔎 Grid searching KNeighborsRegressor...")
    knn = KNeighborsRegressor()
    knn_grid = GridSearchCV(knn, param_grid, cv=cv, n_jobs=n_jobs, scoring="neg_mean_squared_error", verbose=verbose)
    knn_grid.fit(x_train, y_train_log)
    return knn_grid


def cv_linear(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose=0,
) -> GridSearchCV:
    """Runs GridSearchCV for LinearRegression."""
    print("🔎 Grid searching LinearRegression...")
    lin = LinearRegression()
    lin_grid = GridSearchCV(lin, param_grid, cv=cv, n_jobs=n_jobs, scoring="neg_mean_squared_error", verbose=verbose)
    lin_grid.fit(x_train, y_train_log)
    return lin_grid


def cv_poly_ridge(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose=0,
) -> GridSearchCV:
    """Runs GridSearchCV for PolynomialFeatures + Ridge."""
    print("🔎 Grid searching Quadratic (Polynomial + Ridge)...")
    quad_pipe = Pipeline([("poly", PolynomialFeatures(include_bias=False)), ("ridge", Ridge())])
    quad_grid = GridSearchCV(
        quad_pipe, param_grid, cv=cv, n_jobs=n_jobs, scoring="neg_mean_squared_error", verbose=verbose
    )
    quad_grid.fit(x_train, y_train_log)
    return quad_grid


def cv_lpr(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose=0,
) -> GridSearchCV:
    """Runs GridSearchCV for SimpleRsklprWrapper (LPR)."""
    print("🔎 Grid searching LPR...")
    lpr = RsklprWrapper()

    simple_grid = GridSearchCV(
        lpr,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )

    simple_grid.fit(x_train, y_train_log)
    return simple_grid


def cv_rsklpr(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose=0,
) -> GridSearchCV:
    """Runs GridSearchCV for SimpleRsklprWrapper (RSKLPR)."""
    print("🔎 Grid searching RSKLPR...")
    lpr = RsklprWrapper()

    simple_grid = GridSearchCV(
        lpr,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )

    simple_grid.fit(x_train, y_train_log)
    return simple_grid


def cv_grclpr_subway(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    train_indices: np.ndarray,
    subway_graph: nx.Graph,
    original_idx_to_station_map: Dict[int, Optional[str]],
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose=0,
) -> GridSearchCV:
    """
    Runs GridSearchCV for the index-aware SubwayRsklprWrapper.
    This was formerly grid_search_cv.
    """
    print("📦 Packing original indices into X_train for GRC-LPR Subway...")

    # Stack the original indices as the last column of x_train
    x_train_with_indices: np.ndarray = np.hstack(
        (x_train, train_indices.reshape(-1, 1))
    )  # train_indices must be a column vector

    print(f"Original x_train shape: {x_train.shape}")
    print(f"New X_train_with_indices shape: {x_train_with_indices.shape}")

    # Define the Estimator and Parameter Grid
    print("⚙️ Defining model and search grid for GRC-LPR Subway...")

    # Instantiate the wrapper with the *global, non-tunable* data
    grclpr_subway_estimator: GrcLprSubwayWrapper = GrcLprSubwayWrapper(
        subway_graph=subway_graph,
        original_idx_to_station_map=original_idx_to_station_map,
    )

    # Set up and run GridSearchCV
    # Note: We run this on 'X_train_with_indices'
    grid_search_subway: GridSearchCV = GridSearchCV(
        grclpr_subway_estimator,
        param_grid=param_grid,  # Use the passed param_grid
        cv=cv,  # Use the passed cv
        n_jobs=n_jobs,  # Use the passed n_jobs
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )

    print("🚀 Starting GridSearchCV for GRC-LPR Subway...")

    # Fit on the data with packed indices
    grid_search_subway.fit(x_train_with_indices, y_train_log)
    return grid_search_subway


def grid_search_multiple_models(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    train_indices: np.ndarray,
    subway_graph: nx.Graph,
    original_idx_to_station_map: Dict[int, Optional[str]],
    cv: int = 5,
    n_jobs: int = -1,
    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
) -> Dict[str, GridSearchCV]:
    """
    Run GridSearchCV for a set of different regressors by calling
    dedicated CV functions for each model. Models included:
      - KNeighborsRegressor (local)
      - KernelRidge
      - LinearRegression (global linear)
      - Polynomial (degree=2) + Ridge (global quadratic)
      - Simple RSKLPR (standard LPR implemented with Rsklpr)
      - Subway RSKLPR (index-aware kernel)

    Args:
        x_train: Features for training (NxD)
        y_train_log: Targets (N,)
        train_indices: Original dataset indices mapped to rows in x_train
        subway_graph: NetworkX graph for subway-aware kernel
        original_idx_to_station_map: Mapping from original index -> nearest station id
        cv: Number of CV folds
        n_jobs: number of parallel jobs for GridSearchCV
        param_grids: Optional dict specifying parameter grids for each named model.
                     Example: {"knn": {"n_neighbors": [5, 10]}, "linear": {}}

    Returns:
        A dict mapping model name -> fitted GridSearchCV instance
    """
    results: Dict[str, GridSearchCV] = {}
    if param_grids is None:
        param_grids = {}

    # 1) KNN
    results["knn"] = cv_knn(x_train, y_train_log, param_grids.get("knn", {}), cv, n_jobs)

    # 2)

    # 3) Linear Regression
    results["linear"] = cv_linear(x_train, y_train_log, param_grids.get("linear", {}), cv, n_jobs)

    # 4) poly_ridge (PolynomialFeatures degree=2 + Ridge)
    results["poly_ridge"] = cv_poly_ridge(x_train, y_train_log, param_grids.get("poly_ridge", {}), cv, n_jobs)

    # 5) Simple (standard) LPR using Rsklpr
    results["simple_lpr"] = cv_lpr(x_train, y_train_log, param_grids.get("simple_lpr", {}), cv, n_jobs)

    # 6) Subway-aware RSKLPR (index-based kernel)
    results["grclpr_subway"] = cv_grclpr_subway(
        x_train=x_train,
        y_train_log=y_train_log,
        train_indices=train_indices,
        subway_graph=subway_graph,
        original_idx_to_station_map=original_idx_to_station_map,
        param_grid=param_grids.get("grclpr_subway", {}),
        cv=cv,
        n_jobs=n_jobs,
    )

    print("✅ Grid search complete for all models.")
    return results


def display_model_results(
    grid_search_results: Dict[str, GridSearchCV],
    x_test: np.ndarray,
    y_test_log: np.ndarray,
    test_indices: np.ndarray,
    price_clip: float = 1000.0,
) -> None:
    """
    Calculates and plots performance metrics for each model from GridSearchCV results.

    Generates two figures:
    1. A grid of scatter/residual plots, one row per model.
    2. A figure with summary bar plots comparing R2, MAE, and RMSE for all models.

    Args:
        grid_search_results: A dictionary mapping model names to fitted GridSearchCV objects.
        x_test: Test set features (without original indices).
        y_test_log: Log-transformed true target values for the test set.
        test_indices: Original dataset indices corresponding to the test set.
        price_clip: The maximum price to display on the plots for readability.
    """
    print("\n📊 Displaying results for all evaluated models...")

    y_test_price = np.expm1(y_test_log)
    num_models = len(grid_search_results)

    if num_models == 0:
        print("No models to display.")
        return

    # --- 1. Create Scatter/Residual Plots ---

    # Create a figure with one row per model, and two columns for the plots
    fig, axes = plt.subplots(num_models, 2, figsize=(15, 5 * num_models), squeeze=False)
    fig.suptitle("Model Performance Comparison (Individual Plots)", fontsize=20, y=1.0)

    # Store metrics for the summary bar plots
    model_metrics = []

    for i, (model_name, grid_search) in enumerate(grid_search_results.items()):
        print(f"\n--- Processing: {model_name} ---")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV Score (neg_mean_squared_error): {grid_search.best_score_:.4f}")

        best_estimator = grid_search.best_estimator_

        # Prepare test data: SubwayRsklprWrapper expects original indices
        if model_name == "grclpr_subway":
            x_test_prepared = np.hstack((x_test, test_indices.reshape(-1, 1)))
        else:
            x_test_prepared = x_test

        # Make predictions
        y_pred_log = best_estimator.predict(x_test_prepared)
        y_pred_price = np.expm1(y_pred_log)

        # --- REQ 1: Calculate R2, MAE, and RMSE in original price space ---
        r2 = r2_score(y_test_price, y_pred_price)
        mae = mean_absolute_error(y_test_price, y_pred_price)
        rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))

        print(f"Test Set R² (Price):   {r2:.4f}")
        print(f"Test Set MAE (Price):  ${mae:.2f}")
        print(f"Test Set RMSE (Price): ${rmse:.2f}")

        # Store metrics for summary plot
        model_metrics.append({"model_name": model_name, "r2": r2, "mae": mae, "rmse": rmse})

        # --- REQ 2: Plot 1: True vs. Predicted Price (with new metrics) ---
        ax1 = axes[i, 0]
        ax1.scatter(
            np.clip(y_test_price, 0, price_clip),
            np.clip(y_pred_price, 0, price_clip),
            alpha=0.1,
            s=10,
        )
        ax1.plot([0, price_clip], [0, price_clip], "r--", label="Perfect Prediction")
        ax1.set_xlabel("True Price ($)")
        ax1.set_ylabel("Predicted Price ($)")
        # Add R² and MAE to the title
        ax1.set_title(f"{model_name}\nTrue vs. Predicted (R²: {r2:.4f}, MAE: ${mae:.2f})", fontsize=14)
        ax1.legend()
        ax1.grid(True)
        ax1.axis("equal")
        ax1.set_xlim(0, price_clip)
        ax1.set_ylim(0, price_clip)

        # --- REQ 2: Plot 2: Residual Plot (with new metrics) ---
        ax2 = axes[i, 1]
        residuals = y_test_price - y_pred_price
        ax2.scatter(np.clip(y_pred_price, 0, price_clip), residuals, alpha=0.1, s=10)
        ax2.axhline(y=0, color="r", linestyle="--", label="No Error")
        ax2.set_xlabel("Predicted Price ($)")
        ax2.set_ylabel("Residuals (True - Predicted) ($)")
        # Add RMSE to the title
        ax2.set_title(f"Residual Plot (RMSE: ${rmse:.2f})", fontsize=14)
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlim(0, price_clip)
        clip_val = np.std(residuals) * 3
        ax2.set_ylim(-max(clip_val, price_clip / 2), max(clip_val, price_clip / 2))

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout for suptitle
    plt.show()

    # --- REQ 3: Create Summary Bar Plots ---

    if not model_metrics:
        print("No metrics to plot in summary.")
        return

    # Convert metrics to a DataFrame for easy plotting
    metrics_df = pd.DataFrame(model_metrics).set_index("model_name")

    # Create a new figure for the bar plots
    fig_bars, (ax_r2, ax_mae, ax_rmse) = plt.subplots(3, 1, figsize=(12, 18))
    fig_bars.suptitle("Test Set Metric Comparison (Summary)", fontsize=20, y=1.02)

    # Plot 1: R² (Higher is better)
    metrics_df["r2"].sort_values(ascending=False).plot(
        kind="barh", ax=ax_r2, colormap="summer", title="R² Score (Higher is Better)"
    )
    ax_r2.set_xlabel("R² Score")
    ax_r2.set_ylabel("")  # Remove model_name label from y-axis
    ax_r2.grid(axis="x", linestyle="--", alpha=0.7)
    # Add data labels
    for i, v in enumerate(metrics_df["r2"].sort_values(ascending=False)):
        ax_r2.text(v + 0.01, i, f"{v:.4f}", va="center", color="black")
    ax_r2.set_xlim(right=metrics_df["r2"].max() * 1.1)  # Add padding

    # Plot 2: MAE (Lower is better)
    metrics_df["mae"].sort_values(ascending=True).plot(
        kind="barh", ax=ax_mae, colormap="autumn", title="Mean Absolute Error (Lower is Better)"
    )
    ax_mae.set_xlabel("Mean Absolute Error ($)")
    ax_mae.set_ylabel("")
    ax_mae.grid(axis="x", linestyle="--", alpha=0.7)
    # Add data labels
    for i, v in enumerate(metrics_df["mae"].sort_values(ascending=True)):
        ax_mae.text(v + 0.5, i, f"${v:.2f}", va="center", color="black")
    ax_mae.set_xlim(right=metrics_df["mae"].max() * 1.1)

    # Plot 3: RMSE (Lower is better)
    metrics_df["rmse"].sort_values(ascending=True).plot(
        kind="barh", ax=ax_rmse, colormap="winter", title="Root Mean Squared Error (Lower is Better)"
    )
    ax_rmse.set_xlabel("Root Mean Squared Error ($)")
    ax_rmse.set_ylabel("")
    ax_rmse.grid(axis="x", linestyle="--", alpha=0.7)
    # Add data labels
    for i, v in enumerate(metrics_df["rmse"].sort_values(ascending=True)):
        ax_rmse.text(v + 0.5, i, f"${v:.2f}", va="center", color="black")
    ax_rmse.set_xlim(right=metrics_df["rmse"].max() * 1.1)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
