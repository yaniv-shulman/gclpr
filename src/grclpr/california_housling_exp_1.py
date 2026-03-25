import math
from typing import Callable, List, Tuple, Any, Dict, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rsklpr.kernels import laplacian_normalized_metric
from rsklpr.rsklpr import Rsklpr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Module-level constants for feature indices
# As defined by load_california_housing_dataset,
# Latitude is the 3rd column (idx 2) and Longitude is the 4th (idx 3)
LAT_IDX: int = 2
LON_IDX: int = 3


def haversine_rbf_kernel_factory(
    length_scale_km: float,
    radius_km: float = 6371.0088,
    lat_idx: int = LAT_IDX,
    lon_idx: int = LON_IDX,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]:
    """
    Factory to create an RBF kernel using Haversine distance.

    This wrapper conforms to the Rsklpr kp kernel signature:
    k(x_0, x_neighbors, dist_x_neighbors, indices_neighbors) -> np.ndarray

    Args:
        length_scale_km: RBF scale in kilometers.
        radius_km: Earth radius in kilometers.
        lat_idx: The column index for latitude in the feature matrix X.
        lon_idx: The column index for longitude in the feature matrix X.

    Returns:
        A kernel function for use in Rsklpr's `kp` list.
    """
    if length_scale_km <= 0:
        raise ValueError("length_scale_km must be positive")
    ls2: float = float(length_scale_km) ** 2

    def _to_rad(a: np.ndarray) -> np.ndarray:
        return np.deg2rad(a)

    def _haversine(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
        """s0 is [1, K], s1 is [N, K]"""
        # Use parameterized indices
        lat0: float = _to_rad(s0[0, lat_idx])
        lon0: float = _to_rad(s0[0, lon_idx])
        lat1: np.ndarray = _to_rad(s1[:, lat_idx])
        lon1: np.ndarray = _to_rad(s1[:, lon_idx])

        dlat: np.ndarray = lat1 - lat0
        dlon: np.ndarray = lon1 - lon0
        h: np.ndarray = (
            np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
        )
        return 2 * radius_km * np.arcsin(np.minimum(1.0, np.sqrt(h)))

    def _kernel(
        x_0: np.ndarray,
        x_neighbors: np.ndarray,
        _: np.ndarray,
        __: int,
        ___: np.ndarray,
    ) -> np.ndarray:
        """
        The actual kernel function.
        - x_0: Target point, shape [1, K]
        - x_neighbors: Neighbor points, shape [N_neighbors, K]
        - _: Ignored distances (dist_x_neighbors)
        - __: Ignored target index in the test set
        - ___: Ignored neighbours indices in the training set
        """
        # Calculate Haversine distance using the specified columns
        d: np.ndarray = _haversine(s0=x_0, s1=x_neighbors)
        # Return RBF kernel weights
        return np.exp(-0.5 * (d**2) / ls2)

    # Return the kernel function, reshaped to (1, N_neighbors) as expected by Rsklpr
    return lambda x0, xn, d, x0_index, indices: _kernel(x0, xn, d, x0_index, indices).reshape(1, -1)


def load_california_housing_dataset() -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Loads and prepares the California Housing dataset.

    Features: 'MedInc', 'AveRooms', 'Latitude', 'Longitude'
    Target: 'MedHouseVal'

    'Latitude' and 'Longitude' are placed at indices 2 and 3, respectively,
    to match the default global constants (LAT_IDX, LON_IDX).

    Non-geospatial features ('MedInc', 'AveRooms') are scaled.

    Returns:
        A tuple of (x_train, x_test, y_train, y_test)
    """
    print("📥 Loading and preparing California Housing dataset...")
    housing: Any = fetch_california_housing(as_frame=True)
    california_df: pd.DataFrame = housing.frame.copy()

    # We will use MedInc, AveRooms, Latitude, and Longitude
    # Latitude and Longitude MUST be at indices LAT_IDX and LON_IDX
    main_features: List[str] = ["MedInc", "AveRooms", "Latitude", "Longitude"]
    X_full = california_df[main_features].values
    y_full = california_df["MedHouseVal"].values

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    # Scale the non-geospatial features
    # This is good practice for distance-based methods like RSKLPR and KNN
    non_geo_indices: List[int] = [0, 1]
    scaler = StandardScaler()
    x_train[:, non_geo_indices] = scaler.fit_transform(x_train[:, non_geo_indices])
    x_test[:, non_geo_indices] = scaler.transform(x_test[:, non_geo_indices])

    print(
        f"Data prepared: {len(y_train)} training samples, {len(y_test)} testing samples."
    )

    print(f"Features: {main_features}")
    return x_train, x_test, y_train, y_test


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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RsklprWrapper":
        """
        Fits the Rsklpr model.
        """
        self.model_: Rsklpr = Rsklpr(
            size_neighborhood=self.size_neighborhood,
            degree=self.degree,
            kp=self.kp,
            kr=self.kr,
            metric_x=self.metric_x,
            metric_x_params=self.metric_x_params,
            suppress_warnings=True,  # Add warning suppression
        )

        self.model_.fit(x=X, y=y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts using the fitted Rsklpr model.
        """
        return self.model_.predict(x=X)


def cv_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 0,
) -> GridSearchCV:
    """Runs GridSearchCV for LinearRegression."""
    print("🔎 Grid searching Global Linear... (CV)")
    model = LinearRegression()
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )
    grid.fit(x_train, y_train)
    return grid


def cv_poly_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
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

    quad_grid.fit(x_train, y_train)
    return quad_grid


def cv_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 0,
) -> GridSearchCV:
    """Runs GridSearchCV for KNeighborsRegressor."""
    print("🔎 Grid searching K-Nearest Neighbors... (CV)")
    model = KNeighborsRegressor()
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )
    grid.fit(x_train, y_train)
    return grid


def cv_rsklpr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 0,
) -> GridSearchCV:
    """Runs GridSearchCV for standard LPR and RSKLPR."""
    print("🔎 Grid searching LPR (Standard)... (CV)")
    model = RsklprWrapper()
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )
    grid.fit(x_train, y_train)
    return grid


def cv_gclpr_geo(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int,
    n_jobs: int,
    verbose: int = 0,
) -> GridSearchCV:
    """Runs GridSearchCV for Geospatial RSKLPR (GRC-LPR)."""
    print("🔎 Grid searching GRC-LPR (Geospatial)... (CV)")
    # The RsklprWrapper is generic; the geospatial part is defined
    # by the `kp` list passed in the `param_grid`.
    model = RsklprWrapper()
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=verbose,
    )
    grid.fit(x_train, y_train)
    return grid


# --- Main Coordinator and Display Functions ---


def grid_search_multiple_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    n_jobs: int = -1,
    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    verbose: int = 2,
) -> Dict[str, Any]:
    """
    Run GridSearchCV for a set of different regressors by calling
    dedicated CV functions for each model.

    Args:
        x_train: Features for training (NxD)
        y_train: Targets (N,)
        cv: Number of CV folds
        n_jobs: number of parallel jobs for GridSearchCV
        param_grids: Optional dict specifying parameter grids for each named model.
                     If None, default grids are used.
        verbose: Verbosity level for GridSearchCV.

    Returns:
        A dict mapping model name -> fitted (GridSearchCV) instance
    """
    print("⚙️ Defining models and search grids for comparison...")

    if param_grids is None:
        param_grids = {}

    # --- Define Kernels for Rsklpr Grids ---
    std_laplacian: List[Callable] = [laplacian_normalized_metric]

    grc_laplacian_5km: List[Callable] = [
        laplacian_normalized_metric,
        haversine_rbf_kernel_factory(length_scale_km=5.0),
    ]

    grc_laplacian_10km: List[Callable] = [
        laplacian_normalized_metric,
        haversine_rbf_kernel_factory(length_scale_km=10.0),
    ]

    grc_laplacian_15km: List[Callable] = [
        laplacian_normalized_metric,
        haversine_rbf_kernel_factory(length_scale_km=15.0),
    ]

    grc_laplacian_20km: List[Callable] = [
        laplacian_normalized_metric,
        haversine_rbf_kernel_factory(length_scale_km=20.0),
    ]

    # --- Define Default Grids ---
    default_grids = {
        "linear": {},
        "poly": {"poly__degree": [2, 3]},
        "knn": {"n_neighbors": list(range(5, 201, 15))},
        "rsklpr_std": {
            "size_neighborhood": list(range(5, 201, 15)),
            "kp": [std_laplacian],
            "degree": [0, 1, 2],
            "metric_x": ["minkowski", "mahalanobis"],
            "kr": ["none", "joint", "conden"],
        },
        "rsklpr_geo": {
            "size_neighborhood": list(range(5, 201, 15)),
            "kp": [
                grc_laplacian_5km,
                grc_laplacian_10km,
                grc_laplacian_15km,
                grc_laplacian_20km,
            ],
            "degree": [0, 1, 2],
            "kr": ["none", "joint", "conden"],
            "metric_x": ["minkowski", "mahalanobis"],
        },
    }

    # Merge user-provided grids with defaults
    for key, grid in default_grids.items():
        if key not in param_grids:
            param_grids[key] = grid

    print("✅ Grids defined.")

    # --- Run Training via CV helpers ---
    fitted_models: Dict[str, Any] = {}

    fitted_models["Global Linear"] = cv_linear(
        x_train, y_train, param_grids["linear"], cv, n_jobs, verbose
    )

    fitted_models["Global Polynomial"] = cv_poly_ridge(
        x_train, y_train, param_grids["poly"], cv, n_jobs, verbose
    )

    fitted_models["K-Nearest Neighbors"] = cv_knn(
        x_train, y_train, param_grids["knn"], cv, n_jobs, verbose
    )

    fitted_models["RSKLPR (Standard)"] = cv_rsklpr(
        x_train, y_train, param_grids["rsklpr_std"], cv, n_jobs, verbose
    )

    fitted_models["GRC-LPR (Geospatial)"] = cv_gclpr_geo(
        x_train, y_train, param_grids["rsklpr_geo"], cv, n_jobs, verbose
    )

    print("\n🎉 All models trained.")
    return fitted_models


def display_model_results(
    grid_search_results: Dict[str, Any],
    x_test: np.ndarray,
    y_test: np.ndarray,
    save_plots: bool = False,
) -> None:
    """
    Evaluates fitted models on the test set, prints a results summary,
    and generates scatter and geospatial error plots.

    Args:
        grid_search_results: The dict of fitted models from grid_search_multiple_models.
        x_test: Test features.
        y_test: Test targets.
        save_plots: Whether to save the generated plots as PNG files.
    """
    print("\n\n--- 📊 Evaluating models on Test Set ---")

    results: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, model in grid_search_results.items():
        print(f"\n--- Evaluating: {name} ---")
        if hasattr(model, "best_params_"):
            print(f"Best CV Params: {model.best_params_}")
            print(f"Best CV Score (neg_mean_squared_error): {model.best_score_:.4f}")

        # Use the best estimator found by GridSearchCV
        best_estimator = model.best_estimator_

        # Make predictions
        y_pred = best_estimator.predict(x_test)
        predictions[name] = y_pred

        # Handle potential NaNs from Rsklpr
        nan_mask: np.ndarray = np.isnan(y_pred)

        if np.any(nan_mask):
            num_nans: int = np.sum(nan_mask)
            print(f"⚠️ Found {num_nans} NaN predictions. Removing them from evaluation.")
            # Filter both y_test and y_pred to only non-NaN values
            y_test_filtered = y_test[~nan_mask]
            y_pred_filtered = y_pred[~nan_mask]
        else:
            y_test_filtered = y_test
            y_pred_filtered = y_pred

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred_filtered))
        r2 = r2_score(y_test_filtered, y_pred_filtered)
        results[name] = {"RMSE": rmse, "R²": r2}
        print(f"✅ Test Set Evaluation: (RMSE: {rmse:.4f} | R²: {r2:.4f})")

    # --- Print Summary Table ---
    print("\n--- 📊 Model Performance Comparison (Test Set) ---")
    results_df = pd.DataFrame(results).T.sort_values(by="RMSE")
    print(results_df.round(4))
    print("-------------------------------------------------\n")

    # --- Generate Plots ---
    plot_prediction_scatter_grid(predictions, results, y_test)
    plot_geospatial_error_maps(predictions, results, x_test, y_test)


def plot_prediction_scatter_grid(
    predictions: Dict[str, np.ndarray],
    results: Dict[str, Dict[str, float]],
    y_test: np.ndarray,
    save_plots: bool = False,
) -> None:
    """
    Plots a grid of actual vs. predicted scatter plots for all models.

    Args:
        predictions: The predictions dictionary from display_model_results.
        results: The results dictionary from display_model_results.
        y_test: The true test target values.
        save_plots: Whether to save the generated plot as a PNG file.
    """
    print("📈 Plotting scatter results...")
    num_models: int = len(predictions)
    ncols: int = 3
    nrows: int = math.ceil(num_models / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 5 * nrows),  # Adjusted height for flexible rows
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    fig.suptitle("Comparison of Model Predictions: Actual vs. Predicted", fontsize=18)

    # Handle case of single row or single plot
    if num_models == 1:
        axes_flat = [axes]
    elif nrows == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()

    plot_min = y_test.min()
    plot_max = np.percentile(y_test, 99.5)  # Use percentile for better viz

    name_to_title: Dict[str, str] = {
        "linear": "Global Linear",
        "poly_ridge": "Global Polynomial",
        "knn": "K-Nearest Neighbors",
        "lpr": "LPR",
        "rsklpr": "RSKLPR (Standard)",
        "gclpr_geo": "GC-LPR (Geospatial)",
    }

    for i, name in enumerate(predictions.keys()):
        ax = axes_flat[i]
        y_pred: np.ndarray = predictions[name]

        # Filter NaNs for plotting
        nan_mask: np.ndarray = np.isnan(y_pred)
        y_test_filtered = y_test[~nan_mask]
        y_pred_filtered = y_pred[~nan_mask]

        ax.scatter(y_test_filtered, y_pred_filtered, alpha=0.1, s=10)

        # Plot the 1:1 diagonal line
        ax.plot([plot_min, plot_max], [plot_min, plot_max], "r--", lw=2)

        # Force all plots to use the same limits
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)

        # Force a square aspect ratio
        ax.set_aspect("equal", "box")

        ax.set_title(
            f"{name_to_title[name]}\nRMSE: {results[name]['RMSE']:.4f} | R²: {results[name]['R²']:.4f}"
        )

        if i >= (num_models - ncols):  # Bottom row
            ax.set_xlabel("Actual Value")
        if i % ncols == 0:  # First column
            ax.set_ylabel("Predicted Value")

    # Turn off any empty subplots
    for i in range(num_models, len(axes_flat)):
        axes_flat[i].axis("off")

    if save_plots:
        plt.savefig("california_housling_exp_1_scatter_grid.png")

    plt.show()



def plot_geospatial_error_maps(
    predictions: Dict[str, np.ndarray],
    results: Dict[str, Dict[str, float]],
    x_test: np.ndarray,
    y_test: np.ndarray,
    save_plots: bool = False,
) -> None:
    """
    Plots 2x2 geospatial error maps for the primary models.

    Args:
        predictions: The predictions dictionary from display_model_results.
        results: The results dictionary from display_model_results.
        x_test: The test features (used to get lat/lon).
        y_test: The true test target values.
        save_plots: Whether to save the generated plot as a PNG file.
    """
    print("🗺️ Plotting geospatial error maps...")

    # These are the 4 models we want to see in the 2x2 grid
    models_to_plot = ["linear", "poly_ridge", "lpr", "gclpr_geo"]

    # Check if all required models are present
    if not all(model_name in predictions for model_name in models_to_plot):
        print(
            f"Warning: Skipping geospatial plot. Missing one or more required models: {models_to_plot}"
        )
        return

    # Get predictions using the correct keys
    pred_lin: np.ndarray = predictions["linear"]
    pred_poly: np.ndarray = predictions["poly_ridge"]
    pred_lpr: np.ndarray = predictions["lpr"]
    pred_geo_lpr: np.ndarray = predictions["gclpr_geo"]

    # Calculate errors using the new variables
    error_lin: np.ndarray = y_test - pred_lin
    error_poly: np.ndarray = y_test - pred_poly
    error_lpr: np.ndarray = y_test - pred_lpr
    error_geo_lpr: np.ndarray = y_test - pred_geo_lpr

    # Get Lat/Lon from the (scaled) X_test using global constants
    test_lon: np.ndarray = x_test[:, LON_IDX]
    test_lat: np.ndarray = x_test[:, LAT_IDX]

    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    fig.suptitle(
        "Geospatial Analysis of Model Performance (Test Set Errors)", fontsize=18
    )

    axes = axes.flatten()

    # Determine color scale using the correct error arrays
    all_errors = np.concatenate(
        [
            error_lin,
            error_poly,
            error_lpr[~np.isnan(error_lpr)],
            error_geo_lpr[~np.isnan(error_geo_lpr)],
        ]
    )

    # Use percentiles to avoid extreme outliers skewing the color map
    vmax = np.nanpercentile(np.abs(all_errors), 99)
    vmin = -vmax

    # Map 1 (Linear)
    sc1 = axes[0].scatter(
        test_lon,
        test_lat,
        c=error_lin,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=5,
        alpha=0.5,
    )

    axes[0].set_title(
        f"Linear Errors (RMSE: {results['linear']['RMSE']:.3f})"  # Use correct key
    )

    fig.colorbar(sc1, ax=axes[0], label="Error (Actual - Pred)")

    # Map 2 (Polynomial)
    sc2 = axes[1].scatter(
        test_lon,
        test_lat,
        c=error_poly,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=5,
        alpha=0.5,
    )

    axes[1].set_title(
        f"Poly Ridge Errors (RMSE: {results['poly_ridge']['RMSE']:.3f})"  # Use correct key
    )

    fig.colorbar(sc2, ax=axes[1], label="Error (Actual - Pred)")

    # Map 3 (Standard LPR)
    nan_mask_lpr: np.ndarray = np.isnan(error_lpr)

    sc3 = axes[2].scatter(
        test_lon[~nan_mask_lpr],
        test_lat[~nan_mask_lpr],
        c=error_lpr[~nan_mask_lpr],
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=5,
        alpha=0.5,
    )

    axes[2].set_title(
        f"LPR (Standard) Errors (RMSE: {results['lpr']['RMSE']:.3f})"  # Use correct key
    )

    fig.colorbar(sc3, ax=axes[2], label="Error (Actual - Pred)")

    # Map 4 (Geospatial LPR)
    nan_mask_geo_lpr: np.ndarray = np.isnan(error_geo_lpr)  # Use correct error variable

    sc4 = axes[3].scatter(
        test_lon[~nan_mask_geo_lpr],
        test_lat[~nan_mask_geo_lpr],
        c=error_geo_lpr[~nan_mask_geo_lpr],  # Use correct error variable
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=5,
        alpha=0.5,
    )

    axes[3].set_title(
        f"GCLPR (Geospatial) Errors (RMSE: {results['gclpr_geo']['RMSE']:.3f})"  # Use correct key
    )

    fig.colorbar(sc4, ax=axes[3], label="Error (Actual - Pred)")

    # Set labels for the bottom plots
    axes[2].set_xlabel("Longitude")
    axes[3].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[2].set_ylabel("Latitude")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_plots:
        plt.savefig("california_housling_exp_1_geospatial_errors.png")

    plt.show()