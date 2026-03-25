import math
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rsklpr.kernels import laplacian_normalized_metric, tricube_normalized_metric
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from grclpr.nyc_airbnb import (
    load_subway_graph,
    calculate_unified_index_to_station_map,
    cv_knn,
    cv_linear,
    cv_poly_ridge,
    cv_lpr,  # Standard LPR
    cv_rsklpr,  # Standard RSKLPR
    cv_grclpr_subway,  # The specialized Subway Wrapper
)

# --- CONFIGURATION ---
N_FOLDS = 10
CV_INNER = 3  # Inner folds for GridSearch
RANDOM_STATE = 42
RESULTS_CSV = "nyc_airbnb_10fold_results_raw.csv"
SUMMARY_CSV = "nyc_airbnb_10fold_summary.csv"
DATA_PATH = Path("/home/yaniv/data/datasets/raw/air_bnb_nyc_listings") / "listings.csv"


def get_full_airbnb_data_for_cv():
    """
    Loads, cleans, and encodes the full dataset for Cross Validation.
    Returns the raw X matrix (unscaled) and the INDICES of columns that need scaling.
    """
    print("📥 Loading and preparing NYC Airbnb dataset (Full)...")
    airbnb_df = pd.read_csv(DATA_PATH)

    # 1. Feature Selection
    features = [
        "host_listings_count", "latitude", "longitude", "room_type",
        "price", "minimum_nights", "number_of_reviews",
        "calculated_host_listings_count", "availability_365",
    ]
    airbnb_df = airbnb_df[features].copy()

    # 2. Cleaning Price
    airbnb_df["price"] = airbnb_df["price"].replace({"\\$": "", ",": ""}, regex=True)
    airbnb_df["price"] = pd.to_numeric(airbnb_df["price"], errors="coerce")
    airbnb_df.dropna(subset=["price"], inplace=True)
    airbnb_df = airbnb_df[airbnb_df["price"] > 0]

    # 3. Filling NaNs
    cols_to_fill = [
        "host_listings_count", "number_of_reviews",
        "calculated_host_listings_count", "availability_365",
    ]
    median_values = airbnb_df[cols_to_fill].median()
    airbnb_df.fillna(median_values, inplace=True)

    # 4. Target Preparation
    airbnb_df["log_price"] = np.log1p(airbnb_df["price"])
    y_log = airbnb_df["log_price"].values
    y_price = airbnb_df["price"].values  # Keep raw price for evaluation

    # 5. One-Hot Encoding
    airbnb_df = pd.get_dummies(airbnb_df, columns=["room_type"], drop_first=True)

    # 6. Define Column Order
    # We want [Numeric + Encoded Features] then [Lat, Lon]
    main_features = [col for col in airbnb_df.columns if col not in ["price", "log_price", "latitude", "longitude"]]
    geo_features = ["latitude", "longitude"]
    all_features = main_features + geo_features

    print(f"Features used ({len(all_features)}): {all_features}")

    # 7. Identify Indices for Scaling
    # We ONLY want to scale these specific continuous variables
    numerical_cols_to_scale = [
        "host_listings_count",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
    ]

    # Calculate integer indices for these columns
    scale_indices = [all_features.index(col) for col in numerical_cols_to_scale if col in all_features]

    # 8. Create Final Arrays
    # Note: listing_coords_all needs to match the index of X exactly
    X = airbnb_df[all_features].values.astype(np.float32)
    listing_coords_all = airbnb_df[geo_features]  # Series/DF for the station mapper

    return X, y_log, y_price, listing_coords_all, scale_indices, all_features


def define_search_grids():
    """Defines search grids for Airbnb models."""

    linear_param_grid = {
        "fit_intercept": [True, False],
        "positive": [False, True],
    }

    param_grid_poly_ridge = {
        "poly__degree": [2, 3],
        "poly__interaction_only": [False, True],
        "ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    }

    knn_param_grid = {
        "n_neighbors": list(range(3, 100, 2)),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "leaf_size": [15, 30, 60],
    }

    # Standard LPR / RSKLPR (No Subway knowledge, just Lat/Lon features)
    kp = [tricube_normalized_metric, laplacian_normalized_metric]
    size_neighborhood = list(range(11, 20, 1)) + list(range(20, 31, 2))
    degree = [0, 1]

    lpr_param_grid = [
        {
            "size_neighborhood": size_neighborhood,
            "degree": degree,
            "kp": kp,
            "metric_x": ["minkowski"],
            "metric_x_params": [{"p": 1}, {"p": 2}],
            "kr": ["none"],
        },
        {
            "size_neighborhood": size_neighborhood,
            "degree": degree,
            "kp": kp,
            "metric_x": ["mahalanobis"],
            "metric_x_params": [None],
            "kr": ["none"],
        },
    ]

    rsklpr_param_grid: List[Dict[str, Any]] = [
        # --- Grid 1: Testing Minkowski metric (p=1 and p=2) ---
        {
            "size_neighborhood": size_neighborhood,
            "degree": degree,
            "kp": kp,  # Base kernels
            "metric_x": ["minkowski"],
            "metric_x_params": [{"p": 1}, {"p": 2}],
            "kr": ["conden", "joint"],  # RSKLPR
        },
        # --- Grid 2: Testing Mahalanobis metric ---
        {
            "size_neighborhood": size_neighborhood,
            "degree": degree,
            "kp": kp,  # Base kernels
            "metric_x": ["mahalanobis"],
            "metric_x_params": [None],  # Mahalanobis takes no params
            "kr": ["conden", "joint"],  # RSKLPR
        },
    ]

    # --- SUBWAY MODELS ---
    distance_scale = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    gclpr_subway_param_grid = [
        # Grid 1: Minkowski
        {
            "size_neighborhood": size_neighborhood,
            "degree": degree,
            "distance_scale": distance_scale,
            "kp": kp,
            "metric_x": ["minkowski"],
            "metric_x_params": [{"p": 1}, {"p": 2}],
            "kr": ["none"],  # Non-Robust
        },
        # Grid 2: Mahalanobis
        {
            "size_neighborhood": size_neighborhood,
            "degree": degree,
            "distance_scale": distance_scale,
            "kp": kp,
            "metric_x": ["mahalanobis"],
            "metric_x_params": [None],
            "kr": ["none"],  # Non-Robust
        },
    ]

    grclpr_geo_param_grid = [
        # Grid 1: Minkowski
        {
            "size_neighborhood": size_neighborhood,
            "degree": degree,
            "distance_scale": distance_scale,
            "kp": kp,
            "metric_x": ["minkowski"],
            "metric_x_params": [{"p": 1}, {"p": 2}],
            "kr": ["conden", "joint"],  # Robust
        },
        # Grid 2: Mahalanobis
        {
            "size_neighborhood": size_neighborhood,
            "degree": degree,
            "distance_scale": distance_scale,
            "kp": kp,
            "metric_x": ["mahalanobis"],
            "metric_x_params": [None],
            "kr": ["conden", "joint"],  # Robust
        },
    ]

    return {
        "linear": linear_param_grid,
        "poly_ridge": param_grid_poly_ridge,
        "knn": knn_param_grid,
        "lpr": lpr_param_grid,
        "rsklpr": rsklpr_param_grid,
        "gclpr_subway": gclpr_subway_param_grid,
        "grclpr_subway": grclpr_geo_param_grid
    }


def plot_airbnb_experiment_results(predictions_map, y_true, coords, save_prefix="nyc_airbnb"):
    """
    Custom plotting logic for Airbnb Experiment.
    Handles dynamic grid layout and specifically uses 'coords' for geospatial mapping.
    """
    print("\n🎨 Generating Airbnb Result Plots...")

    # Identify models that have results
    models = [k for k in predictions_map.keys() if not np.all(np.isnan(predictions_map[k]))]
    num_models = len(models)

    if num_models == 0:
        print("No predictions found to plot.")
        return

    # Calculate Grid Size (e.g., 3 columns)
    cols = 3
    rows = math.ceil(num_models / cols)

    # --- 1. SCATTER PLOTS (Actual vs Predicted) ---
    fig_scat, axes_scat = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    fig_scat.suptitle("NYC Airbnb: Actual vs Predicted Price ($)", fontsize=16)

    axes_flat = axes_scat.flatten() if num_models > 1 else [axes_scat]

    # Plot Limits (Clip top 1% for cleaner visualization)
    plot_max = np.percentile(y_true, 99)

    for i, name in enumerate(models):
        ax = axes_flat[i]
        y_pred = predictions_map[name]
        mask = ~np.isnan(y_pred)

        # Calculate Metrics
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        r2 = r2_score(y_true[mask], y_pred[mask])

        ax.scatter(y_true[mask], y_pred[mask], alpha=0.2, s=5, c='blue')
        ax.plot([0, plot_max], [0, plot_max], 'r--', lw=2)

        ax.set_title(f"{name}\nMAE: ${mae:.0f} | R²: {r2:.3f}")
        ax.set_xlim(0, plot_max)
        ax.set_ylim(0, plot_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if i % cols == 0:
            ax.set_ylabel("Predicted ($)")
        if i >= num_models - cols:
            ax.set_xlabel("Actual ($)")

    # Hide empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.savefig(f"{save_prefix}_scatter.png", dpi=150)
    plt.close()

    # --- 2. GEOSPATIAL ERROR MAPS ---
    fig_geo, axes_geo = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    fig_geo.suptitle("Geospatial Error Distribution (Actual - Predicted)", fontsize=16)

    axes_geo_flat = axes_geo.flatten() if num_models > 1 else [axes_geo]

    lons = coords["longitude"].values
    lats = coords["latitude"].values

    for i, name in enumerate(models):
        ax = axes_geo_flat[i]
        y_pred = predictions_map[name]
        error = y_true - y_pred  # Positive = Underprediction (Price was higher than predicted)

        # Determine color scale (robust to outliers)
        # Use 95th percentile to set the color range
        valid_errors = error[~np.isnan(error)]
        limit = np.percentile(np.abs(valid_errors), 95)

        sc = ax.scatter(lons, lats, c=error, cmap='seismic', s=3, alpha=0.6, vmin=-limit, vmax=limit)

        ax.set_title(f"{name}")
        ax.axis('off')  # Hide axes for map look

        # Add colorbar inside or below? Let's do a simple one per plot
        plt.colorbar(sc, ax=ax, label="Error ($)", shrink=0.7)

    # Hide empty subplots
    for j in range(i + 1, len(axes_geo_flat)):
        axes_geo_flat[j].axis('off')

    plt.savefig(f"{save_prefix}_geo_errors.png", dpi=150)
    plt.close()
    print("✅ Plots saved.")


def run_airbnb_subway_experiment():
    # 1. Load Data & Graph
    subway_graph = load_subway_graph()

    # Load data AND the scaling indices
    X, y_log, y_price, coords, scale_indices, feature_names = get_full_airbnb_data_for_cv()

    # 2. Pre-compute Station Map (Global)
    idx_to_station = calculate_unified_index_to_station_map(
        subway_graph, coords, max_station_dist_km=1.0
    )

    grids = define_search_grids()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_records = []

    # "Perfect Map" Storage (Price Space)
    full_preds_price = {name: np.full_like(y_price, np.nan) for name in grids.keys()}

    print(f"\n🚀 Starting {N_FOLDS}-Fold CV on NYC Airbnb...")

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- 🚇 Processing Fold {fold_i + 1}/{N_FOLDS} ---")

        # A. Slice Data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_log = y_log[train_idx]
        y_test_price = y_price[test_idx]  # For evaluation

        # B. Precise Scaling (Corrected Logic)
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        scaler = StandardScaler()

        # Only scale the specific numerical columns we identified earlier
        X_train_scaled[:, scale_indices] = scaler.fit_transform(X_train[:, scale_indices])
        X_test_scaled[:, scale_indices] = scaler.transform(X_test[:, scale_indices])

        # C. Train Models
        models_to_run = {
            "linear": cv_linear(X_train_scaled, y_train_log, grids["linear"], cv=CV_INNER, n_jobs=-1),
            "poly_ridge": cv_poly_ridge(X_train_scaled, y_train_log, grids["poly_ridge"], cv=CV_INNER, n_jobs=-1),
            "knn": cv_knn(X_train_scaled, y_train_log, grids["knn"], cv=CV_INNER, n_jobs=-1),
            "lpr": cv_lpr(X_train_scaled, y_train_log, grids["lpr"], cv=CV_INNER, n_jobs=-1),
            "rsklpr": cv_rsklpr(X_train_scaled, y_train_log, grids["rsklpr"], cv=CV_INNER, n_jobs=-1),
            "gclpr_subway": cv_grclpr_subway(
                x_train=X_train_scaled,
                y_train_log=y_train_log,
                train_indices=train_idx,
                subway_graph=subway_graph,
                original_idx_to_station_map=idx_to_station,
                param_grid=grids["gclpr_subway"],
                cv=CV_INNER,
                n_jobs=-1
            ),
            "grclpr_subway": cv_grclpr_subway(
                x_train=X_train_scaled,
                y_train_log=y_train_log,
                train_indices=train_idx,
                subway_graph=subway_graph,
                original_idx_to_station_map=idx_to_station,
                param_grid=grids["grclpr_subway"],
                cv=CV_INNER,
                n_jobs=-1
            )
        }

        # E. Predict & Eval
        for name, gs_model in models_to_run.items():
            best_est = gs_model.best_estimator_

            # Prepare Input
            if "subway" in name:
                X_test_input = np.hstack((X_test_scaled, test_idx.reshape(-1, 1)))
            else:
                X_test_input = X_test_scaled

            # Predict (Log Space)
            y_pred_log = best_est.predict(X_test_input)

            # Convert to Price Space
            y_pred_price = np.expm1(y_pred_log)
            y_pred_price = np.maximum(y_pred_price, 0)  # Clip negative prices

            # Store
            full_preds_price[name][test_idx] = y_pred_price

            # Metrics (Dollars)
            mask = ~np.isnan(y_pred_price)
            rmse = np.sqrt(mean_squared_error(y_test_price[mask], y_pred_price[mask]))
            mae = mean_absolute_error(y_test_price[mask], y_pred_price[mask])
            r2 = r2_score(y_test_price[mask], y_pred_price[mask])

            fold_records.append({
                "Fold": fold_i + 1,
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Best_Params": str(gs_model.best_params_)
            })

            print(f"  > {name}: RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
            df_results = pd.DataFrame(fold_records)
            df_results.to_csv(f"fold_{fold_i + 1}_" + RESULTS_CSV, index=False)

    # --- 3. Save & Aggregate ---
    print("\n💾 Saving Results...")
    df_results = pd.DataFrame(fold_records)
    df_results.to_csv(RESULTS_CSV, index=False)

    summary = df_results.groupby("Model")[["RMSE", "MAE", "R2"]].agg(['mean', 'std'])
    print("\n📊 Final 10-Fold Statistics (Price Space):")
    print(summary)
    summary.to_csv(SUMMARY_CSV)

    # --- 4. Plotting (Corrected) ---
    # We pass 'coords' explicitly to ensure correct Lat/Lon mapping
    plot_airbnb_experiment_results(full_preds_price, y_price, coords)


if __name__ == "__main__":
    run_airbnb_subway_experiment()