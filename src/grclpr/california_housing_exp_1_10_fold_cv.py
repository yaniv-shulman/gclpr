from typing import Dict, List, Any, Callable

import numpy as np
import pandas as pd
from rsklpr.kernels import tricube_normalized_metric
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from grclpr.california_housling_exp_1 import (
    cv_linear,
    cv_poly_ridge,
    cv_knn,
    cv_rsklpr,
    cv_gclpr_geo,
    haversine_rbf_kernel_factory,
    laplacian_normalized_metric,
    plot_prediction_scatter_grid,
    plot_geospatial_error_maps
)

# --- CONFIGURATION ---
N_FOLDS = 10
RANDOM_STATE = 42
RESULTS_CSV = "california_10fold_results_raw.csv"
SUMMARY_CSV = "california_10fold_summary.csv"


def get_raw_data():
    """Loads the full dataset for K-Fold slicing."""
    print("📥 Loading full California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    features = ["MedInc", "AveRooms", "Latitude", "Longitude"]
    X = df[features].values
    y = df["MedHouseVal"].values
    return X, y


def define_search_grids():
    """Defines the hyperparameter search space."""
    linear_param_grid: Dict[str, List[Any]] = {
        "fit_intercept": [True, False],
        "positive": [False, True],  # enforce non-negative coefficients
    }

    param_grid_poly_ridge = {
        "poly__degree": [2, 3],
        "poly__interaction_only": [False, True],
        "ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    }

    knn_param_grid: Dict[str, List[Any]] = {
        "n_neighbors": list(range(3, 50, 2)),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "leaf_size": [15, 30, 60],  # this mostly affects speed, not results
    }

    lpr_param_grid: List[Dict[str, Any]] = [
        # --- Grid 1: Testing Minkowski metric (p=1 and p=2) ---
        {
            "size_neighborhood": list(range(7, 151, 10)),
            "degree": [0, 1],
            "kp": [laplacian_normalized_metric, tricube_normalized_metric],  # Base kernels
            "metric_x": ["minkowski"],
            "metric_x_params": [{"p": 1}, {"p": 2}],
            "kr": ["none"],  # standard LPR
        },
        # --- Grid 2: Testing Mahalanobis metric ---
        {
            "size_neighborhood": list(range(7, 151, 10)),
            "degree": [0, 1],
            "kp": [laplacian_normalized_metric, tricube_normalized_metric],  # Base kernels
            "metric_x": ["mahalanobis"],
            "metric_x_params": [None],  # Mahalanobis takes no params
            "kr": ["none"],  # standard LPR
        },
    ]

    rsklpr_param_grid: List[Dict[str, Any]] = [
        # --- Grid 1: Testing Minkowski metric (p=1 and p=2) ---
        {
            "size_neighborhood": list(range(7, 151, 10)),
            "degree": [1],
            "kp": [laplacian_normalized_metric, tricube_normalized_metric],  # Base kernels
            "metric_x": ["minkowski"],
            "metric_x_params": [{"p": 1}, {"p": 2}],
            "kr": ["conden", "joint"],  # standard LPR
        },
        # --- Grid 2: Testing Mahalanobis metric ---
        {
            "size_neighborhood": list(range(7, 151, 10)),
            "degree": [1],
            "kp": [laplacian_normalized_metric, tricube_normalized_metric],  # Base kernels
            "metric_x": ["mahalanobis"],
            "metric_x_params": [None],  # Mahalanobis takes no params
            "kr": ["conden", "joint"],  # standard LPR
        },
    ]

    # 1. Define the base kernels to test
    base_kernels_to_test: List[
        Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]
    ] = [laplacian_normalized_metric, tricube_normalized_metric]

    # 2. Define the geospatial kernel length scales to test (in km)
    geo_scales_km: List[float] = [3.0, 5.0, 7.5, 10.0, 15.0, 25.0]

    # 3. Create the list of kernel combinations
    # This will be the value for the 'kp' key in our grid
    geo_kp_lists: List[
        List[Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]]
    ] = []

    base_kernel: Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]

    for base_kernel in base_kernels_to_test:
        scale: float

        for scale in geo_scales_km:
            geo_kp_lists.append(
                [base_kernel, haversine_rbf_kernel_factory(length_scale_km=scale)]
            )

    gclpr_geo_param_grid: List[Dict[str, Any]] = [
        # --- Grid 1: Testing Minkowski metric (p=1 and p=2) ---
        {
            "size_neighborhood": list(range(7, 151, 10)),
            "degree": [1],
            "kp": geo_kp_lists,  # Use the lists we generated above
            "metric_x": ["minkowski"],
            "metric_x_params": [{"p": 1}, {"p": 2}],
            "kr": ["none"],
        },
        # --- Grid 2: Testing Mahalanobis metric ---
        {
            "size_neighborhood": list(range(7, 151, 10)),
            "degree": [1],
            "kp": geo_kp_lists,  # Use the same kernel lists
            "metric_x": ["mahalanobis"],
            "metric_x_params": [None],  # Mahalanobis takes no params
            "kr": ["none"],
        },
    ]

    grclpr_geo_param_grid: List[Dict[str, Any]] = [
        # --- Grid 1: Testing Minkowski metric (p=1 and p=2) ---
        {
            "size_neighborhood": list(range(7, 151, 10)),
            "degree": [1],
            "kp": geo_kp_lists,  # Use the lists we generated above
            "metric_x": ["minkowski"],
            "metric_x_params": [{"p": 1}, {"p": 2}],
            "kr": ["conden", "joint"],
        },
        # --- Grid 2: Testing Mahalanobis metric ---
        {
            "size_neighborhood": list(range(7, 151, 10)),
            "degree": [1],
            "kp": geo_kp_lists,  # Use the same kernel lists
            "metric_x": ["mahalanobis"],
            "metric_x_params": [None],  # Mahalanobis takes no params
            "kr": ["conden", "joint"],
        },
    ]

    return {
        "linear": linear_param_grid,
        "poly_ridge": param_grid_poly_ridge,
        "knn": knn_param_grid,
        "lpr": lpr_param_grid,
        "rsklpr": rsklpr_param_grid,
        "gclpr_geo": gclpr_geo_param_grid,
        "grclpr_geo": grclpr_geo_param_grid,
    }


def run_strict_kfold_experiment():
    X, y = get_raw_data()
    grids = define_search_grids()

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_records = []

    # Master Arrays for "Perfect Map" Reconstruction
    full_predictions = {
        name: np.full_like(y, np.nan) for name in grids.keys()
    }

    print(f"\n🚀 Starting {N_FOLDS}-Fold Cross-Validation (Extensive Search)...")

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- 📂 Processing Fold {fold_i + 1}/{N_FOLDS} ---")

        # A. Slice
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # B. Scale (Non-Geo features only)
        scaler = StandardScaler()
        non_geo_cols = [0, 1]

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[:, non_geo_cols] = scaler.fit_transform(X_train[:, non_geo_cols])
        X_test_scaled[:, non_geo_cols] = scaler.transform(X_test[:, non_geo_cols])
        cv = 6

        # C. Train (Inner CV Loop)
        models_to_run = {
            "linear": cv_linear(X_train_scaled, y_train, grids["linear"], cv=cv, n_jobs=-1),
            "poly_ridge": cv_poly_ridge(X_train_scaled, y_train, grids["poly_ridge"], cv=cv, n_jobs=-1),
            "knn": cv_knn(X_train_scaled, y_train, grids["knn"], cv=cv, n_jobs=-1),
            "lpr": cv_rsklpr(X_train_scaled, y_train, grids["lpr"], cv=cv, n_jobs=-1),
            "rsklpr": cv_rsklpr(X_train_scaled, y_train, grids["rsklpr"], cv=cv, n_jobs=-1),
            "gclpr_geo": cv_gclpr_geo(X_train_scaled, y_train, grids["gclpr_geo"], cv=cv, n_jobs=-1),
            "grclpr_geo": cv_gclpr_geo(X_train_scaled, y_train, grids["grclpr_geo"], cv=cv, n_jobs=-1),
        }

        # D. Predict & Log
        for name, gs_model in models_to_run.items():
            best_estimator = gs_model.best_estimator_
            y_pred = best_estimator.predict(X_test_scaled)

            # 1. Store in Master Array
            full_predictions[name][test_idx] = y_pred

            # 2. Calculate Metrics (RMSE, MAE, R2)
            mask = ~np.isnan(y_pred)
            rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            mae = mean_absolute_error(y_test[mask], y_pred[mask])
            r2 = r2_score(y_test[mask], y_pred[mask])

            # 3. Log
            best_params_str = str(gs_model.best_params_)

            fold_records.append({
                "Fold": fold_i + 1,
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Best_Params": best_params_str
            })

            print(f"  > {name}: RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
            df_results = pd.DataFrame(fold_records)
            df_results.to_csv(f"fold_{fold_i + 1}_" + RESULTS_CSV, index=False)

    # --- 3. SAVE RESULTS ---
    print("\n💾 Saving CSV logs...")
    df_results = pd.DataFrame(fold_records)
    df_results.to_csv(RESULTS_CSV, index=False)

    # Calculate Mean and Std for RMSE, MAE, and R2
    summary = df_results.groupby("Model")[["RMSE", "MAE", "R2"]].agg(['mean', 'std'])
    print("\n📊 Final 10-Fold Statistics:")
    print(summary)
    summary.to_csv(SUMMARY_CSV)

    # --- 4. GENERATE PLOTS ---
    print("\n🎨 Generating Paper Plots...")

    agg_results_for_plots = {}
    for name in full_predictions.keys():
        y_pred_full = full_predictions[name]
        mask = ~np.isnan(y_pred_full)

        # Calculate overall metrics on reconstructed full dataset
        agg_results_for_plots[name] = {
            "RMSE": np.sqrt(mean_squared_error(y[mask], y_pred_full[mask])),
            "MAE": mean_absolute_error(y[mask], y_pred_full[mask]),  # <--- NEW
            "R²": r2_score(y[mask], y_pred_full[mask])
        }

    # Pass the FULL arrays to existing plotting functions
    plot_prediction_scatter_grid(full_predictions, agg_results_for_plots, y, save_plots=True)
    plot_geospatial_error_maps(full_predictions, agg_results_for_plots, X, y, save_plots=True)


if __name__ == "__main__":
    run_strict_kfold_experiment()
