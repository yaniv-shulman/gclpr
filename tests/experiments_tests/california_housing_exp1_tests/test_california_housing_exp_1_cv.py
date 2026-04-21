"""Tests for California Housing Experiment 1."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

matplotlib.use("Agg")

from gclpr.experiments.california_housing_exp1 import california_housing_exp_1_cv as exp1_cv
from gclpr.experiments.california_housing_exp1 import california_housling_exp_1 as exp1_utils


class FakeSearch:
    """Minimal fitted search object for tests."""

    def __init__(self, estimator: BaseEstimator, best_params: dict[str, object]) -> None:
        self.best_estimator_ = estimator
        self.best_params_ = best_params
        self.best_score_ = -0.123


class OffsetEstimator(BaseEstimator, RegressorMixin):
    """Simple estimator used to make fold-level behavior deterministic."""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OffsetEstimator":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.mean_ + self.offset, dtype=float)


@pytest.fixture
def fake_frame() -> pd.DataFrame:
    """Create a fake California Housing frame."""
    return pd.DataFrame(
        {
            "MedInc": [1.0, 2.0, 3.0, 4.0],
            "AveRooms": [5.0, 6.0, 7.0, 8.0],
            "Latitude": [34.0, 35.0, 36.0, 37.0],
            "Longitude": [-118.0, -119.0, -120.0, -121.0],
            "MedHouseVal": [10.0, 11.0, 12.0, 13.0],
        }
    )


def test_load_raw_data_uses_repo_data_dir_and_supports_subsampling(
    monkeypatch: pytest.MonkeyPatch,
    fake_frame: pd.DataFrame,
) -> None:
    """The dataset loader should read from the repo data cache and subsample."""

    recorded: dict[str, object] = {}

    def fake_fetch_california_housing(*, as_frame: bool, data_home: str) -> SimpleNamespace:
        recorded["as_frame"] = as_frame
        recorded["data_home"] = data_home
        return SimpleNamespace(frame=fake_frame)

    monkeypatch.setattr(exp1_cv, "fetch_california_housing", fake_fetch_california_housing)

    x, y = exp1_cv.load_raw_data(max_samples=2, seed=7)

    assert recorded["as_frame"] is True
    assert Path(cast(str, recorded["data_home"])) == exp1_cv._data_dir
    assert x.shape == (2, 4)
    assert y.shape == (2,)


def test_scale_non_geo_features_only_changes_first_two_columns() -> None:
    """Only the non-geospatial columns should be standardized."""

    x_train = np.array([[1.0, 2.0, 34.0, -118.0], [3.0, 6.0, 35.0, -117.0]])
    x_test = np.array([[2.0, 4.0, 40.0, -100.0]])

    x_train_scaled, x_test_scaled = exp1_cv.scale_non_geo_features(x_train, x_test)

    assert np.allclose(np.mean(x_train_scaled[:, :2], axis=0), [0.0, 0.0])
    assert np.allclose(np.std(x_train_scaled[:, :2], axis=0), [1.0, 1.0])
    assert np.allclose(x_train_scaled[:, 2:], x_train[:, 2:])
    assert np.allclose(x_test_scaled[:, 2:], x_test[:, 2:])


def test_build_geo_kernel_grid_and_haversine_factory() -> None:
    """Kernel builders should produce the expected combinations and shapes."""

    grid = exp1_cv.build_geo_kernel_grid([5.0, 10.0])
    assert len(grid) == 4

    kernel = exp1_utils.haversine_rbf_kernel_factory(length_scale_km=5.0)
    x0 = np.array([[0.0, 0.0, 34.0, -118.0]])
    xn = np.array([[0.0, 0.0, 34.0, -118.0], [0.0, 0.0, 35.0, -118.0]])
    weights = kernel(x0, xn, np.zeros(2), 0, np.array([0, 1]))

    assert weights.shape == (1, 2)
    assert weights[0, 0] >= weights[0, 1]

    with pytest.raises(ValueError):
        exp1_utils.haversine_rbf_kernel_factory(length_scale_km=0.0)


def test_build_search_grids_profiles_and_invalid_profile() -> None:
    """Grid profiles should expose the expected search spaces."""

    full = exp1_cv.build_search_grids("full")
    reduced = exp1_cv.build_search_grids("reduced")
    smoke = exp1_cv.build_search_grids("smoke")

    assert set(full) == set(exp1_cv._model_order)
    assert set(reduced) == set(exp1_cv._model_order)
    assert set(smoke) == set(exp1_cv._model_order)
    assert "leaf_size" in full["knn"]
    assert smoke["lpr"][0]["size_neighborhood"] == [11]

    with pytest.raises(ValueError):
        exp1_cv.build_search_grids("bad")


def test_compute_metrics_strict_rmse_and_scoring_resolution() -> None:
    """Metric helpers should reject invalid predictions and build the scorer."""

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.nan, 4.0])

    with pytest.raises(ValueError):
        exp1_cv.compute_metrics(y_true, y_pred)

    assert exp1_cv.strict_rmse(y_true, y_pred) == float("inf")
    assert exp1_cv.strict_rmse(y_true, np.array([np.nan, np.nan, np.nan])) == float("inf")
    assert exp1_cv.strict_rmse(y_true, np.array([1.0, 2.0, 4.0])) == pytest.approx(np.sqrt(1.0 / 3.0))

    valid, non_finite_fraction = exp1_cv.prediction_validity(y_pred)
    assert valid is False
    assert non_finite_fraction == pytest.approx(1.0 / 3.0)

    for scoring in ("strict_rmse",):
        scorer = exp1_cv.resolve_scoring(scoring)
        assert callable(scorer)

    assert exp1_cv.resolve_scoring("neg_mean_squared_error") == "neg_mean_squared_error"

    with pytest.raises(ValueError):
        exp1_cv.resolve_scoring("bad_scoring")


def test_normalize_config_smoke_mode_overrides_fields() -> None:
    """Smoke mode should force the small debug configuration."""

    config = exp1_cv.normalize_config(exp1_cv.ExperimentConfig(smoke=True, max_samples=None))

    assert config.grid_profile == "smoke"
    assert config.outer_folds == 2
    assert config.inner_folds == 2
    assert config.generate_plots is False
    assert config.max_samples == 800
    assert config.output_prefix.endswith("_smoke")

    with pytest.raises(ValueError):
        exp1_cv.normalize_config(exp1_cv.ExperimentConfig(scoring="bad_scoring"))


def test_run_search_dispatches_to_selected_search_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model search dispatcher should pass resolved scoring through."""

    observed: dict[str, object] = {}

    def fake_search(**kwargs: object) -> str:
        observed.update(kwargs)
        return "search"

    def fake_cv(
        x_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: object,
        cv: int,
        n_jobs: int,
        verbose: int,
        search_mode: str,
        n_iter: int | None,
        random_state: int,
        scoring: object,
    ) -> str:
        return fake_search(
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

    monkeypatch.setitem(exp1_cv._search_dispatch, "knn", fake_cv)
    config = exp1_cv.ExperimentConfig(search_mode="random", random_search_n_iter=5)
    result = exp1_cv.run_search("knn", np.ones((3, 4)), np.ones(3), {"knn": {"a": [1]}}, config)

    assert result == "search"
    assert observed["search_mode"] == "random"
    assert observed["n_iter"] == 5


def test_pilot_tune_models_collects_estimators_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pilot tuning should return one estimator and log row per model."""

    def fake_run_search(
        model_key: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        search_grids: dict[str, object],
        config: exp1_cv.ExperimentConfig,
    ) -> FakeSearch:
        return FakeSearch(OffsetEstimator(offset=float(len(model_key))), {"model": model_key})

    monkeypatch.setattr(exp1_cv, "run_search", fake_run_search)

    x = np.arange(80, dtype=float).reshape(20, 4)
    y = np.arange(20, dtype=float)
    estimators, params, rows = exp1_cv.pilot_tune_models(
        x,
        y,
        {key: {} for key in exp1_cv._model_order},
        exp1_cv.ExperimentConfig(pilot_fraction=0.5),
    )

    assert set(estimators) == set(exp1_cv._model_order)
    assert set(params) == set(exp1_cv._model_order)
    assert len(rows) == len(exp1_cv._model_order)


def test_evaluate_fixed_estimators_returns_fold_records() -> None:
    """Fixed-estimator evaluation should emit one record per fold and model."""

    x = np.arange(120, dtype=float).reshape(30, 4)
    y = np.linspace(0.0, 1.0, 30)
    config = exp1_cv.ExperimentConfig(outer_folds=3, generate_plots=False)
    best_estimators = {key: OffsetEstimator(offset=0.1) for key in exp1_cv._model_order}
    selected_params = {key: {"chosen": key} for key in exp1_cv._model_order}

    rows, predictions = exp1_cv.evaluate_fixed_estimators(
        x,
        y,
        best_estimators,
        selected_params,
        config,
    )

    assert len(rows) == 3 * len(exp1_cv._model_order)
    assert set(predictions) == set(exp1_cv._model_order)
    assert all(np.isfinite(predictions[key]).all() for key in predictions)
    assert all(row["Valid"] for row in rows)
    assert all(row["Non_Finite_Pct"] == 0.0 for row in rows)


def test_evaluate_fixed_estimators_marks_invalid_predictions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid fold predictions should be logged and excluded from metrics."""

    class NaNEstimator(BaseEstimator, RegressorMixin):
        def fit(self, X: np.ndarray, y: np.ndarray) -> "NaNEstimator":
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            values = np.zeros(X.shape[0], dtype=float)
            values[::2] = np.nan
            return values

    x = np.arange(120, dtype=float).reshape(30, 4)
    y = np.linspace(0.0, 1.0, 30)
    config = exp1_cv.ExperimentConfig(outer_folds=2, generate_plots=False)
    best_estimators = {key: NaNEstimator() for key in exp1_cv._model_order}
    selected_params = {key: {"chosen": key} for key in exp1_cv._model_order}

    rows, predictions = exp1_cv.evaluate_fixed_estimators(
        x,
        y,
        best_estimators,
        selected_params,
        config,
    )

    assert len(rows) == 2 * len(exp1_cv._model_order)
    assert any(not row["Valid"] for row in rows)
    assert all(np.isnan(row["RMSE"]) for row in rows)
    assert all(row["Non_Finite_Pct"] > 0.0 for row in rows)
    assert any(np.isnan(predictions[key]).any() for key in predictions)

    captured = capsys.readouterr()
    assert "invalid predictions" in captured.out


def test_evaluate_nested_cv_uses_search_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested evaluation should consume the best estimator from each search result."""

    def fake_run_search(
        model_key: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        search_grids: dict[str, object],
        config: exp1_cv.ExperimentConfig,
    ) -> FakeSearch:
        estimator = OffsetEstimator(offset=0.0).fit(x_train, y_train)
        return FakeSearch(estimator, {"model": model_key})

    monkeypatch.setattr(exp1_cv, "run_search", fake_run_search)

    x = np.arange(120, dtype=float).reshape(30, 4)
    y = np.linspace(0.0, 1.0, 30)
    rows, predictions = exp1_cv.evaluate_nested_cv(
        x,
        y,
        {key: {} for key in exp1_cv._model_order},
        exp1_cv.ExperimentConfig(outer_folds=2),
    )

    assert len(rows) == 2 * len(exp1_cv._model_order)
    assert set(predictions) == set(exp1_cv._model_order)


def test_save_outputs_writes_tables_metadata_and_summary_plot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Saving outputs should create the expected files under the output directory."""

    monkeypatch.setattr(exp1_cv, "_out_dir", tmp_path)

    plot_calls: list[str] = []

    def fake_scatter(*args: object, **kwargs: object) -> None:
        plot_calls.append("scatter")
        output_dir = cast(Path, kwargs["output_dir"])
        (output_dir / "california_housling_exp_1_scatter_grid.png").write_text("ok")

    def fake_map(*args: object, **kwargs: object) -> None:
        plot_calls.append("map")
        output_dir = cast(Path, kwargs["output_dir"])
        (output_dir / "california_housling_exp_1_geospatial_errors.png").write_text("ok")

    monkeypatch.setattr(exp1_cv, "plot_prediction_scatter_grid", fake_scatter)
    monkeypatch.setattr(exp1_cv, "plot_geospatial_error_maps", fake_map)

    rows = [
        {
            "Fold": 1,
            "Model": "KNN",
            "Protocol": "pilot_tune_then_refit",
            "RMSE": 1.0,
            "MAE": 0.9,
            "R2": 0.1,
            "Selected_Params": "{}",
        },
        {
            "Fold": 1,
            "Model": "LPR",
            "Protocol": "pilot_tune_then_refit",
            "RMSE": 0.8,
            "MAE": 0.7,
            "R2": 0.2,
            "Selected_Params": "{}",
        },
    ]
    predictions = {key: np.array([1.0, 2.0]) for key in exp1_cv._model_order}
    config = exp1_cv.ExperimentConfig(generate_plots=True, output_prefix="unit")

    exp1_cv.save_outputs(
        np.array([[0.0, 0.0, 34.0, -118.0], [1.0, 1.0, 35.0, -117.0]]),
        np.array([1.0, 2.0]),
        rows,
        predictions,
        config,
        pilot_rows=[{"Model": "KNN"}],
    )

    assert (tmp_path / "unit_raw.csv").exists()
    assert (tmp_path / "unit_summary.csv").exists()
    assert (tmp_path / "unit_metadata.json").exists()
    assert (tmp_path / "unit_pilot_selection.csv").exists()
    assert (tmp_path / "unit_rmse_boxplot.png").exists()
    assert plot_calls == ["scatter", "map"]


def test_save_outputs_summaries_only_valid_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Summary outputs should exclude invalid rows from aggregated statistics."""

    monkeypatch.setattr(exp1_cv, "_out_dir", tmp_path)
    monkeypatch.setattr(exp1_cv, "plot_prediction_scatter_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp1_cv, "plot_geospatial_error_maps", lambda *args, **kwargs: None)

    rows = [
        {
            "Fold": 1,
            "Model": "KNN",
            "Protocol": "pilot_tune_then_refit",
            "Valid": True,
            "Non_Finite_Pct": 0.0,
            "RMSE": 1.0,
            "MAE": 0.9,
            "R2": 0.1,
            "Selected_Params": "{}",
        },
        {
            "Fold": 2,
            "Model": "KNN",
            "Protocol": "pilot_tune_then_refit",
            "Valid": False,
            "Non_Finite_Pct": 50.0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Selected_Params": "{}",
        },
    ]

    exp1_cv.save_outputs(
        np.array([[0.0, 0.0, 34.0, -118.0], [1.0, 1.0, 35.0, -117.0]]),
        np.array([1.0, 2.0]),
        rows,
        {"knn": np.array([1.0, np.nan])},
        exp1_cv.ExperimentConfig(generate_plots=False, output_prefix="valid_only"),
    )

    summary = pd.read_csv(tmp_path / "valid_only_summary.csv", header=[0, 1], index_col=0)
    metadata = pd.read_json(tmp_path / "valid_only_metadata.json", typ="series")

    assert summary.loc["KNN", ("RMSE", "mean")] == pytest.approx(1.0)
    assert bool(metadata["reporting_uses_valid_runs_only"]) is True


def test_run_experiment_routes_to_selected_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The top-level runner should choose the requested protocol."""

    x = np.arange(120, dtype=float).reshape(30, 4)
    y = np.linspace(0.0, 1.0, 30)

    monkeypatch.setattr(exp1_cv, "load_raw_data", lambda max_samples=None, seed=42: (x, y))
    monkeypatch.setattr(exp1_cv, "build_search_grids", lambda grid_profile="full": {"grid": grid_profile})

    called: dict[str, int] = {"pilot": 0, "nested": 0, "save": 0}

    def fake_pilot(
        x: np.ndarray,
        y: np.ndarray,
        search_grids: dict[str, object],
        config: exp1_cv.ExperimentConfig,
    ) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, object]], list[dict[str, object]]]:
        called["pilot"] += 1
        estimators: dict[str, BaseEstimator] = {key: OffsetEstimator() for key in exp1_cv._model_order}
        params: dict[str, dict[str, object]] = {key: {"p": key} for key in exp1_cv._model_order}
        return estimators, params, [{"Model": "pilot"}]

    def fake_fixed(
        x: np.ndarray,
        y: np.ndarray,
        best_estimators: dict[str, BaseEstimator],
        selected_params: dict[str, dict[str, object]],
        config: exp1_cv.ExperimentConfig,
    ) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        return (
            [{"Fold": 1, "Model": "KNN", "RMSE": 1.0, "MAE": 1.0, "R2": 0.0}],
            {key: np.zeros(len(y)) for key in exp1_cv._model_order},
        )

    def fake_nested(
        x: np.ndarray,
        y: np.ndarray,
        search_grids: dict[str, object],
        config: exp1_cv.ExperimentConfig,
    ) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        called["nested"] += 1
        return (
            [{"Fold": 1, "Model": "KNN", "RMSE": 1.0, "MAE": 1.0, "R2": 0.0}],
            {key: np.zeros(len(y)) for key in exp1_cv._model_order},
        )

    def fake_save(*args: object, **kwargs: object) -> None:
        called["save"] += 1

    monkeypatch.setattr(exp1_cv, "pilot_tune_models", fake_pilot)
    monkeypatch.setattr(exp1_cv, "evaluate_fixed_estimators", fake_fixed)
    monkeypatch.setattr(exp1_cv, "evaluate_nested_cv", fake_nested)
    monkeypatch.setattr(exp1_cv, "save_outputs", fake_save)

    df = exp1_cv.run_experiment(exp1_cv.ExperimentConfig(protocol="pilot_tune_then_refit", generate_plots=False))
    assert called["pilot"] == 1
    assert called["nested"] == 0
    assert called["save"] == 1
    assert not df.empty

    called["pilot"] = 0
    called["nested"] = 0
    called["save"] = 0
    df_nested = exp1_cv.run_experiment(exp1_cv.ExperimentConfig(protocol="nested_cv", generate_plots=False))
    assert called["pilot"] == 0
    assert called["nested"] == 1
    assert called["save"] == 1
    assert not df_nested.empty


def test_rsklpr_wrapper_fit_and_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scikit-learn wrapper should delegate fit and predict to ``Rsklpr``."""

    class FakeRsklpr:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.fitted = False

        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            self.fitted = True

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], 3.14)

    monkeypatch.setattr(exp1_utils, "Rsklpr", FakeRsklpr)

    wrapper = exp1_utils.RsklprWrapper(size_neighborhood=7, degree=1)
    fitted = wrapper.fit(np.ones((4, 4)), np.arange(4, dtype=float))
    preds = fitted.predict(np.ones((3, 4)))

    assert fitted is wrapper
    assert np.allclose(preds, 3.14)


def test_search_space_size_and_run_search_cv_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helper search utilities should cover grid, random, and invalid modes."""

    assert exp1_utils._search_space_size({"offset": [0.0, 1.0]}) == 2
    assert exp1_utils._search_space_size(object()) == 0

    recorded: dict[str, object] = {}

    class FakeGridSearch:
        def __init__(self, **kwargs: object) -> None:
            recorded["grid_init"] = kwargs

        def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
            recorded["grid_fit"] = (x_train.copy(), y_train.copy())

    class FakeRandomSearch:
        def __init__(self, **kwargs: object) -> None:
            recorded["random_init"] = kwargs

        def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
            recorded["random_fit"] = (x_train.copy(), y_train.copy())

    monkeypatch.setattr(exp1_utils, "GridSearchCV", FakeGridSearch)
    monkeypatch.setattr(exp1_utils, "RandomizedSearchCV", FakeRandomSearch)
    monkeypatch.setattr(exp1_utils, "_search_space_size", lambda param_grid: 7)

    x = np.ones((5, 4))
    y = np.arange(5, dtype=float)
    model = OffsetEstimator()

    grid_search = exp1_utils._run_search_cv(
        model=model,
        x_train=x,
        y_train=y,
        param_grid={"offset": [0.0]},
        cv=2,
        n_jobs=1,
        search_mode="grid",
    )
    assert isinstance(grid_search, FakeGridSearch)
    grid_init = cast(dict[str, Any], recorded["grid_init"])
    assert grid_init["estimator"] is model
    assert grid_init["param_grid"] == {"offset": [0.0]}
    assert "grid_fit" in recorded

    random_search = exp1_utils._run_search_cv(
        model=model,
        x_train=x,
        y_train=y,
        param_grid={"offset": [0.0, 1.0]},
        cv=2,
        n_jobs=1,
        search_mode="random",
        n_iter=None,
        random_state=11,
    )
    assert isinstance(random_search, FakeRandomSearch)
    random_init = cast(dict[str, Any], recorded["random_init"])
    assert random_init["estimator"] is model
    assert random_init["n_iter"] == 7
    assert random_init["random_state"] == 11
    assert "random_fit" in recorded

    with pytest.raises(ValueError):
        exp1_utils._run_search_cv(
            model=model,
            x_train=x,
            y_train=y,
            param_grid={"offset": [0.0]},
            cv=2,
            n_jobs=1,
            search_mode="bad_mode",
        )


def test_search_helpers_delegate_to_run_search_cv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search helper wrappers should delegate to the shared search implementation."""

    seen: list[type[BaseEstimator]] = []

    def fake_run_search_cv(
        model: BaseEstimator,
        x_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: object,
        cv: int,
        n_jobs: int,
        verbose: int = 0,
        search_mode: str = "grid",
        n_iter: int | None = None,
        random_state: int = 42,
        scoring: object = "neg_mean_squared_error",
    ) -> str:
        seen.append(type(model))
        return "ok"

    monkeypatch.setattr(exp1_utils, "_run_search_cv", fake_run_search_cv)

    x = np.ones((6, 4))
    y = np.arange(6, dtype=float)
    assert exp1_utils.cv_knn(x, y, {"n_neighbors": [3]}, cv=2, n_jobs=1) == "ok"
    assert exp1_utils.cv_rsklpr(x, y, [{"size_neighborhood": [3]}], cv=2, n_jobs=1) == "ok"
    assert exp1_utils.cv_gclpr_geo(x, y, [{"size_neighborhood": [3]}], cv=2, n_jobs=1) == "ok"
    assert seen[0].__name__ == "KNeighborsRegressor"
    assert seen[1].__name__ == "RsklprWrapper"
    assert seen[2].__name__ == "RsklprWrapper"


def test_plot_helpers_call_show_when_not_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both plotting helpers should use ``plt.show`` in interactive mode."""

    show_calls: list[str] = []
    monkeypatch.setattr(exp1_utils.plt, "show", lambda: show_calls.append("show"))

    predictions = {
        "knn": np.array([1.0, 2.0, 3.0]),
        "lpr": np.array([1.0, 2.5, 3.0]),
        "rsklpr": np.array([1.0, 2.0, 2.5]),
        "gclpr_geo": np.array([1.1, 2.0, 3.1]),
        "grclpr_geo": np.array([0.9, 2.1, 2.9]),
    }
    results = {key: {"RMSE": 1.0, "R²": 0.5} for key in predictions}
    x = np.array(
        [
            [0.0, 0.0, 34.0, -118.0],
            [1.0, 1.0, 35.0, -117.0],
            [2.0, 2.0, 36.0, -116.0],
        ]
    )
    y = np.array([1.0, 2.0, 3.0])

    exp1_utils.plot_prediction_scatter_grid(
        predictions,
        results,
        y,
        save_plots=False,
    )
    exp1_utils.plot_geospatial_error_maps(
        predictions,
        results,
        x,
        y,
        save_plots=False,
    )

    assert show_calls == ["show", "show"]


def test_plot_geospatial_error_maps_skips_when_models_missing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The geospatial plot helper should return early when required models are absent."""

    show_calls: list[str] = []
    monkeypatch.setattr(exp1_utils.plt, "show", lambda: show_calls.append("show"))

    exp1_utils.plot_geospatial_error_maps(
        predictions={"lpr": np.array([1.0, 2.0])},
        results={"lpr": {"RMSE": 1.0}},
        x_test=np.array([[0.0, 0.0, 34.0, -118.0], [1.0, 1.0, 35.0, -117.0]]),
        y_test=np.array([1.0, 2.0]),
        save_plots=True,
        output_dir=tmp_path,
    )

    captured = capsys.readouterr()
    assert "Skipping geospatial plot" in captured.out
    assert show_calls == []
    assert not (tmp_path / "california_housling_exp_1_geospatial_errors.png").exists()


def test_plot_helpers_save_expected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plot helpers should save the paper figure filenames."""

    monkeypatch.setattr(exp1_utils.plt, "show", lambda: None)

    predictions = {
        "knn": np.array([1.0, 2.0, 3.0]),
        "lpr": np.array([1.0, 2.0, 3.0]),
        "rsklpr": np.array([1.0, 2.0, 3.0]),
        "gclpr_geo": np.array([1.0, 2.0, 3.0]),
        "grclpr_geo": np.array([1.0, 2.0, 3.0]),
    }
    results = {key: {"RMSE": 1.0, "R²": 0.5} for key in predictions}
    x = np.array(
        [
            [0.0, 0.0, 34.0, -118.0],
            [1.0, 1.0, 35.0, -117.0],
            [2.0, 2.0, 36.0, -116.0],
        ]
    )
    y = np.array([1.0, 2.0, 3.0])

    exp1_utils.plot_prediction_scatter_grid(
        predictions,
        results,
        y,
        save_plots=True,
        output_dir=tmp_path,
    )
    exp1_utils.plot_geospatial_error_maps(
        predictions,
        results,
        x,
        y,
        save_plots=True,
        output_dir=tmp_path,
    )

    assert (tmp_path / "california_housling_exp_1_scatter_grid.png").exists()
    assert (tmp_path / "california_housling_exp_1_geospatial_errors.png").exists()
