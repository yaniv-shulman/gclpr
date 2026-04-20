"""Tests for NYC Airbnb Experiment 2."""

from pathlib import Path
from typing import cast

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

matplotlib.use("Agg")

from grclpr.experiments.nyc_airbnb_exp2 import nyc_airbnb as exp2_utils
from grclpr.experiments.nyc_airbnb_exp2 import nyc_airbnb_exp_2_cv as exp2_cv


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

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OffsetEstimator":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.mean_ + self.offset, dtype=float)


@pytest.fixture
def subway_graph() -> nx.Graph:
    """Create a small subway graph for tests."""

    graph = nx.Graph()
    graph.add_node("A", name="Alpha", lat=40.0, lon=-73.9)
    graph.add_node("B", name="Bravo", lat=40.01, lon=-73.91)
    graph.add_node("C", name="Charlie", lat=40.02, lon=-73.92)
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    return graph


@pytest.fixture
def airbnb_frame() -> pd.DataFrame:
    """Create a small Airbnb-like frame."""

    return pd.DataFrame(
        {
            "host_listings_count": [1.0, 2.0, 3.0, 4.0],
            "latitude": [40.0, 40.01, 40.02, 40.03],
            "longitude": [-73.9, -73.91, -73.92, -73.93],
            "room_type": ["Entire home/apt", "Private room", "Entire home/apt", "Private room"],
            "price": ["$100.00", "$200.00", "$300.00", "$400.00"],
            "minimum_nights": [1, 2, 3, 4],
            "number_of_reviews": [10.0, np.nan, 30.0, 40.0],
            "calculated_host_listings_count": [1.0, 2.0, np.nan, 4.0],
            "availability_365": [100.0, 200.0, 300.0, np.nan],
        }
    )


def test_load_subway_graph_uses_given_path(
    monkeypatch: pytest.MonkeyPatch,
    subway_graph: nx.Graph,
) -> None:
    """The subway graph loader should delegate to ``networkx.read_graphml``."""

    recorded: dict[str, object] = {}

    def fake_read_graphml(path: Path) -> nx.Graph:
        recorded["path"] = path
        return subway_graph

    monkeypatch.setattr(exp2_utils.nx, "read_graphml", fake_read_graphml)
    graph = exp2_utils.load_subway_graph(Path("example.graphml"))

    assert graph is subway_graph
    assert recorded["path"] == Path("example.graphml")


def test_haversine_station_map_and_kernel_factory(
    subway_graph: nx.Graph,
) -> None:
    """Station mapping and the subway kernel should behave as expected."""

    coords = pd.DataFrame(
        {
            "latitude": [40.0, 40.01, 40.5],
            "longitude": [-73.9, -73.91, -73.5],
        }
    )

    assert exp2_utils.haversine(40.0, -73.9, 40.0, -73.9) == pytest.approx(0.0)

    station_map = exp2_utils.calculate_unified_index_to_station_map(
        subway_graph=subway_graph,
        listing_coords=coords,
        max_station_dist_km=2.0,
    )
    assert station_map[0] == "A"
    assert station_map[1] == "B"
    assert station_map[2] is None

    kernel = exp2_utils.subway_shortest_path_kernel_factory(
        graph=subway_graph,
        original_idx_to_station_map=station_map,
        train_set_original_indices=np.array([0, 1, 2]),
        predict_set_original_indices=np.array([0, 1]),
        distance_scale=1.0,
    )

    weights = kernel(
        np.zeros((1, 2)),
        np.zeros((3, 2)),
        np.zeros(3),
        0,
        np.array([[0, 1, 2]]),
    )
    assert weights.shape == (1, 3)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(np.exp(-1.0))
    assert weights[0, 2] == pytest.approx(exp2_utils._epsilon_similarity)

    with pytest.raises(TypeError):
        exp2_utils.subway_shortest_path_kernel_factory(
            graph=subway_graph,
            original_idx_to_station_map=station_map,
            train_set_original_indices=np.array([[0, 1]]),
            predict_set_original_indices=np.array([0, 1]),
            distance_scale=1.0,
        )


def test_plot_subway_kernel_similarity_save_and_missing_station(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    subway_graph: nx.Graph,
    tmp_path: Path,
) -> None:
    """Kernel similarity plots should save files and skip unknown stations."""

    monkeypatch.setattr(exp2_utils.plt, "show", lambda: None)

    exp2_utils.plot_subway_kernel_similarity(
        subway_graph=subway_graph,
        reference_station_id="Alpha",
        distance_scale=1.0,
        save_plot=True,
        output_dir=tmp_path,
    )
    exp2_utils.plot_subway_kernel_similarity(
        subway_graph=subway_graph,
        reference_station_id="Alpha",
        distance_scale=1.0,
        save_plot=True,
        output_dir=tmp_path,
        zoom_graph_radius=1,
    )

    assert (tmp_path / "subway_kernel_similarity.png").exists()
    assert (tmp_path / "subway_kernel_similarity_zoomed_1.png").exists()

    exp2_utils.plot_subway_kernel_similarity(
        subway_graph=subway_graph,
        reference_station_id="Missing",
        distance_scale=1.0,
        save_plot=True,
        output_dir=tmp_path,
    )
    captured = capsys.readouterr()
    assert "Could not resolve station" in captured.out


def test_load_raw_data_uses_repo_paths_and_cleans(
    monkeypatch: pytest.MonkeyPatch,
    airbnb_frame: pd.DataFrame,
) -> None:
    """The NYC loader should clean price, encode room type, and support subsampling."""

    recorded: dict[str, object] = {}

    def fake_read_csv(path: Path) -> pd.DataFrame:
        recorded["path"] = path
        return airbnb_frame.copy()

    monkeypatch.setattr(exp2_cv.pd, "read_csv", fake_read_csv)

    x, y_log, y_price, coords, scale_indices, feature_names = exp2_cv.load_raw_data(
        max_samples=3,
        seed=7,
    )

    assert recorded["path"] == exp2_cv._airbnb_data_path
    assert x.shape[0] == 3
    assert y_log.shape == (3,)
    assert y_price.shape == (3,)
    assert list(coords.columns) == ["latitude", "longitude"]
    assert len(scale_indices) > 0
    assert "latitude" in feature_names and "longitude" in feature_names


def test_scale_search_grids_and_small_helpers() -> None:
    """Small runner helpers should behave consistently."""

    x_train = np.array([[1.0, 2.0, 40.0, -73.9], [3.0, 6.0, 40.1, -74.0]])
    x_test = np.array([[2.0, 4.0, 41.0, -73.5]])

    x_train_scaled, x_test_scaled = exp2_cv.scale_non_geo_features(
        x_train=x_train,
        x_test=x_test,
        scale_indices=[0, 1],
    )

    assert np.allclose(np.mean(x_train_scaled[:, :2], axis=0), [0.0, 0.0])
    assert np.allclose(x_train_scaled[:, 2:], x_train[:, 2:])
    assert np.allclose(x_test_scaled[:, 2:], x_test[:, 2:])

    full = exp2_cv.build_search_grids("full")
    reduced = exp2_cv.build_search_grids("reduced")
    smoke = exp2_cv.build_search_grids("smoke")
    assert set(full) == set(exp2_cv._model_order)
    assert set(reduced) == set(exp2_cv._model_order)
    assert set(smoke) == set(exp2_cv._model_order)

    with pytest.raises(ValueError):
        exp2_cv.build_search_grids("bad")

    assert exp2_cv.uses_subway_context("gclpr_subway") is True
    assert exp2_cv.uses_subway_context("lpr") is False

    augmented = exp2_cv.augment_with_original_indices(
        np.ones((2, 3)),
        np.array([10, 11]),
    )
    assert augmented.shape == (2, 4)
    assert np.allclose(augmented[:, -1], [10.0, 11.0])

    prices = exp2_cv.inverse_transform_log_predictions(np.array([0.0, np.log(2.0) - 1.0]))
    assert np.all(prices >= 0.0)


def test_compute_metrics_strict_rmse_and_scoring_resolution() -> None:
    """Metric helpers should reject invalid predictions and build the scorer."""

    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, np.nan, 290.0])
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.array([110.0, 190.0, 290.0]))

    with pytest.raises(ValueError):
        exp2_cv.compute_metrics(y_true, y_pred)

    assert exp2_cv.strict_rmse(y_true, y_pred) == float("inf")
    assert exp2_cv.strict_rmse(y_true, np.array([100.0, 200.0, 290.0])) == pytest.approx(np.sqrt(100.0 / 3.0))
    assert exp2_cv.raw_price_rmse_from_log(y_true_log, np.array([0.0, np.nan, 0.0])) == float("inf")
    assert exp2_cv.raw_price_rmse_from_log(y_true_log, y_pred_log) == pytest.approx(10.0)

    valid, non_finite_fraction = exp2_cv.prediction_validity(y_pred)
    assert valid is False
    assert non_finite_fraction == pytest.approx(1.0 / 3.0)

    for scoring in (
        "raw_price_rmse_from_log",
        "strict_log_rmse",
    ):
        scorer = exp2_cv.resolve_scoring(scoring)
        assert callable(scorer)
    assert exp2_cv.resolve_scoring("neg_mean_squared_error") == "neg_mean_squared_error"
    with pytest.raises(ValueError):
        exp2_cv.resolve_scoring("bad_scoring")


def test_normalize_config_smoke_mode_overrides_fields() -> None:
    """Smoke mode should force the small debug configuration."""

    config = exp2_cv.normalize_config(exp2_cv.ExperimentConfig(smoke=True, max_samples=None))

    assert config.grid_profile == "smoke"
    assert config.outer_folds == 2
    assert config.inner_folds == 2
    assert config.generate_plots is False
    assert config.max_samples == 1000
    assert config.output_prefix.endswith("_smoke")

    with pytest.raises(ValueError):
        exp2_cv.normalize_config(exp2_cv.ExperimentConfig(scoring="bad_scoring"))
    with pytest.raises(ValueError):
        exp2_cv.normalize_config(exp2_cv.ExperimentConfig(holdout_fraction=0.0))
    with pytest.raises(ValueError):
        exp2_cv.normalize_config(exp2_cv.ExperimentConfig(holdout_repeats=0))


def test_run_search_dispatches_to_standard_and_subway_search(
    monkeypatch: pytest.MonkeyPatch,
    subway_graph: nx.Graph,
) -> None:
    """The model search dispatcher should route subway and non-subway models correctly."""

    observed: dict[str, dict[str, object]] = {}

    def fake_standard(**kwargs: object) -> str:
        observed["standard"] = cast(dict[str, object], kwargs)
        return "standard"

    def fake_subway(**kwargs: object) -> str:
        observed["subway"] = cast(dict[str, object], kwargs)
        return "subway"

    monkeypatch.setitem(exp2_cv._search_dispatch, "knn", fake_standard)
    monkeypatch.setitem(exp2_cv._search_dispatch, "gclpr_subway", fake_subway)

    config = exp2_cv.ExperimentConfig(search_mode="random", random_search_n_iter=5)
    station_map: dict[int, str | None] = {0: "A"}

    standard_result = exp2_cv.run_search(
        model_key="knn",
        x_train=np.ones((3, 4)),
        y_train_log=np.ones(3),
        train_indices=np.array([0, 1, 2]),
        subway_graph=subway_graph,
        original_idx_to_station_map=station_map,
        search_grids={"knn": {"a": [1]}},
        config=config,
    )
    subway_result = exp2_cv.run_search(
        model_key="gclpr_subway",
        x_train=np.ones((3, 4)),
        y_train_log=np.ones(3),
        train_indices=np.array([0, 1, 2]),
        subway_graph=subway_graph,
        original_idx_to_station_map=station_map,
        search_grids={"gclpr_subway": {"a": [1]}},
        config=config,
    )

    assert standard_result == "standard"
    assert subway_result == "subway"
    assert observed["standard"]["search_mode"] == "random"
    assert observed["subway"]["train_indices"] is not None


def test_pilot_tune_models_collects_estimators_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    subway_graph: nx.Graph,
) -> None:
    """Pilot tuning should return one estimator and log row per model."""

    def fake_run_search(
        model_key: str,
        x_train: np.ndarray,
        y_train_log: np.ndarray,
        train_indices: np.ndarray,
        subway_graph: nx.Graph,
        original_idx_to_station_map: dict[int, str | None],
        search_grids: dict[str, object],
        config: exp2_cv.ExperimentConfig,
    ) -> FakeSearch:
        return FakeSearch(OffsetEstimator(offset=float(len(model_key))), {"model": model_key})

    monkeypatch.setattr(exp2_cv, "run_search", fake_run_search)

    x = np.arange(80, dtype=float).reshape(20, 4)
    y_log = np.arange(20, dtype=float)
    estimators, params, rows = exp2_cv.pilot_tune_models(
        x=x,
        y_log=y_log,
        scale_indices=[0, 1],
        subway_graph=subway_graph,
        original_idx_to_station_map={index: "A" for index in range(20)},
        search_grids={key: {} for key in exp2_cv._model_order},
        config=exp2_cv.ExperimentConfig(pilot_fraction=0.5),
    )

    assert set(estimators) == set(exp2_cv._model_order)
    assert set(params) == set(exp2_cv._model_order)
    assert len(rows) == len(exp2_cv._model_order)


def test_evaluate_fixed_estimators_and_invalid_predictions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fixed-estimator evaluation should handle valid and invalid predictions."""

    class NaNEstimator(BaseEstimator, RegressorMixin):
        def fit(self, x: np.ndarray, y: np.ndarray) -> "NaNEstimator":
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            values = np.zeros(x.shape[0], dtype=float)
            values[::2] = np.nan
            return values

    x = np.arange(120, dtype=float).reshape(30, 4)
    y_log = np.linspace(0.0, 1.0, 30)
    y_price = np.linspace(100.0, 200.0, 30)

    config = exp2_cv.ExperimentConfig(outer_folds=2, generate_plots=False)
    best_estimators = {key: OffsetEstimator(offset=0.1) for key in exp2_cv._model_order}
    selected_params = {key: {"chosen": key} for key in exp2_cv._model_order}

    rows, predictions = exp2_cv.evaluate_fixed_estimators(
        x=x,
        y_log=y_log,
        y_price=y_price,
        scale_indices=[0, 1],
        best_estimators=best_estimators,
        selected_params=selected_params,
        config=config,
    )

    assert len(rows) == 2 * len(exp2_cv._model_order)
    assert set(predictions) == set(exp2_cv._model_order)
    assert all(row["Valid"] for row in rows)

    nan_estimators = {key: NaNEstimator() for key in exp2_cv._model_order}
    bad_rows, bad_predictions = exp2_cv.evaluate_fixed_estimators(
        x=x,
        y_log=y_log,
        y_price=y_price,
        scale_indices=[0, 1],
        best_estimators=nan_estimators,
        selected_params=selected_params,
        config=config,
    )
    assert any(not row["Valid"] for row in bad_rows)
    assert all(np.isnan(row["RMSE"]) for row in bad_rows)
    assert any(np.isnan(bad_predictions[key]).any() for key in bad_predictions)

    captured = capsys.readouterr()
    assert "invalid predictions" in captured.out


def test_evaluate_nested_cv_uses_search_results(
    monkeypatch: pytest.MonkeyPatch,
    subway_graph: nx.Graph,
) -> None:
    """Nested evaluation should consume the best estimator from each search result."""

    def fake_run_search(
        model_key: str,
        x_train: np.ndarray,
        y_train_log: np.ndarray,
        train_indices: np.ndarray,
        subway_graph: nx.Graph,
        original_idx_to_station_map: dict[int, str | None],
        search_grids: dict[str, object],
        config: exp2_cv.ExperimentConfig,
    ) -> FakeSearch:
        estimator = OffsetEstimator(offset=0.0).fit(x_train, y_train_log)
        return FakeSearch(estimator, {"model": model_key})

    monkeypatch.setattr(exp2_cv, "run_search", fake_run_search)

    x = np.arange(120, dtype=float).reshape(30, 4)
    y_log = np.linspace(0.0, 1.0, 30)
    y_price = np.linspace(100.0, 200.0, 30)
    rows, predictions = exp2_cv.evaluate_nested_cv(
        x=x,
        y_log=y_log,
        y_price=y_price,
        scale_indices=[0, 1],
        subway_graph=subway_graph,
        original_idx_to_station_map={index: "A" for index in range(30)},
        search_grids={key: {} for key in exp2_cv._model_order},
        config=exp2_cv.ExperimentConfig(outer_folds=2),
    )

    assert len(rows) == 2 * len(exp2_cv._model_order)
    assert set(predictions) == set(exp2_cv._model_order)


def test_evaluate_holdout_cv_uses_search_results(
    monkeypatch: pytest.MonkeyPatch,
    subway_graph: nx.Graph,
) -> None:
    """Holdout evaluation should tune on train and score each held-out repeat."""

    def fake_run_search(
        model_key: str,
        x_train: np.ndarray,
        y_train_log: np.ndarray,
        train_indices: np.ndarray,
        subway_graph: nx.Graph,
        original_idx_to_station_map: dict[int, str | None],
        search_grids: dict[str, object],
        config: exp2_cv.ExperimentConfig,
    ) -> FakeSearch:
        estimator = OffsetEstimator(offset=0.0).fit(x_train, y_train_log)
        return FakeSearch(estimator, {"model": model_key})

    monkeypatch.setattr(exp2_cv, "run_search", fake_run_search)

    x = np.arange(120, dtype=float).reshape(30, 4)
    y_log = np.linspace(0.0, 1.0, 30)
    y_price = np.linspace(100.0, 200.0, 30)
    rows, predictions = exp2_cv.evaluate_holdout_cv(
        x=x,
        y_log=y_log,
        y_price=y_price,
        scale_indices=[0, 1],
        subway_graph=subway_graph,
        original_idx_to_station_map={index: "A" for index in range(30)},
        search_grids={key: {} for key in exp2_cv._model_order},
        config=exp2_cv.ExperimentConfig(
            protocol="holdout_cv",
            holdout_fraction=0.2,
            holdout_repeats=2,
        ),
    )

    assert len(rows) == 2 * len(exp2_cv._model_order)
    assert set(predictions) == set(exp2_cv._model_order)
    assert sum(np.isfinite(predictions["knn"])) >= 6
    assert {row["Fold"] for row in rows} == {1, 2}


def test_save_outputs_and_valid_only_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Saving outputs should create files and summarize only valid rows."""

    monkeypatch.setattr(exp2_cv, "_out_dir", tmp_path)

    plot_calls: list[str] = []

    def fake_scatter(*args: object, **kwargs: object) -> None:
        plot_calls.append("scatter")
        output_dir = cast(Path, kwargs["output_dir"])
        (output_dir / "nyc_airbnb_exp_2_scatter_grid.png").write_text("ok")

    def fake_map(*args: object, **kwargs: object) -> None:
        plot_calls.append("map")
        output_dir = cast(Path, kwargs["output_dir"])
        (output_dir / "nyc_airbnb_exp_2_geospatial_errors.png").write_text("ok")

    monkeypatch.setattr(exp2_cv, "plot_prediction_scatter_grid", fake_scatter)
    monkeypatch.setattr(exp2_cv, "plot_geospatial_error_maps", fake_map)

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
            "Non_Finite_Pct": 25.0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Selected_Params": "{}",
        },
    ]
    coords = pd.DataFrame(
        {
            "latitude": [40.0, 40.1],
            "longitude": [-73.9, -73.8],
        }
    )
    predictions = {"knn": np.array([100.0, np.nan])}
    config = exp2_cv.ExperimentConfig(generate_plots=True, output_prefix="unit")

    exp2_cv.save_outputs(
        coords=coords,
        y_price=np.array([100.0, 200.0]),
        fold_records=rows,
        full_predictions=predictions,
        config=config,
        pilot_rows=[{"Model": "KNN"}],
    )

    assert (tmp_path / "unit_raw.csv").exists()
    assert (tmp_path / "unit_summary.csv").exists()
    assert (tmp_path / "unit_metadata.json").exists()
    assert (tmp_path / "unit_pilot_selection.csv").exists()
    assert (tmp_path / "unit_rmse_boxplot.png").exists()
    assert plot_calls == ["scatter", "map"]

    summary = pd.read_csv(tmp_path / "unit_summary.csv", header=[0, 1], index_col=0)
    metadata = pd.read_json(tmp_path / "unit_metadata.json", typ="series")
    assert summary.loc["KNN", ("RMSE", "mean")] == pytest.approx(1.0)
    assert bool(metadata["reporting_uses_valid_runs_only"]) is True

    holdout_predictions = {"knn": np.array([100.0, np.nan])}
    holdout_config = exp2_cv.ExperimentConfig(protocol="holdout_cv", generate_plots=True, output_prefix="holdout")
    exp2_cv.save_outputs(
        coords=coords,
        y_price=np.array([100.0, 200.0]),
        fold_records=rows,
        full_predictions=holdout_predictions,
        config=holdout_config,
        pilot_rows=None,
    )
    assert (tmp_path / "holdout_raw.csv").exists()
    assert (tmp_path / "holdout_summary.csv").exists()
    assert (tmp_path / "holdout_metadata.json").exists()
    assert not (tmp_path / "holdout_rmse_boxplot.png").exists()

    repeated_holdout_rows = rows + [
        {
            "Fold": 2,
            "Model": "KNN",
            "Protocol": "holdout_cv",
            "Valid": True,
            "Non_Finite_Pct": 0.0,
            "RMSE": 1.2,
            "MAE": 1.1,
            "R2": 0.2,
            "Selected_Params": "{}",
        },
    ]
    repeated_holdout_config = exp2_cv.ExperimentConfig(
        protocol="holdout_cv",
        holdout_repeats=2,
        generate_plots=True,
        output_prefix="holdout_repeated",
    )
    exp2_cv.save_outputs(
        coords=coords,
        y_price=np.array([100.0, 200.0]),
        fold_records=repeated_holdout_rows,
        full_predictions=holdout_predictions,
        config=repeated_holdout_config,
        pilot_rows=None,
    )
    assert (tmp_path / "holdout_repeated_rmse_boxplot.png").exists()


def test_run_experiment_routes_to_selected_protocol(
    monkeypatch: pytest.MonkeyPatch,
    subway_graph: nx.Graph,
) -> None:
    """The top-level runner should choose the requested protocol."""

    x = np.arange(120, dtype=float).reshape(30, 4)
    y_log = np.linspace(0.0, 1.0, 30)
    y_price = np.linspace(100.0, 200.0, 30)
    coords = pd.DataFrame(
        {
            "latitude": np.linspace(40.0, 40.5, 30),
            "longitude": np.linspace(-73.9, -73.4, 30),
        }
    )

    monkeypatch.setattr(
        exp2_cv,
        "load_raw_data",
        lambda max_samples=None, seed=42: (x, y_log, y_price, coords, [0, 1], ["a", "b"]),
    )
    monkeypatch.setattr(exp2_cv, "load_subway_graph", lambda path: subway_graph)
    monkeypatch.setattr(
        exp2_cv,
        "calculate_unified_index_to_station_map",
        lambda subway_graph, listing_coords, max_station_dist_km=1.0: {
            index: "A" for index in range(len(listing_coords))
        },
    )
    monkeypatch.setattr(exp2_cv, "build_search_grids", lambda grid_profile="full": {"grid": grid_profile})

    called: dict[str, int] = {"pilot": 0, "nested": 0, "holdout": 0, "save": 0}

    def fake_pilot(
        x: np.ndarray,
        y_log: np.ndarray,
        scale_indices: list[int],
        subway_graph: nx.Graph,
        original_idx_to_station_map: dict[int, str | None],
        search_grids: dict[str, object],
        config: exp2_cv.ExperimentConfig,
    ) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, object]], list[dict[str, object]]]:
        called["pilot"] += 1
        estimators: dict[str, BaseEstimator] = {key: OffsetEstimator() for key in exp2_cv._model_order}
        params: dict[str, dict[str, object]] = {key: {"p": key} for key in exp2_cv._model_order}
        return estimators, params, [{"Model": "pilot"}]

    def fake_fixed(
        x: np.ndarray,
        y_log: np.ndarray,
        y_price: np.ndarray,
        scale_indices: list[int],
        best_estimators: dict[str, BaseEstimator],
        selected_params: dict[str, dict[str, object]],
        config: exp2_cv.ExperimentConfig,
    ) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        return (
            [
                {
                    "Fold": 1,
                    "Model": "KNN",
                    "Valid": True,
                    "Non_Finite_Pct": 0.0,
                    "RMSE": 1.0,
                    "MAE": 1.0,
                    "R2": 0.0,
                }
            ],
            {key: np.zeros(len(y_price)) for key in exp2_cv._model_order},
        )

    def fake_nested(
        x: np.ndarray,
        y_log: np.ndarray,
        y_price: np.ndarray,
        scale_indices: list[int],
        subway_graph: nx.Graph,
        original_idx_to_station_map: dict[int, str | None],
        search_grids: dict[str, object],
        config: exp2_cv.ExperimentConfig,
    ) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        called["nested"] += 1
        return (
            [
                {
                    "Fold": 1,
                    "Model": "KNN",
                    "Valid": True,
                    "Non_Finite_Pct": 0.0,
                    "RMSE": 1.0,
                    "MAE": 1.0,
                    "R2": 0.0,
                }
            ],
            {key: np.zeros(len(y_price)) for key in exp2_cv._model_order},
        )

    def fake_holdout(
        x: np.ndarray,
        y_log: np.ndarray,
        y_price: np.ndarray,
        scale_indices: list[int],
        subway_graph: nx.Graph,
        original_idx_to_station_map: dict[int, str | None],
        search_grids: dict[str, object],
        config: exp2_cv.ExperimentConfig,
    ) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        called["holdout"] += 1
        return (
            [
                {
                    "Fold": 1,
                    "Model": "KNN",
                    "Valid": True,
                    "Non_Finite_Pct": 0.0,
                    "RMSE": 1.0,
                    "MAE": 1.0,
                    "R2": 0.0,
                }
            ],
            {key: np.zeros(len(y_price)) for key in exp2_cv._model_order},
        )

    def fake_save(*args: object, **kwargs: object) -> None:
        called["save"] += 1

    monkeypatch.setattr(exp2_cv, "pilot_tune_models", fake_pilot)
    monkeypatch.setattr(exp2_cv, "evaluate_fixed_estimators", fake_fixed)
    monkeypatch.setattr(exp2_cv, "evaluate_nested_cv", fake_nested)
    monkeypatch.setattr(exp2_cv, "evaluate_holdout_cv", fake_holdout)
    monkeypatch.setattr(exp2_cv, "save_outputs", fake_save)

    df = exp2_cv.run_experiment(exp2_cv.ExperimentConfig(protocol="pilot_tune_then_refit", generate_plots=False))
    assert called["pilot"] == 1
    assert called["nested"] == 0
    assert called["holdout"] == 0
    assert called["save"] == 1
    assert not df.empty

    called["pilot"] = 0
    called["nested"] = 0
    called["holdout"] = 0
    called["save"] = 0
    df_nested = exp2_cv.run_experiment(exp2_cv.ExperimentConfig(protocol="nested_cv", generate_plots=False))
    assert called["pilot"] == 0
    assert called["nested"] == 1
    assert called["holdout"] == 0
    assert called["save"] == 1
    assert not df_nested.empty

    called["pilot"] = 0
    called["nested"] = 0
    called["holdout"] = 0
    called["save"] = 0
    df_holdout = exp2_cv.run_experiment(exp2_cv.ExperimentConfig(protocol="holdout_cv", generate_plots=False))
    assert called["pilot"] == 0
    assert called["nested"] == 0
    assert called["holdout"] == 1
    assert called["save"] == 1
    assert not df_holdout.empty


def test_wrapper_and_search_helpers(
    monkeypatch: pytest.MonkeyPatch,
    subway_graph: nx.Graph,
) -> None:
    """Wrappers and helper search functions should delegate correctly."""

    class FakeRsklpr:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.fit_args: tuple[np.ndarray, np.ndarray] | None = None

        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            self.fit_args = (x.copy(), y.copy())

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], 3.14)

    monkeypatch.setattr(exp2_utils, "Rsklpr", FakeRsklpr)

    generic = exp2_utils.RsklprWrapper(size_neighborhood=7, degree=1)
    assert np.allclose(generic.fit(np.ones((4, 4)), np.arange(4, dtype=float)).predict(np.ones((3, 4))), 3.14)

    monkeypatch.setattr(
        exp2_utils,
        "subway_shortest_path_kernel_factory",
        lambda **kwargs: (lambda x0, xn, dists, index, indices: np.ones((1, len(np.asarray(indices).ravel())))),
    )
    subway_wrapper = exp2_utils.SubwayRsklprWrapper(
        size_neighborhood=7,
        degree=1,
        subway_graph=subway_graph,
        original_idx_to_station_map={0: "A", 1: "B", 2: "C"},
    )
    x_train = np.hstack((np.ones((4, 4)), np.arange(4).reshape(-1, 1)))
    fitted = subway_wrapper.fit(x_train, np.arange(4, dtype=float))
    preds = fitted.predict(np.hstack((np.ones((3, 4)), np.arange(3).reshape(-1, 1))))
    assert np.allclose(preds, 3.14)

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

    monkeypatch.setattr(exp2_utils, "_run_search_cv", fake_run_search_cv)

    x = np.ones((6, 4))
    y = np.arange(6, dtype=float)
    assert exp2_utils.cv_knn(x, y, {"n_neighbors": [3]}, cv=2, n_jobs=1) == "ok"
    assert exp2_utils.cv_rsklpr(x, y, [{"size_neighborhood": [3]}], cv=2, n_jobs=1) == "ok"
    assert (
        exp2_utils.cv_subway(
            x_train=x,
            y_train=y,
            train_indices=np.arange(6),
            subway_graph=subway_graph,
            original_idx_to_station_map={index: "A" for index in range(6)},
            param_grid=[{"size_neighborhood": [3]}],
            cv=2,
            n_jobs=1,
        )
        == "ok"
    )
    assert seen[0].__name__ == "KNeighborsRegressor"
    assert seen[1].__name__ == "RsklprWrapper"
    assert seen[2].__name__ == "SubwayRsklprWrapper"


def test_plot_helpers_save_show_and_skip(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Plot helpers should save files, call ``show``, and skip missing panels."""

    show_calls: list[str] = []
    monkeypatch.setattr(exp2_utils.plt, "show", lambda: show_calls.append("show"))

    predictions = {
        "knn": np.array([100.0, 200.0, 300.0]),
        "lpr": np.array([110.0, 190.0, 290.0]),
        "rsklpr": np.array([105.0, 195.0, 295.0]),
        "gclpr_subway": np.array([101.0, 201.0, 299.0]),
        "grclpr_subway": np.array([99.0, 198.0, 301.0]),
    }
    results = {key: {"RMSE": 1.0, "R2": 0.5} for key in predictions}
    coords = pd.DataFrame(
        {
            "latitude": [40.0, 40.1, 40.2],
            "longitude": [-73.9, -73.8, -73.7],
        }
    )
    y_true = np.array([100.0, 200.0, 300.0])

    exp2_utils.plot_prediction_scatter_grid(
        predictions=predictions,
        results=results,
        y_true=y_true,
        save_plots=False,
    )
    exp2_utils.plot_geospatial_error_maps(
        predictions=predictions,
        results=results,
        coords=coords,
        y_true=y_true,
        save_plots=False,
    )
    assert show_calls == ["show", "show"]

    exp2_utils.plot_prediction_scatter_grid(
        predictions=predictions,
        results=results,
        y_true=y_true,
        save_plots=True,
        output_dir=tmp_path,
    )
    exp2_utils.plot_geospatial_error_maps(
        predictions=predictions,
        results=results,
        coords=coords,
        y_true=y_true,
        save_plots=True,
        output_dir=tmp_path,
    )
    assert (tmp_path / "nyc_airbnb_exp_2_scatter_grid.png").exists()
    assert (tmp_path / "nyc_airbnb_exp_2_geospatial_errors.png").exists()

    exp2_utils.plot_geospatial_error_maps(
        predictions={"lpr": np.array([1.0, 2.0])},
        results={"lpr": {"RMSE": 1.0, "R2": 0.1}},
        coords=pd.DataFrame({"latitude": [1.0, 2.0], "longitude": [3.0, 4.0]}),
        y_true=np.array([1.0, 2.0]),
        save_plots=True,
        output_dir=tmp_path,
    )
    captured = capsys.readouterr()
    assert "Skipping geospatial plot" in captured.out
