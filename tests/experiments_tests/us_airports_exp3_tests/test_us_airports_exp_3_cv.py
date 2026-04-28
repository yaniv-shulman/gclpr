"""Tests for the US airport network Experiment 3 runner and helpers."""

from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from gclpr.experiments.us_airports_exp3 import us_airports_exp_3_cv as exp3_cv
from gclpr.experiments.us_airports_exp3 import us_airports_network as exp3_utils


class FakeSearch:
    """Minimal search object compatible with the runner expectations."""

    def __init__(self, estimator: BaseEstimator, params: dict[str, Any]) -> None:
        """Store the fitted-estimator attributes expected by the runner."""
        self.best_estimator_ = estimator
        self.best_params_ = params
        self.best_score_ = -1.0


class OffsetEstimator(BaseEstimator, RegressorMixin):
    """Estimator that predicts the first feature plus an offset."""

    def __init__(self, offset: float = 0.0) -> None:
        """Persist the deterministic prediction offset."""
        self.offset = offset

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OffsetEstimator":
        """Record that the estimator saw a training split."""
        self.was_fit_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the first feature shifted by the configured offset."""
        if x.ndim != 2:
            raise ValueError("expected 2D features")
        return x[:, 0].astype(float) + self.offset


@pytest.fixture
def airport_graph() -> nx.Graph:
    """Small synthetic airport graph for unit tests."""
    graph = nx.Graph()
    graph.add_edge("ATL", "HOU", count=10.0, inverse_count=0.1)
    graph.add_edge("HOU", "LAX", count=5.0, inverse_count=0.2)
    nx.set_node_attributes(
        graph,
        {
            "ATL": (-84.4277, 33.6407),
            "HOU": (-95.3414, 29.9844),
            "LAX": (-118.4085, 33.9416),
        },
        "pos",
    )
    nx.set_node_attributes(graph, {"ATL": 1.0, "HOU": 1.5, "LAX": 2.0}, "delay")
    return graph


def test_load_airport_data_and_graph(tmp_path: Path) -> None:
    """The local airport loader should build a graph, features, and target."""

    airports = pd.DataFrame(
        [
            {"iata": "ATL", "name": "Atlanta", "city": "Atlanta", "state": "GA", "longitude": -84.4, "latitude": 33.6},
            {"iata": "HOU", "name": "Houston", "city": "Houston", "state": "TX", "longitude": -95.3, "latitude": 29.9},
            {
                "iata": "LAX",
                "name": "Los Angeles",
                "city": "Los Angeles",
                "state": "CA",
                "longitude": -118.4,
                "latitude": 33.9,
            },
        ]
    )
    routes = pd.DataFrame(
        [
            {"origin": "ATL", "destination": "HOU", "count": 10.0},
            {"origin": "HOU", "destination": "LAX", "count": 5.0},
        ]
    )
    airports_path = tmp_path / "airports.csv"
    routes_path = tmp_path / "routes.csv"
    airports.to_csv(airports_path, index=False)
    routes.to_csv(routes_path, index=False)

    graph, df_features, x, y = exp3_utils.load_airport_data_and_graph(
        airports_csv_path=airports_path,
        routes_csv_path=routes_path,
        random_generator=np.random.default_rng(0),
    )

    assert graph.number_of_nodes() == 3
    assert list(df_features.columns) == ["pagerank", "betweenness", "degree", "latitude", "longitude"]
    assert x.shape == (3, 5)
    assert y.shape == (3,)
    assert np.all(y > 0.0)
    assert set(nx.get_node_attributes(graph, "delay")) == {"ATL", "HOU", "LAX"}


def test_scale_non_positional_features_preserves_coordinates() -> None:
    """Only the non-coordinate columns should be standardized."""
    x_train = np.array([[1.0, 2.0, 10.0, 20.0], [2.0, 4.0, 30.0, 40.0]])
    x_test = np.array([[3.0, 6.0, 50.0, 60.0]])

    x_train_scaled, x_test_scaled = exp3_utils.scale_non_positional_features(x_train, x_test)

    assert np.allclose(x_train_scaled[:, -2:], x_train[:, -2:])
    assert np.allclose(x_test_scaled[:, -2:], x_test[:, -2:])
    assert not np.allclose(x_train_scaled[:, :-2], x_train[:, :-2])


def test_airport_network_kernel_factory_uses_graph_distance(airport_graph: nx.Graph) -> None:
    """The graph kernel should decay with unweighted hop distance."""
    mapping = {0: "ATL", 1: "HOU", 2: "LAX"}
    kernel = exp3_utils.airport_network_kernel_factory(
        graph=airport_graph,
        original_idx_to_node_map=mapping,
        train_set_original_indices=np.array([0, 1, 2]),
        predict_set_original_indices=np.array([0, 1, 2]),
        distance_scale=1.0,
    )

    similarities = kernel(
        x_0=None,
        x_neighbors=None,
        dist_x_neighbors=None,
        index_x_0=0,
        indices_neighbors=np.array([[0, 1, 2]]),
    ).ravel()

    assert similarities[0] == pytest.approx(1.0)
    assert similarities[1] == pytest.approx(float(np.exp(-1.0)))
    assert similarities[2] == pytest.approx(float(np.exp(-2.0)))


def test_graph_wrapper_and_search_helpers(
    monkeypatch: pytest.MonkeyPatch,
    airport_graph: nx.Graph,
) -> None:
    """Wrappers and helper search functions should delegate correctly."""

    class FakeRsklpr:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.fit_args: tuple[np.ndarray, np.ndarray] | None = None

        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            self.fit_args = (x.copy(), y.copy())

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], 2.5)

    monkeypatch.setattr(exp3_utils, "Rsklpr", FakeRsklpr)

    generic = exp3_utils.RsklprWrapper(size_neighborhood=7, degree=1)
    assert np.allclose(generic.fit(np.ones((4, 4)), np.arange(4, dtype=float)).predict(np.ones((3, 4))), 2.5)

    graph_wrapper = exp3_utils.GraphRsklprWrapper(
        size_neighborhood=7,
        degree=1,
        airport_network=airport_graph,
        original_idx_to_node_map={0: "ATL", 1: "HOU", 2: "LAX"},
    )
    x_train = np.hstack((np.ones((4, 4)), np.arange(4).reshape(-1, 1)))
    fitted = graph_wrapper.fit(x_train, np.arange(4, dtype=float))
    preds = fitted.predict(np.hstack((np.ones((3, 4)), np.arange(3).reshape(-1, 1))))
    assert np.allclose(preds, 2.5)

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
        del x_train, y_train, param_grid, cv, n_jobs, verbose, search_mode, n_iter, random_state, scoring
        seen.append(type(model))
        return "ok"

    monkeypatch.setattr(exp3_utils, "_run_search_cv", fake_run_search_cv)

    x = np.ones((6, 4))
    y = np.arange(6, dtype=float)
    assert exp3_utils.cv_knn(x, y, {"n_neighbors": [3]}, cv=2, n_jobs=1) == "ok"
    assert exp3_utils.cv_rsklpr(x, y, [{"size_neighborhood": [3]}], cv=2, n_jobs=1) == "ok"
    assert (
        exp3_utils.cv_graph(
            x_train=x,
            y_train=y,
            train_indices=np.arange(6),
            airport_network=airport_graph,
            original_idx_to_node_map={index: "ATL" for index in range(6)},
            param_grid=[{"size_neighborhood": [3]}],
            cv=2,
            n_jobs=1,
        )
        == "ok"
    )
    assert seen[0].__name__ == "KNeighborsRegressor"
    assert seen[1].__name__ == "RsklprWrapper"
    assert seen[2].__name__ == "GraphRsklprWrapper"


def test_metrics_and_scoring() -> None:
    """Metrics helpers should be strict about invalid predictions."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.nan, 2.0])

    with pytest.raises(ValueError):
        exp3_cv.compute_metrics(y_true, y_pred)

    assert exp3_cv.strict_rmse(y_true, y_pred) == float("inf")
    assert exp3_cv.strict_rmse(y_true, np.array([1.0, 2.0, 4.0])) == pytest.approx(np.sqrt(1.0 / 3.0))
    valid, non_finite_fraction = exp3_cv.prediction_validity(y_pred)
    assert valid is False
    assert non_finite_fraction == pytest.approx(1.0 / 3.0)

    scorer = exp3_cv.resolve_scoring("strict_rmse")
    assert callable(scorer)
    assert exp3_cv.resolve_scoring("neg_mean_squared_error") == "neg_mean_squared_error"
    with pytest.raises(ValueError):
        exp3_cv.resolve_scoring("bad_scoring")


def test_normalize_config_smoke_mode_overrides_fields() -> None:
    """Smoke mode should force the small debug configuration."""
    config = exp3_cv.normalize_config(exp3_cv.ExperimentConfig(smoke=True, max_samples=None))

    assert config.grid_profile == "smoke"
    assert config.inner_folds == 2
    assert config.generate_plots is False
    assert config.max_samples == 80
    assert config.output_prefix.endswith("_smoke")

    with pytest.raises(ValueError):
        exp3_cv.normalize_config(exp3_cv.ExperimentConfig(protocol="bad"))
    with pytest.raises(ValueError):
        exp3_cv.normalize_config(exp3_cv.ExperimentConfig(holdout_fraction=0.0))
    with pytest.raises(ValueError):
        exp3_cv.normalize_config(exp3_cv.ExperimentConfig(holdout_repeats=0))


def test_evaluate_holdout_cv_uses_search_results(
    monkeypatch: pytest.MonkeyPatch,
    airport_graph: nx.Graph,
) -> None:
    """Repeated holdout evaluation should tune on train and score each repeat."""

    def fake_run_search(
        model_key: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        train_indices: np.ndarray,
        airport_network: nx.Graph,
        original_idx_to_node_map: dict[int, str | None],
        search_grids: dict[str, object],
        config: exp3_cv.ExperimentConfig,
    ) -> FakeSearch:
        del model_key, train_indices, airport_network, original_idx_to_node_map, search_grids, config
        estimator = OffsetEstimator(offset=0.0).fit(x_train, y_train)
        return FakeSearch(estimator, {"ok": True})

    monkeypatch.setattr(exp3_cv, "run_search", fake_run_search)

    x = np.arange(120, dtype=float).reshape(30, 4)
    y = np.linspace(0.0, 1.0, 30)
    rows, predictions = exp3_cv.evaluate_holdout_cv(
        x=x,
        y=y,
        airport_network=airport_graph,
        original_idx_to_node_map={index: "ATL" for index in range(30)},
        search_grids={key: {} for key in exp3_cv._model_order},
        config=exp3_cv.ExperimentConfig(protocol="holdout_cv", holdout_repeats=2),
    )

    assert len(rows) == 2 * len(exp3_cv._model_order)
    assert set(predictions) == set(exp3_cv._model_order)
    assert sum(np.isfinite(predictions["knn"])) >= 6
    assert {row["Fold"] for row in rows} == {1, 2}


def test_save_outputs_and_valid_only_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    airport_graph: nx.Graph,
) -> None:
    """Saving outputs should create files and summarize only valid rows."""
    monkeypatch.setattr(exp3_cv, "_out_dir", tmp_path)

    plot_calls: list[str] = []

    def fake_scatter(*args: object, **kwargs: object) -> None:
        plot_calls.append("scatter")
        output_dir = cast(Path, kwargs["output_dir"])
        (output_dir / "us_airports_exp_3_scatter_grid.png").write_text("ok")

    def fake_setup(*args: object, **kwargs: object) -> None:
        plot_calls.append("setup")
        output_dir = cast(Path, kwargs["output_dir"])
        (output_dir / "us_airport_delay_network.png").write_text("ok")

    def fake_kernel(*args: object, **kwargs: object) -> None:
        plot_calls.append("kernel")
        output_dir = cast(Path, kwargs["output_dir"])
        ref = cast(str, kwargs["reference_node_id"]).lower()
        (output_dir / f"us_airport_delay_sim_{ref}.png").write_text("ok")

    monkeypatch.setattr(exp3_cv, "plot_prediction_scatter_grid", fake_scatter)
    monkeypatch.setattr(exp3_cv, "plot_airport_network_map", fake_setup)
    monkeypatch.setattr(exp3_cv, "plot_graph_kernel_similarity", fake_kernel)

    rows = [
        {
            "Fold": 1,
            "Model": "KNN",
            "Protocol": "holdout_cv",
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
            "Protocol": "holdout_cv",
            "Valid": False,
            "Non_Finite_Pct": 25.0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Selected_Params": "{}",
        },
    ]
    predictions = {"knn": np.array([1.0, np.nan])}
    config = exp3_cv.ExperimentConfig(generate_plots=True, output_prefix="unit", holdout_repeats=2)

    exp3_cv.save_outputs(
        airport_network=airport_graph,
        y=np.array([1.0, 2.0]),
        fold_records=rows,
        full_predictions=predictions,
        config=config,
    )

    assert (tmp_path / "unit_raw.csv").exists()
    assert (tmp_path / "unit_summary.csv").exists()
    assert (tmp_path / "unit_metadata.json").exists()
    assert (tmp_path / "unit_rmse_boxplot.png").exists()
    assert plot_calls == ["scatter", "setup", "kernel", "kernel"]

    summary = pd.read_csv(tmp_path / "unit_summary.csv", header=[0, 1], index_col=0)
    metadata = pd.read_json(tmp_path / "unit_metadata.json", typ="series")
    assert summary.loc["KNN", ("RMSE", "mean")] == pytest.approx(1.0)
    assert bool(metadata["reporting_uses_valid_runs_only"]) is True


def test_run_experiment_routes_to_holdout_protocol(
    monkeypatch: pytest.MonkeyPatch,
    airport_graph: nx.Graph,
) -> None:
    """The top-level runner should execute the holdout protocol."""
    x = np.arange(120, dtype=float).reshape(30, 4)
    y = np.linspace(0.0, 1.0, 30)
    coords = pd.DataFrame(
        {
            "latitude": np.linspace(30.0, 40.0, 30),
            "longitude": np.linspace(-120.0, -70.0, 30),
        }
    )

    monkeypatch.setattr(
        exp3_cv,
        "load_raw_data",
        lambda max_samples=None, seed=42, count_weighted_delay=True: (
            airport_graph,
            x,
            y,
            coords,
            {index: "ATL" for index in range(len(y))},
            ["pagerank", "betweenness", "degree", "latitude", "longitude"],
        ),
    )
    monkeypatch.setattr(exp3_cv, "build_search_grids", lambda grid_profile="full": {"grid": grid_profile})

    called: dict[str, int] = {"holdout": 0, "save": 0}

    def fake_holdout(
        x: np.ndarray,
        y: np.ndarray,
        airport_network: nx.Graph,
        original_idx_to_node_map: dict[int, str | None],
        search_grids: dict[str, object],
        config: exp3_cv.ExperimentConfig,
    ) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
        del x, y, airport_network, original_idx_to_node_map, search_grids, config
        called["holdout"] += 1
        prediction_length = 30
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
            {key: np.zeros(prediction_length) for key in exp3_cv._model_order},
        )

    def fake_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        called["save"] += 1

    monkeypatch.setattr(exp3_cv, "evaluate_holdout_cv", fake_holdout)
    monkeypatch.setattr(exp3_cv, "save_outputs", fake_save)

    df = exp3_cv.run_experiment(exp3_cv.ExperimentConfig(protocol="holdout_cv", generate_plots=False))

    assert called["holdout"] == 1
    assert called["save"] == 1
    assert not df.empty
