"""Tests for the Hungary chickenpox Experiment 4 runner and helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from grclpr.experiments.hungary_chickenpox_exp4 import (
    hungary_chickenpox as exp4_utils,
)
from grclpr.experiments.hungary_chickenpox_exp4 import (
    hungary_chickenpox_exp_4_cv as exp4_cv,
)


class FakeSearch:
    """Minimal search object compatible with the runner expectations."""

    def __init__(self, estimator: BaseEstimator, params: dict[str, Any]) -> None:
        self.best_estimator_ = estimator
        self.best_params_ = params
        self.best_score_ = -1.0


class OffsetEstimator(BaseEstimator, RegressorMixin):
    """Estimator that predicts the first feature plus an offset."""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OffsetEstimator":
        self.was_fit_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("expected 2D features")
        return x[:, 0].astype(float) + self.offset


@pytest.fixture
def chickenpox_payload() -> dict[str, Any]:
    """Small synthetic chickenpox payload for unit tests."""
    return {
        "edges": [[0, 1], [1, 2], [2, 3]],
        "FX": [
            [1, 2, 1, 0],
            [2, 3, 2, 1],
            [3, 4, 3, 2],
            [4, 5, 4, 3],
            [5, 6, 5, 4],
            [6, 7, 6, 5],
            [7, 8, 7, 6],
            [8, 9, 8, 7],
            [9, 10, 9, 8],
            [10, 11, 10, 9],
        ],
    }


@pytest.fixture
def chickenpox_json_path(tmp_path: Path, chickenpox_payload: dict[str, Any]) -> Path:
    """Persist the synthetic payload to a temporary JSON path."""
    json_path = tmp_path / "chickenpox.json"
    json_path.write_text(json.dumps(chickenpox_payload))
    return json_path


def test_download_chickenpox_json_if_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chickenpox_payload: dict[str, Any],
) -> None:
    """The public JSON should be downloaded into the requested cache path."""

    class FakeResponse(io.BytesIO):
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            self.close()

    target_path = tmp_path / "downloaded.json"
    payload_bytes = json.dumps(chickenpox_payload).encode("utf-8")

    monkeypatch.setattr(
        exp4_utils.urllib.request,
        "urlopen",
        lambda url, context=None: FakeResponse(payload_bytes),
    )

    resolved = exp4_utils.download_chickenpox_json_if_missing(target_path)

    assert resolved == target_path
    assert json.loads(target_path.read_text()) == chickenpox_payload


def test_load_chickenpox_data_and_graph(
    monkeypatch: pytest.MonkeyPatch,
    chickenpox_json_path: Path,
) -> None:
    """The local loader should build the graph, flattened features, and target."""
    monkeypatch.setattr(
        exp4_utils,
        "download_chickenpox_json_if_missing",
        lambda raw_json_path: chickenpox_json_path,
    )

    graph, x, y, metadata = exp4_utils.load_chickenpox_data_and_graph(
        raw_json_path=chickenpox_json_path,
        lags=3,
    )

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert x.shape == (28, 3)
    assert y.shape == (28,)
    assert metadata.shape[0] == 28
    assert set(metadata.columns) == {"sample_idx", "time_idx", "forecast_step", "node_idx", "node_name"}
    assert metadata["time_idx"].min() == 0
    assert metadata["time_idx"].max() == 6


def test_build_sample_idx_to_node_id_map(chickenpox_json_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sample-to-node map should align with the flattened metadata."""
    monkeypatch.setattr(
        exp4_utils,
        "download_chickenpox_json_if_missing",
        lambda raw_json_path: chickenpox_json_path,
    )
    _, _, _, metadata = exp4_utils.load_chickenpox_data_and_graph(
        raw_json_path=chickenpox_json_path,
        lags=2,
    )
    mapping = exp4_utils.build_sample_idx_to_node_id_map(metadata=metadata)
    assert mapping[0] == 0
    assert mapping[1] == 1
    assert mapping[3] == 3


def test_scale_features() -> None:
    """Lag features should be standardized on the training split."""
    x_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    x_test = np.array([[5.0, 6.0]])
    x_train_scaled, x_test_scaled = exp4_utils.scale_features(x_train, x_test)
    assert np.allclose(x_train_scaled.mean(axis=0), 0.0)
    assert np.all(np.isfinite(x_test_scaled))


def test_county_graph_kernel_factory_uses_hop_distance() -> None:
    """The graph kernel should decay with unweighted hop distance."""
    graph = nx.path_graph(3)
    mapping = {0: 0, 1: 1, 2: 2}
    kernel = exp4_utils.county_graph_kernel_factory(
        graph=graph,
        sample_idx_to_node_id=mapping,
        train_sample_indices=np.array([0, 1, 2]),
        predict_sample_indices=np.array([0, 1, 2]),
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


def test_graph_wrapper_and_search_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrappers and helper search functions should delegate correctly."""

    class FakeRsklpr:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.fit_args: tuple[np.ndarray, np.ndarray] | None = None

        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            self.fit_args = (x.copy(), y.copy())

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], 2.5)

    monkeypatch.setattr(exp4_utils, "Rsklpr", FakeRsklpr)

    generic = exp4_utils.RsklprWrapper(size_neighborhood=7, degree=1)
    assert np.allclose(generic.fit(np.ones((4, 4)), np.arange(4, dtype=float)).predict(np.ones((3, 4))), 2.5)

    graph = nx.path_graph(3)
    graph_wrapper = exp4_utils.GraphRsklprWrapper(
        size_neighborhood=7,
        degree=1,
        county_graph=graph,
        sample_idx_to_node_id={0: 0, 1: 1, 2: 2},
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
        cv: object,
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

    monkeypatch.setattr(exp4_utils, "_run_search_cv", fake_run_search_cv)

    x = np.ones((6, 4))
    y = np.arange(6, dtype=float)
    time_splits = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
    assert exp4_utils.cv_knn(x, y, {"n_neighbors": [3]}, cv=time_splits, n_jobs=1) == "ok"
    assert exp4_utils.cv_rsklpr(x, y, [{"size_neighborhood": [3]}], cv=time_splits, n_jobs=1) == "ok"
    assert (
        exp4_utils.cv_graph(
            x_train=x,
            y_train=y,
            train_indices=np.arange(6),
            county_graph=graph,
            sample_idx_to_node_id={index: index % 3 for index in range(6)},
            param_grid=[{"size_neighborhood": [3]}],
            cv=time_splits,
            n_jobs=1,
        )
        == "ok"
    )
    assert seen[0].__name__ == "KNeighborsRegressor"
    assert seen[1].__name__ == "RsklprWrapper"
    assert seen[2].__name__ == "GraphRsklprWrapper"


def test_plot_helpers_save_outputs(tmp_path: Path) -> None:
    """Plot helpers should save figures without raising."""
    predictions = {
        "knn": np.array([1.0, 1.2, 1.4]),
        "lpr": np.array([1.0, 1.1, 1.3]),
        "rsklpr": np.array([0.9, 1.0, 1.2]),
        "gclpr_graph": np.array([1.0, 1.0, 1.1]),
        "grclpr_graph": np.array([0.95, 1.05, 1.15]),
    }
    results = {key: {"RMSE": 0.1, "MAE": 0.1, "R2": 0.9} for key in predictions}
    y_true = np.array([1.0, 1.0, 1.0])
    graph = nx.path_graph(3)
    metadata = pd.DataFrame(
        {
            "sample_idx": [0, 1, 2],
            "time_idx": [0, 0, 0],
            "forecast_step": [4, 4, 4],
            "node_idx": [0, 1, 2],
            "node_name": ["County 0", "County 1", "County 2"],
        }
    )
    nx.set_node_attributes(graph, {0: "County 0", 1: "County 1", 2: "County 2"}, "name")

    exp4_utils.plot_prediction_scatter_grid(
        predictions=predictions,
        results=results,
        y_true=y_true,
        save_plots=True,
        output_dir=tmp_path,
    )
    exp4_utils.plot_graph_error_maps(
        graph=graph,
        metadata=metadata,
        y_true=y_true,
        predictions=predictions,
        save_plot=True,
        output_dir=tmp_path,
    )

    assert (tmp_path / "hungary_chickenpox_exp_4_scatter_grid.png").exists()
    assert (tmp_path / "hungary_chickenpox_exp_4_graph_errors.png").exists()


def test_time_series_sample_splits() -> None:
    """Rolling-origin splits should respect time ordering at the sample level."""
    sample_time_indices = np.repeat(np.arange(8), 2)
    splits = exp4_cv.build_time_series_sample_splits(sample_time_indices=sample_time_indices, n_splits=3)

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert sample_time_indices[train_idx].max() < sample_time_indices[test_idx].min()
        assert np.intersect1d(train_idx, test_idx).size == 0


def test_metrics_and_scoring() -> None:
    """Metric helpers should be strict about invalid predictions."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.nan, 2.0])

    with pytest.raises(ValueError):
        exp4_cv.compute_metrics(y_true, y_pred)

    assert exp4_cv.strict_rmse(y_true, y_pred) == float("inf")
    assert callable(exp4_cv.resolve_scoring("strict_rmse"))
    assert exp4_cv.resolve_scoring("r2") == "r2"

    with pytest.raises(ValueError):
        exp4_cv.resolve_scoring("bad-score")


def test_build_search_grids_and_normalize_config() -> None:
    """Grid selection and smoke normalization should be deterministic."""
    smoke_grid = exp4_cv.build_search_grids("smoke")
    assert smoke_grid["knn"]["n_neighbors"] == [3]

    reduced_grid = exp4_cv.build_search_grids("reduced")
    assert reduced_grid["gclpr_graph"][0]["distance_scale"] == [0.5, 1.0, 1.5, 2.0]

    normalized = exp4_cv.normalize_config(exp4_cv.ExperimentConfig(smoke=True))
    assert normalized.grid_profile == "smoke"
    assert normalized.outer_splits == 3
    assert normalized.inner_folds == 2

    with pytest.raises(ValueError):
        exp4_cv.normalize_config(exp4_cv.ExperimentConfig(protocol="bad"))


def test_run_search_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search dispatch should route graph and non-graph models correctly."""
    calls: list[str] = []

    def fake_knn(**kwargs: object) -> str:
        calls.append("knn")
        return "knn"

    def fake_graph(**kwargs: object) -> str:
        calls.append("graph")
        return "graph"

    monkeypatch.setitem(exp4_cv._search_dispatch, "knn", fake_knn)
    monkeypatch.setitem(exp4_cv._search_dispatch, "gclpr_graph", fake_graph)

    x = np.ones((12, 4))
    y = np.arange(12, dtype=float)
    result_knn = exp4_cv.run_search(
        model_key="knn",
        x_train=x,
        y_train=y,
        train_indices=np.arange(12),
        train_time_indices=np.repeat(np.arange(6), 2),
        county_graph=nx.path_graph(3),
        sample_idx_to_node_id={index: index % 3 for index in range(12)},
        search_grids={"knn": {"n_neighbors": [3]}, "gclpr_graph": [{"size_neighborhood": [3]}]},
        config=exp4_cv.ExperimentConfig(inner_folds=2),
    )
    result_graph = exp4_cv.run_search(
        model_key="gclpr_graph",
        x_train=x,
        y_train=y,
        train_indices=np.arange(12),
        train_time_indices=np.repeat(np.arange(6), 2),
        county_graph=nx.path_graph(3),
        sample_idx_to_node_id={index: index % 3 for index in range(12)},
        search_grids={"knn": {"n_neighbors": [3]}, "gclpr_graph": [{"size_neighborhood": [3]}]},
        config=exp4_cv.ExperimentConfig(inner_folds=2),
    )

    assert result_knn == "knn"
    assert result_graph == "graph"
    assert calls == ["knn", "graph"]


def test_evaluate_rolling_origin_cv(monkeypatch: pytest.MonkeyPatch) -> None:
    """The temporal evaluator should produce one record per fold and model."""
    x = np.column_stack(
        [
            np.linspace(0.0, 1.0, 40),
            np.linspace(1.0, 2.0, 40),
            np.linspace(2.0, 3.0, 40),
            np.linspace(3.0, 4.0, 40),
        ]
    )
    y = x[:, 0] + 1.0
    metadata = pd.DataFrame(
        {
            "sample_idx": np.arange(40),
            "time_idx": np.repeat(np.arange(10), 4),
            "forecast_step": np.repeat(np.arange(4, 14), 4),
            "node_idx": np.tile(np.arange(4), 10),
            "node_name": [f"County {index % 4}" for index in range(40)],
        }
    )

    def fake_run_search(
        model_key: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        train_indices: np.ndarray,
        train_time_indices: np.ndarray,
        county_graph: nx.Graph,
        sample_idx_to_node_id: dict[int, int],
        search_grids: dict[str, Any],
        config: exp4_cv.ExperimentConfig,
    ) -> FakeSearch:
        del (
            x_train,
            y_train,
            train_indices,
            train_time_indices,
            county_graph,
            sample_idx_to_node_id,
            search_grids,
            config,
        )
        estimator = OffsetEstimator(offset=1.0).fit(np.ones((2, 2)), np.ones(2))
        return FakeSearch(estimator=estimator, params={"model": model_key})

    monkeypatch.setattr(exp4_cv, "run_search", fake_run_search)

    records, predictions = exp4_cv.evaluate_rolling_origin_cv(
        x=x,
        y=y,
        metadata=metadata,
        county_graph=nx.path_graph(4),
        sample_idx_to_node_id={index: index % 4 for index in range(40)},
        search_grids={key: {} for key in exp4_cv._model_order},
        config=exp4_cv.ExperimentConfig(outer_splits=3, inner_folds=2),
    )

    assert len(records) == 3 * len(exp4_cv._model_order)
    assert set(predictions) == set(exp4_cv._model_order)
    assert all(np.isfinite(record["RMSE"]) for record in records)


def test_save_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Saving outputs should persist raw results, summary, metadata, and plots."""
    monkeypatch.setattr(exp4_cv, "_out_dir", tmp_path)

    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, {0: "County 0", 1: "County 1", 2: "County 2"}, "name")
    metadata = pd.DataFrame(
        {
            "sample_idx": [0, 1, 2, 3, 4, 5],
            "time_idx": [0, 0, 0, 1, 1, 1],
            "forecast_step": [4, 4, 4, 5, 5, 5],
            "node_idx": [0, 1, 2, 0, 1, 2],
            "node_name": ["County 0", "County 1", "County 2", "County 0", "County 1", "County 2"],
        }
    )
    y = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    fold_records = [
        {
            "Fold": 1,
            "Model": "KNN",
            "Protocol": "rolling_origin_cv",
            "Valid": True,
            "Non_Finite_Pct": 0.0,
            "RMSE": 0.1,
            "MAE": 0.1,
            "R2": 0.9,
            "Train_Time_Max": 0,
            "Test_Time_Min": 1,
            "Test_Time_Max": 1,
            "Selected_Params": "{}",
        }
    ]
    full_predictions = {
        "knn": np.array([1.0, 2.0, 3.0, 1.5, 2.4, 3.4]),
        "lpr": np.array([1.0, 2.1, 2.9, 1.4, 2.6, 3.6]),
        "rsklpr": np.array([1.0, 2.2, 3.1, 1.6, 2.5, 3.3]),
        "gclpr_graph": np.array([1.0, 2.0, 3.1, 1.5, 2.5, 3.4]),
        "grclpr_graph": np.array([1.0, 1.9, 3.0, 1.4, 2.4, 3.6]),
    }

    exp4_cv.save_outputs(
        county_graph=graph,
        metadata=metadata,
        y=y,
        fold_records=fold_records,
        full_predictions=full_predictions,
        config=exp4_cv.ExperimentConfig(output_prefix="exp4_test"),
    )

    assert (tmp_path / "exp4_test_raw.csv").exists()
    assert (tmp_path / "exp4_test_summary.csv").exists()
    assert (tmp_path / "exp4_test_metadata.json").exists()
    assert (tmp_path / "hungary_chickenpox_exp_4_scatter_grid.png").exists()
    assert (tmp_path / "hungary_chickenpox_exp_4_graph_errors.png").exists()
    assert (tmp_path / "exp4_test_rmse_boxplot.png").exists()


def test_run_experiment_routes_to_selected_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """The top-level runner should execute the configured temporal protocol."""
    x = np.ones((20, 4))
    y = np.arange(20, dtype=float)
    metadata = pd.DataFrame(
        {
            "sample_idx": np.arange(20),
            "time_idx": np.repeat(np.arange(5), 4),
            "forecast_step": np.repeat(np.arange(4, 9), 4),
            "node_idx": np.tile(np.arange(4), 5),
            "node_name": [f"County {index % 4}" for index in range(20)],
        }
    )

    monkeypatch.setattr(
        exp4_cv,
        "load_raw_data",
        lambda lags=4, max_snapshots=None: (
            nx.path_graph(4),
            x,
            y,
            metadata,
            {idx: idx % 4 for idx in range(20)},
            ["lag_4"],
        ),
    )
    monkeypatch.setattr(
        exp4_cv,
        "build_search_grids",
        lambda grid_profile="full": {"grid": grid_profile},
    )

    called = {"eval": 0, "save": 0}

    def fake_evaluate(
        x: np.ndarray,
        y: np.ndarray,
        metadata: pd.DataFrame,
        county_graph: nx.Graph,
        sample_idx_to_node_id: dict[int, int],
        search_grids: dict[str, Any],
        config: exp4_cv.ExperimentConfig,
    ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        del x, y, metadata, county_graph, sample_idx_to_node_id, search_grids, config
        called["eval"] += 1
        return (
            [
                {
                    "Fold": 1,
                    "Model": "KNN",
                    "Protocol": "rolling_origin_cv",
                    "Valid": True,
                    "Non_Finite_Pct": 0.0,
                    "RMSE": 1.0,
                    "MAE": 1.0,
                    "R2": 0.0,
                    "Train_Time_Max": 0,
                    "Test_Time_Min": 1,
                    "Test_Time_Max": 1,
                    "Selected_Params": "{}",
                }
            ],
            {key: np.zeros(20) for key in exp4_cv._model_order},
        )

    def fake_save(*args: object, **kwargs: object) -> None:
        called["save"] += 1

    monkeypatch.setattr(exp4_cv, "evaluate_rolling_origin_cv", fake_evaluate)
    monkeypatch.setattr(exp4_cv, "save_outputs", fake_save)

    df = exp4_cv.run_experiment(exp4_cv.ExperimentConfig(generate_plots=False))
    assert called == {"eval": 1, "save": 1}
    assert list(df["Model"]) == ["KNN"]
