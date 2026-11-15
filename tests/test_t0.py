"""Offline T0 boundary tests: real preprocessing, fake upstream tensors/pipeline."""

import builtins
import subprocess
import sys
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest, day_ahead_local_day
from solarcast.evaluation import compare_models
from solarcast.models import (
    ForecastContext,
    ForecastDataError,
    ForecastRequest,
    SeasonalPersistenceForecaster,
)
from solarcast.models import t0 as adapter


class Tensor:
    def __init__(self, values: npt.NDArray[np.float32]) -> None:
        self.values = values

    def detach(self) -> "Tensor":
        return self

    def cpu(self) -> "Tensor":
        return self

    def numpy(self) -> npt.NDArray[np.float32]:
        return self.values


class Output:
    def __init__(self, values: npt.NDArray[np.float32]) -> None:
        self.quantiles = Tensor(values)


class Pipeline:
    max_horizon = 1024

    def __init__(self) -> None:
        self.calls: list[tuple[npt.NDArray[np.float32], dict[str, Any]]] = []
        self.output: npt.NDArray[np.float32] | None = None
        self.failure: Exception | None = None

    def predict(self, inputs: npt.NDArray[np.float32], **kwargs: Any) -> Output:
        self.calls.append((inputs[0].copy(), kwargs))
        if self.failure:
            raise self.failure
        steps = kwargs["horizon"]
        raw = (
            self.output
            if self.output is not None
            else np.arange(steps, dtype=np.float32)[None, :, None] - 10
        )
        assert inputs.ndim == 2 and inputs.shape[0] == 1
        return Output(raw)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> tuple[Pipeline, list[int]]:
    pipeline, loads = Pipeline(), []

    def runtime(config: adapter.T0Config) -> adapter._Runtime:
        def load() -> adapter._Loaded:
            loads.append(1)
            return adapter._Loaded(cast(adapter._Model, pipeline), 0.2, 0.3)

        return adapter._Runtime("fake-0.6.1", load, lambda: None)

    monkeypatch.setattr(adapter, "_runtime", runtime)
    return pipeline, loads


def context() -> ForecastContext:
    request = ForecastRequest(
        "2025-03-01T11:00Z", pd.date_range("2025-03-01T12:00Z", periods=4, freq="15min")
    )
    times = pd.date_range("2025-03-01T08:00Z", periods=10, freq="15min")
    history = pd.DataFrame({"target_time": times, "pv_generation_mw": np.arange(10, dtype=float)})
    return ForecastContext(request, history, history_cutoff="2025-03-01T10:30Z")


def test_conversion_bridge_p50_metadata_and_reused_loading(
    fake: tuple[Pipeline, list[int]],
) -> None:
    pipeline, loads = fake
    model = adapter.T0Forecaster(adapter.T0Config(context_length=4))
    ctx = context()
    result = model.predict(ctx)
    values, settings = pipeline.calls[0]
    np.testing.assert_array_equal(values, [6, 7, 8, 9])
    assert values.dtype == np.float32
    assert settings == {
        "horizon": 10,
        "quantile_levels": [0.5],
        "context_length": 4,
        "trim_leading_missing": False,
        "symmetric_averaging": False,
        "clip_nonnegative": False,
        "snap_integers": False,
        "drop_constant_covariates": False,
    }
    assert result.predictions.target_time.tolist() == list(ctx.request.target_times)
    assert result.predictions.prediction_mw.tolist() == [-4, -3, -2, -1]
    assert result.issue_time == ctx.request.issue_time
    assert result.model_id == "t0" and result.unit == "MW"
    assert result.target_variable == "pv_generation" and result.model_version == "fake-0.6.1"
    assert model.metadata.checkpoint == "theforecastingcompany/t0-gamma"
    assert model.metadata.point_policy == "native quantile 0.5 (P50)"
    assert model.cache_seconds == 0.2 and model.load_seconds == 0.3
    model.load()
    model.predict(ctx)
    assert loads == [1] and len(pipeline.calls) == 2
    assert model.last_inference is not None and model.last_inference.rollout_steps == 10


def test_interior_null_absent_rows_and_trailing_gap_keep_cutoff_anchor(
    fake: tuple[Pipeline, list[int]],
) -> None:
    pipeline, _ = fake
    ctx = context()
    history = ctx.history.drop(index=[7, 9])
    history.loc[6, "pv_generation_mw"] = pd.NA
    model = adapter.T0Forecaster(adapter.T0Config(context_length=4))
    model.predict(ForecastContext(ctx.request, history, history_cutoff=ctx.history_cutoff))
    values, settings = pipeline.calls[0]
    np.testing.assert_allclose(values, [np.nan, np.nan, 8, np.nan], equal_nan=True)
    assert settings["horizon"] == 10  # Not shifted backward by trailing missingness.
    assert model.last_inference is not None
    assert model.last_inference.missing_slots == 3 and model.last_inference.observed_slots == 1


def test_short_history_is_not_extended_to_configured_cap(fake: tuple[Pipeline, list[int]]) -> None:
    ctx = context()
    history = ctx.history.iloc[-2:]
    model = adapter.T0Forecaster(adapter.T0Config(context_length=1344))
    model.predict(ForecastContext(ctx.request, history, history_cutoff=ctx.history_cutoff))
    values, settings = fake[0].calls[0]
    np.testing.assert_array_equal(values, [8, 9])
    assert settings["context_length"] == 2
    assert model.last_inference is not None and model.last_inference.context_slots == 2


def test_leading_supplied_missing_slot_is_preserved(fake: tuple[Pipeline, list[int]]) -> None:
    ctx = context()
    history = ctx.history.iloc[-4:]
    history.loc[history.index[0], "pv_generation_mw"] = pd.NA
    adapter.T0Forecaster().predict(
        ForecastContext(ctx.request, history, history_cutoff=ctx.history_cutoff)
    )
    np.testing.assert_allclose(fake[0].calls[0][0], [np.nan, 7, 8, 9], equal_nan=True)
    assert fake[0].calls[0][1]["trim_leading_missing"] is False


@pytest.mark.parametrize(
    "issue,count,rollout",
    [
        ("2025-03-29T10:00Z", 92, 146),
        ("2025-06-14T09:00Z", 96, 150),
        ("2025-10-25T09:00Z", 100, 154),
    ],
)
def test_local_day_dst_rollout_and_output(
    fake: tuple[Pipeline, list[int]], issue: str, count: int, rollout: int
) -> None:
    pipeline, _ = fake
    stamp = pd.Timestamp(issue)
    cutoff = stamp - pd.Timedelta("30min")
    targets = day_ahead_local_day(stamp)
    history = pd.DataFrame(
        {
            "target_time": pd.date_range(
                cutoff - pd.Timedelta("7D"), cutoff, freq="15min", inclusive="left"
            ),
            "pv_generation_mw": 0.0,
        }
    )
    model = adapter.T0Forecaster()
    result = model.predict(
        ForecastContext(ForecastRequest(stamp, targets), history, history_cutoff=cutoff)
    )
    assert len(result.predictions) == count
    assert pipeline.calls[0][1]["horizon"] == rollout
    expected = np.asarray((targets - cutoff) // pd.Timedelta("15min")) - 10
    np.testing.assert_array_equal(result.predictions.prediction_mw, expected)


def test_sparse_requested_targets_are_selected_by_timestamp(
    fake: tuple[Pipeline, list[int]],
) -> None:
    pipeline, _ = fake
    ctx = context()
    targets = ctx.request.target_times[[0, 3]]
    forecast = adapter.T0Forecaster().predict(
        ForecastContext(
            ForecastRequest(ctx.request.issue_time, targets),
            ctx.history,
            history_cutoff=ctx.history_cutoff,
        )
    )
    assert forecast.predictions.prediction_mw.tolist() == [-4, -1]
    assert pipeline.calls[0][1]["horizon"] == 10


@pytest.mark.parametrize("kind", ["empty", "all_null", "outside_window", "overflow"])
def test_unusable_context_fails_without_loading(
    fake: tuple[Pipeline, list[int]], kind: str
) -> None:
    pipeline, loads = fake
    ctx = context()
    history = ctx.history
    if kind == "empty":
        history = history.iloc[:0]
    elif kind == "all_null":
        history["pv_generation_mw"] = pd.array([None] * len(history), dtype="Float64")
    elif kind == "outside_window":
        history = history.iloc[:2]
    else:
        history["pv_generation_mw"] = 1e300
    model = adapter.T0Forecaster(adapter.T0Config(context_length=4))
    with pytest.raises(ForecastDataError):
        model.predict(ForecastContext(ctx.request, history, history_cutoff=ctx.history_cutoff))
    assert loads == [] and pipeline.calls == []


@pytest.mark.parametrize("shape", [(1, 9, 1), (10, 1), (2, 10, 1)])
def test_bad_upstream_shape_fails(fake: tuple[Pipeline, list[int]], shape: tuple[int, ...]) -> None:
    fake[0].output = np.zeros(shape, dtype=np.float32)
    with pytest.raises(RuntimeError, match="Expected quantiles shape"):
        adapter.T0Forecaster().predict(context())


def test_failures_nonfinite_output_and_stale_timing(fake: tuple[Pipeline, list[int]]) -> None:
    pipeline, _ = fake
    model = adapter.T0Forecaster()
    model.predict(context())
    pipeline.failure = ValueError("upstream exploded")
    with pytest.raises(RuntimeError, match="cutoff.*context.*rollout") as error:
        model.predict(context())
    assert isinstance(error.value.__cause__, ValueError)
    assert model.last_inference is None
    pipeline.failure = None
    pipeline.output = np.full((1, 10, 1), np.nan, dtype=np.float32)
    with pytest.raises(RuntimeError, match="finite"):
        model.predict(context())


def test_horizon_and_checkpoint_limits_are_explicit(fake: tuple[Pipeline, list[int]]) -> None:
    ctx = context()
    future = pd.DatetimeIndex([ctx.history_cutoff + pd.Timedelta("20D")])
    with pytest.raises(ForecastDataError, match="1024"):
        adapter.T0Forecaster().predict(
            ForecastContext(
                ForecastRequest(ctx.request.issue_time, future),
                ctx.history,
                history_cutoff=ctx.history_cutoff,
            )
        )
    fake[0].max_horizon = 5
    with pytest.raises(ForecastDataError, match="single-pass"):
        adapter.T0Forecaster().predict(ctx)
    fake[0].max_horizon = 0
    with pytest.raises(RuntimeError, match="prediction limit"):
        adapter.T0Forecaster().load()


@pytest.mark.parametrize(
    "settings",
    [
        {"context_length": 0},
        {"context_length": 8193},
        {"context_length": True},
        {"context_length": 2.5},
        {"device": "auto"},
        {"device": "mps"},
        {"revision": "main"},
        {"local_files_only": 1},
    ],
)
def test_invalid_configuration(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        adapter.T0Config(**settings)


def test_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("torch", "t0"):
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"\[dev,t0\]"):
        adapter.T0Forecaster()


def test_base_imports_do_not_import_heavy_dependencies() -> None:
    program = """
import sys
class BlockHeavy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 't0', 'chronos', 'transformers'}:
            raise AssertionError('Unexpected heavy import: '+fullname)
sys.meta_path.insert(0, BlockHeavy())
import solarcast
import solarcast.data
import solarcast.models.lightgbm
import solarcast.models.t0
import solarcast.models.chronos
import solarcast.backtesting
import solarcast.evaluation
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_backtest_leakage_canary_and_common_support(fake: tuple[Pipeline, list[int]]) -> None:
    pipeline, _ = fake
    times = pd.date_range("2025-02-01T00:00Z", "2025-03-04T00:00Z", freq="15min")
    data = pd.DataFrame({"target_time": times, "pv_generation_mw": 100.0})
    issues = pd.date_range("2025-03-01T10:00Z", periods=2, freq="D")
    data.loc[data.target_time >= issues[0] - pd.Timedelta("30min"), "pv_generation_mw"] = 999999.0
    spec = BacktestSpec(issues, availability_policy=EventTimeCutoff("30min"), context_window="14D")
    tsfm = backtest(model=adapter.T0Forecaster(), data=data, spec=spec)
    assert not np.any(pipeline.calls[0][0] == 999999)
    assert np.any(pipeline.calls[1][0] == 999999)  # Now-past observations may advance into context.
    baseline = backtest(model=SeasonalPersistenceForecaster("48h"), data=data, spec=spec)
    evaluated = compare_models([tsfm, baseline], reference_model=baseline.model_id)
    assert evaluated.summary.n_scored.tolist() == [192, 192]
    assert evaluated.summary.forecast_availability.tolist() == [1.0, 1.0]
    assert tsfm.predictions.actual_mw.eq(
        999999
    ).all()  # Evaluator sees future truth after inference.


def test_engineered_features_are_ignored_and_context_stays_unchanged(
    fake: tuple[Pipeline, list[int]],
) -> None:
    ctx = context()
    history = ctx.history
    features = pd.DataFrame({"target_time": ctx.request.target_times, "untrusted": 999999.0})
    enriched = ForecastContext(
        ctx.request, history, history_cutoff=ctx.history_cutoff, features=features
    )
    model = adapter.T0Forecaster()
    model.predict(enriched)
    assert not np.any(fake[0].calls[-1][0] == 999999)
    pd.testing.assert_frame_equal(enriched.history, history)


def test_upstream_failure_is_retained_by_unchanged_backtester(
    fake: tuple[Pipeline, list[int]],
) -> None:
    fake[0].failure = RuntimeError("backend failed")
    times = pd.date_range("2025-02-01T00:00Z", "2025-03-04T00:00Z", freq="15min")
    data = pd.DataFrame({"target_time": times, "pv_generation_mw": 0.0})
    spec = BacktestSpec(
        pd.date_range("2025-03-01T10:00Z", periods=2, freq="D"),
        availability_policy=EventTimeCutoff("30min"),
        on_error="continue",
    )
    result = backtest(model=adapter.T0Forecaster(), data=data, spec=spec)
    assert result.origins.status.eq("forecast_failed").all()
    assert result.predictions.prediction_mw.isna().all()
    summary = compare_models([result]).summary.iloc[0]
    assert summary.n_expected == 192 and summary.n_scored == 0
    assert summary.n_failed_origins == 2 and summary.forecast_availability == 0
