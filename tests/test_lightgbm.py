"""Native optional LightGBM tests; absent extra skips only this module."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from solarcast.backtesting import (
    BacktestSpec,
    EventTimeCutoff,
    _build_context,
    _prepare_data,
    backtest,
)
from solarcast.evaluation import compare_models
from solarcast.models import Forecaster, ForecastRequest, SeasonalPersistenceForecaster
from solarcast.models.lightgbm import LightGBMConfig, LightGBMForecaster, LightGBMNotFittedError
from solarcast.models.lightgbm_data import FEATURE_NAMES, TrainingDataset, build_training_dataset


def next_hour(issue: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(issue + pd.Timedelta(hours=1), periods=4, freq="15min")


def source() -> pd.DataFrame:
    times = pd.date_range("2026-01-01T00:00Z", periods=12 * 96, freq="15min")
    return pd.DataFrame(
        {"target_time": times, "pv_generation_mw": np.arange(len(times), dtype=float)}
    )


def training(data: pd.DataFrame | None = None) -> TrainingDataset:
    spec = BacktestSpec(
        pd.date_range("2026-01-03T09:00Z", periods=3, freq="D"),
        availability_policy=EventTimeCutoff("30min"),
        target_policy=next_hour,
    )
    return build_training_dataset(
        source() if data is None else data, spec, fit_time="2026-01-07T00:00Z"
    )


def model() -> LightGBMForecaster:
    return LightGBMForecaster(LightGBMConfig(n_estimators=5, min_child_samples=1, num_leaves=3))


def test_predict_before_fit_fails_clearly() -> None:
    req = ForecastRequest("2026-01-09T09:00Z", next_hour(pd.Timestamp("2026-01-09T09:00Z")))
    ctx = _build_context(
        _prepare_data(source()),
        req,
        BacktestSpec(
            pd.DatetimeIndex([req.issue_time]), availability_policy=EventTimeCutoff("30min")
        ),
    )
    with pytest.raises(LightGBMNotFittedError, match="must be fitted"):
        model().predict(ctx)


def test_fit_predict_metadata_determinism_and_model_agnostic_benchmark() -> None:
    dataset = training()
    first, second = model(), model()
    first.fit(dataset)
    second.fit(dataset)
    assert first.training_info.feature_names == FEATURE_NAMES
    assert first.training_info.n_rows == 12
    assert first.training_info.last_training_target < first.training_info.fit_time
    protocol: Forecaster = first
    spec = BacktestSpec(
        pd.date_range("2026-01-09T09:00Z", periods=2, freq="D"),
        availability_policy=EventTimeCutoff("30min"),
        target_policy=next_hour,
    )
    results = [
        backtest(model=protocol, data=source(), spec=spec),
        backtest(model=SeasonalPersistenceForecaster("7D"), data=source(), spec=spec),
    ]
    other = backtest(model=second, data=source(), spec=spec)
    assert np.allclose(
        results[0].predictions.prediction_mw.to_numpy(dtype=float),
        other.predictions.prediction_mw.to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )
    benchmark = compare_models(results, reference_model="seasonal_persistence_7d")
    assert benchmark.summary.n_scored.eq(8).all()
    assert set(benchmark.leaderboard().model_id) == {"lightgbm", "seasonal_persistence_7d"}


def test_predict_before_training_fit_time_is_rejected() -> None:
    fitted = model()
    fitted.fit(training())
    spec = BacktestSpec(
        pd.DatetimeIndex(["2026-01-06T09:00Z"]),
        availability_policy=EventTimeCutoff("30min"),
        target_policy=next_hour,
    )
    request = ForecastRequest(spec.issue_times[0], next_hour(spec.issue_times[0]))
    context = _build_context(_prepare_data(source()), request, spec)
    with pytest.raises(ValueError, match="fit_time"):
        fitted.predict(context)


def test_prediction_feature_schema_mismatch_fails_without_silent_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted = model()
    fitted.fit(training())
    issue = pd.Timestamp("2026-01-09T09:00Z")
    spec = BacktestSpec(pd.DatetimeIndex([issue]), availability_policy=EventTimeCutoff("30min"))
    context = _build_context(
        _prepare_data(source()), ForecastRequest(issue, next_hour(issue)), spec
    )
    monkeypatch.setattr(
        "solarcast.models.lightgbm.lightgbm_features",
        lambda context, **kwargs: pd.DataFrame({"wrong_feature": [1.0]}),
    )
    with pytest.raises(ValueError, match="feature names/order"):
        fitted.predict(context)


@pytest.mark.parametrize(
    "issue,count",
    [("2026-01-31T10:00Z", 96), ("2026-03-28T10:00Z", 92), ("2026-10-24T09:00Z", 100)],
)
def test_lightgbm_accepts_all_dst_lengths_with_missing_history(issue: str, count: int) -> None:
    fitted = model()
    fitted.fit(training())
    spec = BacktestSpec(pd.DatetimeIndex([issue]), availability_policy=EventTimeCutoff("30min"))
    # No reachable training-source values at prediction: supplied history is empty.
    result = backtest(model=fitted, data=source().iloc[:0], spec=spec)
    assert len(result.predictions) == count
    assert result.predictions.prediction_mw.notna().all()
    assert result.predictions.target_time.is_unique


def test_prediction_canary_cannot_reach_into_full_source_and_negatives_are_raw() -> None:
    fitted = model()
    fitted.fit(training(source().assign(pv_generation_mw=-5.0)))
    spec = BacktestSpec(
        pd.DatetimeIndex(["2026-01-09T09:00Z"]),
        availability_policy=EventTimeCutoff("30min"),
        target_policy=next_hour,
    )
    original = source()
    changed = original.copy(deep=True)
    changed.loc[changed.target_time >= pd.Timestamp("2026-01-09T08:30Z"), "pv_generation_mw"] = (
        999999
    )
    a = backtest(model=fitted, data=original, spec=spec)
    b = backtest(model=fitted, data=changed, spec=spec)
    assert np.allclose(
        a.predictions.prediction_mw.to_numpy(dtype=float),
        b.predictions.prediction_mw.to_numpy(dtype=float),
    )
    assert a.predictions.prediction_mw.lt(0).all()


@pytest.mark.parametrize("fault", ["missing", "reordered", "extra"])
def test_feature_schema_is_strict_and_failed_refit_clears_state(fault: str) -> None:
    dataset = training()
    fitted = model()
    fitted.fit(dataset)
    if fault == "missing":
        dataset = replace(dataset, X=dataset.X.drop(columns=FEATURE_NAMES[0]))
    elif fault == "reordered":
        # Swap names without silently accepting a different feature order.
        changed = dataset.X.rename(
            columns={FEATURE_NAMES[0]: "temporary", FEATURE_NAMES[1]: FEATURE_NAMES[0]},
        )
        dataset = replace(dataset, X=changed.rename(columns={"temporary": FEATURE_NAMES[1]}))
    else:
        dataset.X["unexpected"] = 1.0
    with pytest.raises(ValueError, match="feature names/order"):
        fitted.fit(dataset)
    with pytest.raises(LightGBMNotFittedError):
        _ = fitted.training_info


@pytest.mark.parametrize(
    "config",
    [
        {"n_estimators": 0},
        {"num_leaves": 1},
        {"min_child_samples": 0},
        {"learning_rate": float("nan")},
    ],
)
def test_bad_configuration_rejected(config: dict[str, int | float]) -> None:
    # Construct each named parameter with explicit typed values below.
    if "n_estimators" in config:
        with pytest.raises(ValueError):
            LightGBMConfig(n_estimators=0)
    elif "num_leaves" in config:
        with pytest.raises(ValueError):
            LightGBMConfig(num_leaves=1)
    elif "min_child_samples" in config:
        with pytest.raises(ValueError):
            LightGBMConfig(min_child_samples=0)
    else:
        with pytest.raises(ValueError):
            LightGBMConfig(learning_rate=float("nan"))
