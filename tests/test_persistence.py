"""Offline raw-baseline behavior, exact lag semantics and cutoff isolation."""

from datetime import timedelta

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solarcast.features import pv_lag_features
from solarcast.models import (
    ForecastContext,
    ForecastDataError,
    Forecaster,
    ForecastRequest,
    LastValuePersistenceForecaster,
    SeasonalPersistenceForecaster,
)


def history(times: pd.DatetimeIndex, values: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {"target_time": times, "pv_generation_mw": pd.array(values, dtype="Float64")}
    )


def context(values: list[float | None]) -> ForecastContext:
    request = ForecastRequest(
        "2026-06-10T09:00Z", pd.date_range("2026-06-10T10:00Z", periods=3, freq="15min")
    )
    data = history(pd.date_range("2026-06-10T08:00Z", periods=len(values), freq="15min"), values)
    return ForecastContext(request, data, history_cutoff="2026-06-10T09:00Z")


@pytest.mark.parametrize(
    "values,expected",
    [
        ([10.0, 20.0, 30.0], 30.0),
        ([10.0, 20.0, None], 20.0),
        ([10.0, 0.0, None], 0.0),
        ([10.0, -5.0, None], -5.0),
    ],
)
def test_last_value_uses_latest_nonnull_without_clipping(
    values: list[float | None],
    expected: float,
) -> None:
    ctx = context(values)
    result = LastValuePersistenceForecaster().predict(ctx)
    assert list(result.predictions.prediction_mw) == [expected] * 3
    assert result.request is ctx.request
    assert result.model_id == "last_value_persistence"
    assert result.unit == "MW"


@pytest.mark.parametrize("values", [[], [None, None]])
def test_last_value_no_valid_history_fails(values: list[float | None]) -> None:
    with pytest.raises(ForecastDataError, match="no finite non-null.*history_cutoff"):
        LastValuePersistenceForecaster().predict(context(values))


def test_last_value_uses_timestamp_order_and_retains_stale_history() -> None:
    request = ForecastRequest("2026-06-10T09:00Z", pd.DatetimeIndex(["2026-06-11T12:00Z"]))
    data = history(pd.DatetimeIndex(["2026-06-01T12:00Z", "2026-05-01T12:00Z"]), [50, 999])
    result = LastValuePersistenceForecaster().predict(
        ForecastContext(request, data, history_cutoff="2026-06-10T09:00Z")
    )
    assert result.predictions.prediction_mw.iloc[0] == 50


@pytest.mark.parametrize("lag,feature_column", [("24h", "pv_lag_24h_mw"), ("7D", "pv_lag_7d_mw")])
def test_seasonal_exact_lookup_matches_feature_semantics(lag: str, feature_column: str) -> None:
    targets = pd.date_range("2026-06-10T10:00Z", periods=3, freq="15min")
    required = targets - pd.Timedelta(lag)
    # Nearby distractors and shuffled rows prevent positional/nearest lookup from passing.
    data = history(
        pd.DatetimeIndex(
            required.append(pd.DatetimeIndex([required[0] - pd.Timedelta(minutes=15)]))
        ),
        [10, -20, 0, 999],
    ).iloc[::-1]
    req = ForecastRequest("2026-06-10T09:00Z", targets)
    ctx = ForecastContext(req, data, history_cutoff="2026-06-10T09:00Z")
    result = SeasonalPersistenceForecaster(lag).predict(ctx)
    features = pv_lag_features(ctx.history, targets, history_cutoff=ctx.history_cutoff)
    assert list(result.predictions.prediction_mw) == [10, -20, 0]
    assert list(result.predictions.prediction_mw) == list(features[feature_column])
    assert result.predictions.target_time.equals(pd.Series(req.target_times))
    assert result.issue_time == req.issue_time


@pytest.mark.parametrize("fault", ["gap", "null", "cutoff"])
def test_seasonal_missing_match_identifies_required_time_and_never_falls_back(fault: str) -> None:
    targets = pd.DatetimeIndex(["2026-06-10T10:00Z", "2026-06-10T10:15Z"])
    data = history(
        pd.DatetimeIndex(["2026-06-09T10:00Z", "2026-06-09T10:15Z", "2026-06-09T10:30Z"]),
        [10, 20, 999],
    )
    cutoff = "2026-06-10T09:00Z"
    if fault == "gap":
        data = data.drop(index=1)
    elif fault == "null":
        data.loc[1, "pv_generation_mw"] = pd.NA
    else:
        cutoff = "2026-06-09T10:15Z"
        data = data.iloc[:1]  # The required 10:15 interval is incomplete at this cutoff.
    ctx = ForecastContext(
        ForecastRequest("2026-06-10T09:00Z", targets), data, history_cutoff=cutoff
    )
    with pytest.raises(ForecastDataError, match=r"2026-06-09T10:15:00\+00:00"):
        SeasonalPersistenceForecaster().predict(ctx)


def test_day_ahead_24h_profile_can_be_unavailable_at_issue() -> None:
    req = ForecastRequest("2026-06-09T09:00Z", pd.DatetimeIndex(["2026-06-10T12:00Z"]))
    data = history(pd.DatetimeIndex(["2026-06-09T08:45Z"]), [25])
    ctx = ForecastContext(req, data, history_cutoff="2026-06-09T09:00Z")
    with pytest.raises(ForecastDataError, match="2026-06-09T12:00"):
        SeasonalPersistenceForecaster("24h").predict(ctx)


def test_future_source_values_cannot_enter_the_supplied_context() -> None:
    targets = pd.DatetimeIndex(["2026-06-10T10:00Z"])
    req = ForecastRequest("2026-06-10T09:00Z", targets)
    source = history(
        pd.DatetimeIndex(
            ["2026-06-09T10:00Z", "2026-06-10T08:45Z", "2026-06-10T09:00Z", "2026-06-10T10:00Z"]
        ),
        [10, 30, 99999, 88888],
    )
    cutoff = pd.Timestamp("2026-06-10T09:00Z")

    def bounded_context(data: pd.DataFrame) -> ForecastContext:
        allowed = data.loc[data.target_time + pd.Timedelta(minutes=15) <= cutoff]
        return ForecastContext(req, allowed, history_cutoff=cutoff)

    ctx = bounded_context(source)
    changed = source.copy(deep=True)
    changed.loc[2:, "pv_generation_mw"] = -999999
    alternate = bounded_context(changed)
    models: list[Forecaster] = [LastValuePersistenceForecaster(), SeasonalPersistenceForecaster()]
    for model in models:
        assert_frame_equal(model.predict(ctx).predictions, model.predict(alternate).predictions)
    assert LastValuePersistenceForecaster().predict(ctx).predictions.prediction_mw.iloc[0] == 30
    assert SeasonalPersistenceForecaster().predict(ctx).predictions.prediction_mw.iloc[0] == 10
    with pytest.raises(ValueError, match="incomplete/after-cutoff"):
        ForecastContext(req, source, history_cutoff=cutoff)


@pytest.mark.parametrize(
    "day,end,count",
    [
        ("2026-02-01", "2026-02-02", 96),
        ("2026-03-29", "2026-03-30", 92),
        ("2026-10-25", "2026-10-26", 100),
    ],
)
def test_all_baselines_support_dst_and_variable_local_day_lengths(
    day: str,
    end: str,
    count: int,
) -> None:
    start = pd.Timestamp(day, tz="Europe/Brussels").tz_convert("UTC")
    stop = pd.Timestamp(end, tz="Europe/Brussels").tz_convert("UTC")
    targets = pd.date_range(start, stop, freq="15min", inclusive="left")
    req = ForecastRequest(start - pd.Timedelta(hours=1), targets)
    models: list[Forecaster] = [
        LastValuePersistenceForecaster(),
        SeasonalPersistenceForecaster("48h"),
        SeasonalPersistenceForecaster("7D"),
    ]
    for model in models:
        lag = model.lag if isinstance(model, SeasonalPersistenceForecaster) else pd.Timedelta("48h")
        data = history(targets - lag, [float(i) for i in range(count)])
        ctx = ForecastContext(req, data, history_cutoff=req.issue_time)
        forecast = model.predict(ctx)
        assert len(forecast.predictions) == count
        assert forecast.predictions.target_time.is_unique
        assert not hasattr(model, "fit")


@pytest.mark.parametrize(
    "target,elapsed,clock",
    [
        ("2026-03-29T10:00Z", "2026-03-28T10:00Z", "2026-03-28T11:00Z"),
        ("2026-10-25T11:00Z", "2026-10-24T11:00Z", "2026-10-24T10:00Z"),
    ],
)
def test_24h_dst_semantics_are_elapsed_not_previous_local_clock(
    target: str,
    elapsed: str,
    clock: str,
) -> None:
    req = ForecastRequest(pd.Timestamp(target) - pd.Timedelta(hours=1), pd.DatetimeIndex([target]))
    ctx = ForecastContext(
        req, history(pd.DatetimeIndex([elapsed, clock]), [10, 20]), history_cutoff=req.issue_time
    )
    assert SeasonalPersistenceForecaster("24h").predict(ctx).predictions.prediction_mw.iloc[0] == 10


@pytest.mark.parametrize("count", [1, 5, 17])
def test_requested_target_order_and_arbitrary_count_preserved(count: int) -> None:
    targets = pd.date_range("2026-06-10T10:00Z", periods=count, freq="30min")
    data = history(targets - pd.Timedelta("24h"), [float(i) for i in range(count)])
    ctx = ForecastContext(
        ForecastRequest("2026-06-10T09:00Z", targets), data, history_cutoff="2026-06-10T09:00Z"
    )
    output = SeasonalPersistenceForecaster().predict(ctx)
    assert list(output.predictions.prediction_mw) == list(range(count))
    assert output.predictions.target_time.equals(pd.Series(ctx.request.target_times))


@pytest.mark.parametrize(
    "lag,expected",
    [
        ("24h", "seasonal_persistence_24h"),
        ("1D", "seasonal_persistence_24h"),
        ("168h", "seasonal_persistence_7d"),
        ("7D", "seasonal_persistence_7d"),
        ("48h", "seasonal_persistence_48h"),
        (timedelta(minutes=90), "seasonal_persistence_90m"),
    ],
)
def test_lag_identifiers_normalize_equivalent_durations(
    lag: str | timedelta, expected: str
) -> None:
    assert SeasonalPersistenceForecaster(lag).model_id == expected


@pytest.mark.parametrize("lag", ["0h", "-24h", "NaT", "bad", "1min", "24h1s"])
def test_invalid_lag_fails_at_construction(lag: str) -> None:
    with pytest.raises(ValueError, match="quarter-hours"):
        SeasonalPersistenceForecaster(lag)


def test_forecasters_do_not_mutate_context_or_use_supplied_features() -> None:
    req = ForecastRequest("2026-06-10T09:00Z", pd.DatetimeIndex(["2026-06-10T10:00Z"]))
    data = history(pd.DatetimeIndex(["2026-06-09T10:00Z"]), [-5])
    original = data.copy(deep=True)
    features = pd.DataFrame(
        {
            "target_time": req.target_times,
            "solar_apparent_elevation_deg": -10,
            "clearsky_ghi_w_m2": 0,
            "is_daylight": False,
        }
    )
    ctx = ForecastContext(req, data, history_cutoff=req.issue_time, features=features)
    original_context = ctx.history
    for model in (LastValuePersistenceForecaster(), SeasonalPersistenceForecaster()):
        assert model.predict(ctx).predictions.prediction_mw.iloc[0] == -5
        assert_frame_equal(ctx.history, original_context)
    assert_frame_equal(data, original)
