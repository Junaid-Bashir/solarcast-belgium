"""Offline origins, cutoff canaries, policy substitution and actual/context isolation."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solarcast.backtesting import (
    BacktestOriginError,
    BacktestSpec,
    EventTimeCutoff,
    HistorySelection,
    backtest,
    day_ahead_local_day,
)
from solarcast.models import (
    Forecast,
    ForecastContext,
    Forecaster,
    ForecastRequest,
    LastValuePersistenceForecaster,
    SeasonalPersistenceForecaster,
)


def data(start: str = "2026-06-01T00:00Z", periods: int = 10 * 96) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_time": pd.date_range(start, periods=periods, freq="15min"),
            "pv_generation_mw": pd.array(range(periods), dtype="Float64"),
            "monitored_capacity_mw": 1000.0,
        }
    )


def next_hour(issue: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(issue.floor("15min") + pd.Timedelta(hours=1), periods=2, freq="15min")


class RecordingForecaster:
    model_id = "recording_fake"

    def __init__(self) -> None:
        self.contexts: list[ForecastContext] = []

    def predict(self, context: ForecastContext) -> Forecast:
        self.contexts.append(context)
        return Forecast(
            context.request,
            pd.DataFrame({"target_time": context.request.target_times, "prediction_mw": 1.0}),
            model_id=self.model_id,
        )


@pytest.mark.parametrize(
    "issue,expected_date,count",
    [
        ("2026-01-31T23:30+01:00", "2026-02-01", 96),
        ("2026-03-28T11:00+01:00", "2026-03-29", 92),
        ("2026-10-24T11:00+02:00", "2026-10-25", 100),
    ],
)
def test_day_ahead_policy_and_offline_baseline_run(
    issue: str, expected_date: str, count: int
) -> None:
    stamp = pd.Timestamp(issue)
    targets = day_ahead_local_day(stamp)
    assert len(targets) == count
    assert targets.is_unique and targets.is_monotonic_increasing
    assert str(targets.tz) == "UTC"
    assert set(targets.tz_convert("Europe/Brussels").strftime("%Y-%m-%d")) == {expected_date}
    start = targets[0] - pd.Timedelta(days=8)
    source = data(start.isoformat(), 11 * 96)
    result = backtest(
        model=SeasonalPersistenceForecaster("7D"),
        data=source,
        spec=BacktestSpec(pd.DatetimeIndex([stamp]), availability_policy=EventTimeCutoff("30min")),
    )
    assert len(result.predictions) == count
    assert result.predictions.is_scorable.all()
    expected = source.set_index("target_time").pv_generation_mw.reindex(
        targets - pd.Timedelta("7D")
    )
    assert list(result.predictions.prediction_mw) == list(expected)


def test_leakage_canary_actuals_remain_evaluator_side_and_input_not_mutated() -> None:
    source = data("2026-06-10T10:00Z", 12)
    source.loc[source.target_time >= pd.Timestamp("2026-06-10T10:30Z"), "pv_generation_mw"] = 999999
    original = source.copy(deep=True)
    model = RecordingForecaster()
    spec = BacktestSpec(
        pd.DatetimeIndex(["2026-06-10T11:00Z"]),
        availability_policy=EventTimeCutoff("30min"),
        target_policy=next_hour,
    )
    result = backtest(model=model, data=source, spec=spec)
    seen = model.contexts[0]
    assert list(seen.history.pv_generation_mw) == [0, 1]
    assert 999999 not in set(seen.history.pv_generation_mw)
    assert list(seen.history.columns) == ["target_time", "pv_generation_mw"]
    assert seen.features is None
    assert list(result.predictions.actual_mw) == [999999, 999999]
    assert result.predictions.is_scorable.all()
    assert_frame_equal(source, original)
    # Changing every ineligible value changes actuals but cannot change model context.
    changed = source.copy(deep=True)
    changed.loc[
        changed.target_time >= pd.Timestamp("2026-06-10T10:30Z"), "pv_generation_mw"
    ] = -999999
    alternate = RecordingForecaster()
    backtest(model=alternate, data=changed, spec=spec)
    assert_frame_equal(seen.history, alternate.contexts[0].history)


@pytest.mark.parametrize(
    "issue,lag,expected_cutoff,last",
    [
        ("2026-06-10T11:00Z", "0min", "2026-06-10T11:00Z", "2026-06-10T10:45Z"),
        ("2026-06-10T11:00Z", "30min", "2026-06-10T10:30Z", "2026-06-10T10:15Z"),
        ("2026-06-10T11:07Z", "17min", "2026-06-10T10:45Z", "2026-06-10T10:30Z"),
    ],
)
def test_interval_completion_safety_lag_and_floor_boundaries(
    issue: str,
    lag: str,
    expected_cutoff: str,
    last: str,
) -> None:
    model = RecordingForecaster()
    result = backtest(
        model=model,
        data=data("2026-06-10T09:00Z", 20),
        spec=BacktestSpec(
            pd.DatetimeIndex([issue]),
            target_policy=next_hour,
            availability_policy=EventTimeCutoff(lag),
        ),
    )
    ctx = model.contexts[0]
    assert ctx.history_cutoff == pd.Timestamp(expected_cutoff)
    assert ctx.history.target_time.iloc[-1] == pd.Timestamp(last)
    assert result.origins.history_cutoff.iloc[0] == ctx.history_cutoff


def test_context_window_is_issue_anchored_and_explicitly_bounded() -> None:
    model = RecordingForecaster()
    spec = BacktestSpec(
        pd.DatetimeIndex(["2026-06-10T11:00Z"]),
        context_window="1h",
        target_policy=next_hour,
        availability_policy=EventTimeCutoff("30min"),
    )
    backtest(model=model, data=data("2026-06-10T08:00Z", 24), spec=spec)
    ctx = model.contexts[0]
    assert list(ctx.history.target_time) == list(
        pd.date_range("2026-06-10T10:00Z", periods=2, freq="15min")
    )


def test_multiple_origins_advance_context_and_actuals_without_overwriting() -> None:
    issues = pd.date_range("2026-06-09T09:00Z", periods=3, freq="1h")
    model = RecordingForecaster()
    source = data()
    spec = BacktestSpec(issues, target_policy=next_hour, availability_policy=EventTimeCutoff())
    result = backtest(model=model, data=source, spec=spec)
    assert len(result.predictions) == 6
    assert list(result.origins.issue_time) == list(issues)
    assert list(result.origins.history_count.diff().dropna()) == [4, 4]
    indexed = source.set_index("target_time").pv_generation_mw
    assert list(result.predictions.actual_mw) == list(
        indexed.reindex(result.predictions.target_time)
    )
    assert (
        result.predictions.horizon == result.predictions.target_time - result.predictions.issue_time
    ).all()
    assert len(model.contexts) == 3


def test_missing_actual_null_or_absent_is_preserved_unscorable() -> None:
    source = data("2026-06-10T08:00Z", 20)
    source.loc[source.target_time == pd.Timestamp("2026-06-10T12:00Z"), "pv_generation_mw"] = pd.NA
    source = source.loc[source.target_time != pd.Timestamp("2026-06-10T12:15Z")]
    result = backtest(
        model=LastValuePersistenceForecaster(),
        data=source,
        spec=BacktestSpec(
            pd.DatetimeIndex(["2026-06-10T11:00Z"]),
            target_policy=next_hour,
            availability_policy=EventTimeCutoff(),
        ),
    )
    assert result.predictions.actual_mw.isna().all()
    assert result.predictions.prediction_mw.notna().all()
    assert not result.predictions.is_scorable.any()
    assert list(result.origins.status) == ["ok"]


def test_model_agnostic_baseline_predictions_are_manually_correct_and_repeatable() -> None:
    source = data()
    spec = BacktestSpec(
        pd.DatetimeIndex(["2026-06-09T09:00Z", "2026-06-09T10:00Z"]),
        target_policy=next_hour,
        availability_policy=EventTimeCutoff(),
    )
    models: list[Forecaster] = [
        LastValuePersistenceForecaster(),
        SeasonalPersistenceForecaster("7D"),
    ]
    for model in models:
        result = backtest(model=model, data=source, spec=spec)
        repeat = backtest(model=model, data=source.iloc[::-1], spec=spec)
        assert_frame_equal(result.predictions, repeat.predictions)
        assert_frame_equal(result.origins, repeat.origins)
    seasonal = backtest(model=models[1], data=source, spec=spec)
    assert list(seasonal.predictions.prediction_mw) == [136, 137, 140, 141]


class FailFirst(RecordingForecaster):
    def predict(self, context: ForecastContext) -> Forecast:
        if not self.contexts:
            self.contexts.append(context)
            raise RuntimeError("synthetic failure")
        return super().predict(context)


def test_strict_failure_aborts_and_preserves_cause() -> None:
    model = FailFirst()
    with pytest.raises(
        BacktestOriginError, match="recording_fake.*issue.*synthetic failure"
    ) as info:
        backtest(
            model=model,
            data=data(),
            spec=BacktestSpec(
                pd.date_range("2026-06-09T09:00Z", periods=2, freq="1h"),
                target_policy=next_hour,
                availability_policy=EventTimeCutoff(),
            ),
        )
    assert isinstance(info.value.__cause__, RuntimeError)
    assert len(model.contexts) == 1


def test_continue_records_failed_origin_targets_then_later_success() -> None:
    model = FailFirst()
    result = backtest(
        model=model,
        data=data(),
        spec=BacktestSpec(
            pd.date_range("2026-06-09T09:00Z", periods=2, freq="1h"),
            target_policy=next_hour,
            availability_policy=EventTimeCutoff(),
            on_error="continue",
        ),
    )
    assert list(result.origins.status) == ["forecast_failed", "ok"]
    assert result.origins.error_type.iloc[0] == "RuntimeError"
    assert result.origins.error_message.iloc[0] == "synthetic failure"
    assert result.predictions.prediction_mw.iloc[:2].isna().all()
    assert result.predictions.actual_mw.notna().all()
    assert not result.predictions.is_scorable.iloc[:2].any()
    assert result.predictions.is_scorable.iloc[2:].all()


def test_all_failed_origins_keep_typed_rows_and_original_baseline_error() -> None:
    result = backtest(
        model=SeasonalPersistenceForecaster("24h"),
        data=data(),
        spec=BacktestSpec(
            pd.DatetimeIndex(["2026-06-09T09:00Z"]),
            availability_policy=EventTimeCutoff(),
            on_error="continue",
        ),
    )
    assert len(result.predictions) == 96
    assert result.predictions.prediction_mw.isna().all()
    assert result.origins.error_type.iloc[0] == "ForecastDataError"
    assert str(result.predictions.prediction_mw.dtype) == "Float64"
    assert str(result.predictions.issue_time.dtype) == "datetime64[ns, UTC]"


class SyntheticAvailability:
    name = "synthetic_published_at"

    def select(self, data: pd.DataFrame, issue_time: pd.Timestamp) -> HistorySelection:
        selected = EventTimeCutoff().select(data, issue_time)
        return HistorySelection(
            selected.history.loc[selected.history.published_at <= issue_time], selected.cutoff
        )


def test_custom_availability_can_filter_with_supplied_metadata_without_model_access() -> None:
    source = data("2026-06-10T08:00Z", 20)
    source["published_at"] = source.target_time + pd.Timedelta(hours=1)
    model = RecordingForecaster()
    backtest(
        model=model,
        data=source,
        spec=BacktestSpec(
            pd.DatetimeIndex(["2026-06-10T11:00Z"]),
            target_policy=next_hour,
            availability_policy=SyntheticAvailability(),
        ),
    )
    assert model.contexts[0].history.target_time.iloc[-1] == pd.Timestamp("2026-06-10T10:00Z")
    assert "published_at" not in model.contexts[0].history


class BadSelection:
    name = "invalid_policy"

    def select(self, data: pd.DataFrame, issue_time: pd.Timestamp) -> HistorySelection:
        return HistorySelection(data.iloc[:0], issue_time + pd.Timedelta(hours=1))


def test_custom_policy_cannot_bypass_context_temporal_validation() -> None:
    result = backtest(
        model=RecordingForecaster(),
        data=data(),
        spec=BacktestSpec(
            pd.DatetimeIndex(["2026-06-09T09:00Z"]),
            target_policy=next_hour,
            availability_policy=BadSelection(),
            on_error="continue",
        ),
    )
    assert result.origins.status.iloc[0] == "forecast_failed"
    assert result.origins.error_type.iloc[0] == "ValueError"
    assert pd.isna(result.origins.history_cutoff.iloc[0])


@pytest.mark.parametrize(
    "issues",
    [
        pd.DatetimeIndex(["2026-06-09T09:00"]),
        pd.DatetimeIndex([], tz="UTC"),
        pd.DatetimeIndex(["2026-06-09T09:00Z", "2026-06-09T09:00Z"]),
        pd.DatetimeIndex(["2026-06-09T10:00Z", "2026-06-09T09:00Z"]),
        pd.DatetimeIndex([pd.NaT], tz="UTC"),
    ],
)
def test_invalid_issue_schedules_rejected(issues: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError):
        BacktestSpec(issues, availability_policy=EventTimeCutoff())


@pytest.mark.parametrize("bad", ["-1h", "NaT", "bad"])
def test_bad_safety_lag_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        EventTimeCutoff(bad)


@pytest.mark.parametrize("bad", ["0h", "-1h", "NaT"])
def test_bad_context_window_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        BacktestSpec(
            pd.DatetimeIndex(["2026-06-09T09:00Z"]),
            availability_policy=EventTimeCutoff(),
            context_window=bad,
        )


def test_invalid_data_and_target_policy_are_not_hidden_by_continue() -> None:
    spec = BacktestSpec(
        pd.DatetimeIndex(["2026-06-09T09:00Z"]),
        availability_policy=EventTimeCutoff(),
        on_error="continue",
    )
    for invalid in (
        pd.concat([data(), data()]),
        data().assign(target_time=data().target_time.dt.tz_localize(None)),
        data().assign(pv_generation_mw=float("inf")),
    ):
        with pytest.raises(ValueError):
            backtest(model=RecordingForecaster(), data=invalid, spec=spec)
    with pytest.raises(ValueError):
        backtest(
            model=RecordingForecaster(),
            data=data(),
            spec=BacktestSpec(
                spec.issue_times,
                availability_policy=EventTimeCutoff(),
                on_error="continue",
                target_policy=lambda issue: pd.DatetimeIndex([], tz="UTC"),
            ),
        )


class WrongRequest(RecordingForecaster):
    def predict(self, context: ForecastContext) -> Forecast:
        wrong = ForecastRequest(
            context.request.issue_time - pd.Timedelta(hours=1), context.request.target_times
        )
        return Forecast(
            wrong,
            pd.DataFrame({"target_time": wrong.target_times, "prediction_mw": 1.0}),
            model_id=self.model_id,
        )


def test_returned_forecast_issue_must_match_current_origin() -> None:
    with pytest.raises(BacktestOriginError, match="does not match"):
        backtest(
            model=WrongRequest(),
            data=data(),
            spec=BacktestSpec(
                pd.DatetimeIndex(["2026-06-09T09:00Z"]), availability_policy=EventTimeCutoff()
            ),
        )
