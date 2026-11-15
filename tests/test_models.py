"""Offline forecasting contracts and statically checked structural fake adapters."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solarcast.features import calendar_features
from solarcast.models import Forecast, ForecastContext, Forecaster, ForecastRequest


def request() -> ForecastRequest:
    return ForecastRequest(
        "2026-06-10T09:07Z", pd.date_range("2026-06-10T22:00Z", periods=4, freq="15min")
    )


def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_time": pd.date_range("2026-06-10T08:00Z", periods=4, freq="15min"),
            "pv_generation_mw": pd.array([1.0, None, -1.0, 3.0], dtype="Float64"),
            "monitored_capacity_mw": 1000.0,
            "source_dataset": "synthetic",
        }
    )


def predictions(req: ForecastRequest) -> pd.DataFrame:
    return pd.DataFrame({"target_time": req.target_times, "prediction_mw": [-1.0, 0.0, 2.0, 3.0]})


class StatelessFake:
    """No fit method: deliberately constant synthetic predictions, not persistence."""

    model_id = "stateless_fake"

    def predict(self, context: ForecastContext) -> Forecast:
        return Forecast(
            context.request,
            pd.DataFrame({"target_time": context.request.target_times, "prediction_mw": 1.0}),
            model_id=self.model_id,
        )


class TrainableFake:
    """Toy state gate: no learning algorithm or production training API."""

    model_id = "trainable_fake"

    def __init__(self) -> None:
        self._trained = False

    def fit(self, training_data: pd.DataFrame) -> None:
        if training_data.empty:
            raise ValueError("Synthetic training input must not be empty")
        self._trained = True

    def predict(self, context: ForecastContext) -> Forecast:
        if not self._trained:
            raise RuntimeError("trainable_fake must be fitted before predict")
        return Forecast(
            context.request,
            pd.DataFrame({"target_time": context.request.target_times, "prediction_mw": 2.0}),
            model_id=self.model_id,
        )


def run_forecast(model: Forecaster, context: ForecastContext) -> Forecast:
    """The same typed call site for stateless, trained or pretrained adapters."""
    return model.predict(context)


def test_protocols_support_stateless_and_trainable_models_without_inheritance() -> None:
    context = ForecastContext(request(), history(), history_cutoff="2026-06-10T09:00Z")
    stateless: Forecaster = StatelessFake()
    trainable = TrainableFake()
    as_forecaster: Forecaster = trainable
    assert not hasattr(stateless, "fit")
    with pytest.raises(RuntimeError, match="must be fitted"):
        run_forecast(as_forecaster, context)
    trainable.fit(context.history)
    for model in (stateless, as_forecaster):
        result = run_forecast(model, context)
        assert result.model_id == model.model_id
        assert result.request is context.request
        assert result.predictions.target_time.equals(pd.Series(context.request.target_times))


def test_request_normalizes_offsets_derives_horizons_and_preserves_issue() -> None:
    req = ForecastRequest(
        "2026-06-10T11:07+02:00", pd.date_range("2026-06-11T00:00+02:00", periods=4, freq="15min")
    )
    assert req.issue_time == pd.Timestamp("2026-06-10T09:07Z")
    assert str(req.target_times.tz) == "UTC"
    assert req.horizons[0] == pd.Timedelta(hours=12, minutes=53)
    assert req.horizons[-1] == pd.Timedelta(hours=13, minutes=38)


@pytest.mark.parametrize("issue", ["2026-06-10", "2026-06-10T09:00", "NaT", "bad timestamp"])
def test_invalid_issue_time_rejected(issue: str) -> None:
    with pytest.raises(ValueError):
        ForecastRequest(issue, request().target_times)


@pytest.mark.parametrize(
    "times",
    [
        pd.DatetimeIndex(["2026-06-10T22:00"]),
        pd.DatetimeIndex([], tz="UTC"),
        pd.DatetimeIndex(["2026-06-10T22:00Z", "2026-06-10T22:00Z"]),
        pd.DatetimeIndex(["2026-06-10T22:15Z", "2026-06-10T22:00Z"]),
        pd.DatetimeIndex(["2026-06-10T09:00Z"]),
        pd.DatetimeIndex(["2026-06-10T08:45Z"]),
        pd.DatetimeIndex(["2026-06-10T22:01Z"]),
        pd.DatetimeIndex([pd.NaT], tz="UTC"),
    ],
)
def test_invalid_request_targets_rejected(times: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError):
        ForecastRequest("2026-06-10T09:00Z", times)


@pytest.mark.parametrize(
    "start,end,count",
    [
        ("2026-02-01", "2026-02-02", 96),
        ("2026-03-29", "2026-03-30", 92),
        ("2026-10-25", "2026-10-26", 100),
    ],
)
def test_variable_length_local_days_supported(start: str, end: str, count: int) -> None:
    first = pd.Timestamp(start, tz="Europe/Brussels")
    targets = pd.date_range(
        first, pd.Timestamp(end, tz="Europe/Brussels"), freq="15min", inclusive="left"
    )
    req = ForecastRequest(first - pd.Timedelta(hours=12), targets)
    forecast = Forecast(
        req, pd.DataFrame({"target_time": targets, "prediction_mw": 0.0}), model_id="dst_fake"
    )
    assert len(req.target_times) == len(forecast.predictions) == count
    assert req.target_times.is_unique


def test_sparse_explicit_target_grid_is_allowed() -> None:
    times = request().target_times[[0, 3]]
    req = ForecastRequest("2026-06-10T09:00Z", times)
    assert len(req.horizons) == 2


def test_context_bounds_history_preserves_missing_values_and_aligns_features() -> None:
    req = request()
    raw = history().iloc[::-1]
    context = ForecastContext(
        req,
        raw,
        history_cutoff="2026-06-10T11:00+02:00",
        features=calendar_features(req.target_times),
    )
    assert context.history.target_time.is_monotonic_increasing
    assert pd.isna(context.history.pv_generation_mw.iloc[1])
    assert context.history.pv_generation_mw.iloc[2] == -1
    assert list(context.history.columns) == ["target_time", "pv_generation_mw"]
    assert "available_at" not in context.history
    assert context.features is not None


@pytest.mark.parametrize("cutoff", ["2026-06-10T09:15Z", "2026-06-10T09:01Z", "2026-06-10"])
def test_invalid_context_cutoff_rejected(cutoff: str) -> None:
    with pytest.raises(ValueError):
        ForecastContext(request(), history(), history_cutoff=cutoff)


def test_context_rejects_incomplete_or_post_cutoff_history_instead_of_dropping() -> None:
    for stamp in ("2026-06-10T09:00Z", "2026-06-10T09:15Z"):
        source = history()
        source.loc[0, "target_time"] = pd.Timestamp(stamp)
        with pytest.raises(ValueError, match="incomplete/after-cutoff"):
            ForecastContext(request(), source, history_cutoff="2026-06-10T09:00Z")


@pytest.mark.parametrize("fault", ["duplicate", "naive", "misaligned", "infinite"])
def test_context_validates_normalized_history(fault: str) -> None:
    source = history()
    if fault == "duplicate":
        source = pd.concat([source, source])
    elif fault == "naive":
        source["target_time"] = source.target_time.dt.tz_localize(None)
    elif fault == "misaligned":
        source.loc[0, "target_time"] = pd.Timestamp("2026-06-10T08:01Z")
    else:
        source.loc[0, "pv_generation_mw"] = float("inf")
    with pytest.raises(ValueError):
        ForecastContext(request(), source, history_cutoff="2026-06-10T09:00Z")


def test_empty_history_is_allowed_for_models_that_need_no_context() -> None:
    context = ForecastContext(request(), history().iloc[:0], history_cutoff="2026-06-10T09:00Z")
    assert context.history.empty
    assert context.features is None
    assert len(StatelessFake().predict(context).predictions) == 4


@pytest.mark.parametrize("fault", ["reversed", "missing", "duplicate", "label", "capacity"])
def test_context_rejects_wrong_feature_targets_and_label_columns(fault: str) -> None:
    req = request()
    features = calendar_features(req.target_times)
    if fault == "reversed":
        features = features.iloc[::-1]
    elif fault == "missing":
        features = features.iloc[:2]
    elif fault == "duplicate":
        features = pd.concat([features, features.iloc[:1]])
    elif fault == "label":
        features["pv_generation_mw"] = 100.0
    else:
        features["monitored_capacity_mw"] = 1000.0
    with pytest.raises(ValueError):
        ForecastContext(req, history(), history_cutoff="2026-06-10T09:00Z", features=features)


def test_forecast_metadata_export_and_negative_prediction_policy() -> None:
    req = request()
    result = Forecast(req, predictions(req), model_id="synthetic", model_version="test-v1")
    assert result.issue_time == req.issue_time
    assert result.model_id == "synthetic"
    assert result.model_version == "test-v1"
    assert result.predictions.prediction_mw.iloc[0] == -1
    assert list(result.predictions.columns) == ["target_time", "prediction_mw"]
    frame = result.to_frame()
    assert (frame.issue_time == req.issue_time).all()
    assert pd.TimedeltaIndex(frame.horizon).equals(req.horizons)
    assert (frame.unit == "MW").all()
    assert (frame.target_variable == "pv_generation").all()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None])
def test_missing_or_nonfinite_predictions_rejected(value: float | None) -> None:
    req = request()
    table = predictions(req)
    table["prediction_mw"] = pd.array([value, 0, 1, 2], dtype="Float64")
    with pytest.raises(ValueError, match="finite and non-missing"):
        Forecast(req, table, model_id="invalid")


@pytest.mark.parametrize("value", ["1.0", True, 1 + 2j])
def test_non_real_prediction_types_rejected(value: str | bool | complex) -> None:
    req = request()
    table = predictions(req).assign(prediction_mw=value)
    with pytest.raises(ValueError, match="real numeric"):
        Forecast(req, table, model_id="invalid")


@pytest.mark.parametrize("fault", ["duplicate", "reversed", "missing", "different", "naive"])
def test_forecast_must_match_the_request(fault: str) -> None:
    req = request()
    table = predictions(req)
    if fault == "duplicate":
        table.loc[1, "target_time"] = table.target_time.iloc[0]
    elif fault == "reversed":
        table = table.iloc[::-1]
    elif fault == "missing":
        table = table.iloc[:2]
    elif fault == "different":
        table["target_time"] += pd.Timedelta(minutes=15)
    else:
        table["target_time"] = table.target_time.dt.tz_localize(None)
    with pytest.raises(ValueError):
        Forecast(req, table, model_id="invalid")


def test_metadata_and_units_are_validated() -> None:
    req = request()
    for kwargs in (
        {"model_id": " "},
        {"model_id": "fake", "unit": "kW"},
        {"model_id": "fake", "model_version": ""},
        {"model_id": "fake", "target_variable": ""},
    ):
        with pytest.raises(ValueError):
            Forecast(req, predictions(req), **kwargs)


def test_defensive_copies_protect_validated_objects() -> None:
    times = pd.date_range("2026-06-10T22:00Z", periods=4, freq="15min")
    req = ForecastRequest("2026-06-10T09:07Z", times)
    source = history()
    features = calendar_features(req.target_times)
    raw_result = predictions(req)
    context = ForecastContext(req, source, history_cutoff="2026-06-10T09:00Z", features=features)
    result = Forecast(req, raw_result, model_id="immutable_fake")
    history_copy, prediction_copy = context.history, result.predictions
    source.loc[:, "pv_generation_mw"] = 999.0
    features.loc[:, "local_hour"] = 99
    raw_result.loc[:, "prediction_mw"] = 999.0
    returned_history = context.history
    returned_history.loc[:, "pv_generation_mw"] = 888.0
    returned_features = context.features
    assert returned_features is not None
    returned_features.loc[:, "local_hour"] = 88
    returned_predictions = result.predictions
    returned_predictions.loc[:, "prediction_mw"] = 888.0
    exported = result.to_frame()
    exported.loc[:, "prediction_mw"] = 777.0
    assert_frame_equal(context.history, history_copy)
    assert_frame_equal(result.predictions, prediction_copy)
    actual_features = context.features
    assert actual_features is not None
    assert (actual_features.local_hour < 24).all()
    # Frozen scalar metadata cannot be reassigned by ordinary application code.
    attribute = "issue_time"
    with pytest.raises(FrozenInstanceError):
        setattr(req, attribute, pd.Timestamp("2030-01-01T00:00Z"))


def test_history_feature_missing_values_remain_allowed() -> None:
    req = request()
    features = calendar_features(req.target_times).assign(pv_lag_24h_mw=np.nan)
    context = ForecastContext(req, history(), history_cutoff="2026-06-10T09:00Z", features=features)
    copied = context.features
    assert copied is not None
    assert copied.pv_lag_24h_mw.isna().all()
