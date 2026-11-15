"""Vintage-correct NWP training and full-prediction canaries; no provider network."""

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solarcast.backtesting import (
    BacktestSpec,
    EventTimeCutoff,
    _build_context,
    _prepare_data,
    backtest,
)
from solarcast.data.weather import FEATURES, ForecastRun, SpatialWeights, WeatherError, WeatherStore
from solarcast.data.weather.types import PRODUCT, PROVIDER
from solarcast.models import ForecastRequest
from solarcast.models.lightgbm import LightGBMConfig, LightGBMForecaster
from solarcast.models.lightgbm_data import (
    FEATURE_NAMES,
    build_training_dataset,
    lightgbm_features,
    validate_feature_matrix,
)


def pv(start: str = "2025-06-01T00Z", periods: int = 16 * 96) -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq="15min")
    return pd.DataFrame(
        {
            "target_time": times,
            "pv_generation_mw": np.maximum(0, np.sin(np.arange(periods) * np.pi / 48)) * 1000,
        }
    )


def store(days: list[str], *, sentinel: bool = True) -> WeatherStore:
    cells = ((50.5, 3.0), (50.5, 4.5), (50.5, 5.5))
    weights = SpatialWeights(
        pd.DataFrame(
            {"latitude": [50.5] * 3, "longitude": [3.0, 4.5, 5.5], "area_m2": [1.0, 2.0, 1.0]}
        ),
        expected_cells=cells,
        boundary_id="synthetic",
    )
    runs = []
    for day in days:
        for hour in [0, 6] if sentinel else [0]:
            reference = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=hour)
            rows = []
            for step in range(18, 49):
                for latitude, longitude in cells:
                    rows.append(
                        {
                            "reference_time": reference,
                            "valid_time": reference + pd.Timedelta(hours=step),
                            "forecast_lead": pd.Timedelta(hours=step),
                            "latitude": latitude,
                            "longitude": longitude,
                            "dswrf_start_hour": ((step - 1) // 6) * 6,
                            "dswrf_end_hour": step,
                            "dswrf_average_w_m2": 999999.0 if hour == 6 else 100.0,
                            "cloud_percent": 40.0,
                            "temperature_2m_k": 293.15,
                        }
                    )
            runs.append(
                ForecastRun(
                    reference,
                    pd.DataFrame(rows),
                    version=f"{day}_{hour}",
                    metadata={"provider": PROVIDER, "product": PRODUCT},
                )
            )
    return WeatherStore(tuple(runs), weights)


def spec() -> BacktestSpec:
    return BacktestSpec(
        pd.date_range("2025-06-10T09Z", periods=2, freq="D"),
        availability_policy=EventTimeCutoff("30min"),
        context_window="14D",
    )


def test_public_weather_orchestration_preserves_vintages_and_missing_origin_accounting() -> None:
    """The public layer uses the same local vintage selection at fit and inference."""
    pytest.importorskip("lightgbm")
    from solarcast import BenchmarkConfig, benchmark

    config = BenchmarkConfig(
        issue_times=pd.date_range("2025-06-14T09Z", periods=2, freq="D"),
        observation_safety_lag="30min",
        training_issue_times=spec().issue_times,
        fit_time="2025-06-13T08Z",
        lightgbm=LightGBMConfig(n_estimators=5, min_child_samples=1),
    )
    local = store(["2025-06-10", "2025-06-11", "2025-06-14", "2025-06-15"])
    result = benchmark(
        data=pv(), config=config, models=["lightgbm_history_radiation_cloud"], weather=local
    )
    assert result.leaderboard().n_scored.iloc[0] == 192
    assert result.origins.weather_reference_time.dt.hour.eq(0).all()
    assert result.origins.weather_availability_basis.eq("assumed").all()
    info = result.metadata["model_metadata"][  # type: ignore[index]
        "lightgbm_history_radiation_cloud"
    ]["training"]
    assert info["feature_set"] == "history+radiation+cloud"
    assert len(info["feature_names"]) == 19
    missing = benchmark(
        data=pv(),
        config=config,
        models=["lightgbm_history_radiation_cloud"],
        weather=store(["2025-06-10", "2025-06-11"]),
    )
    assert missing.leaderboard().n_failed_origins.iloc[0] == 2
    assert missing.predictions.prediction_mw.isna().all()
    assert len(missing.predictions) == 192


def test_mandatory_weather_training_leakage_canary_and_feature_identity() -> None:
    weather = store(["2025-06-10", "2025-06-11"])
    history = build_training_dataset(pv(), spec(), fit_time="2025-06-13T08Z")
    informed = build_training_dataset(
        pv(), spec(), fit_time="2025-06-13T08Z", feature_set="history+nwp", weather_provider=weather
    )
    assert_frame_equal(history.X, informed.X[list(FEATURE_NAMES)])
    assert informed.y.equals(history.y)
    assert informed.X[list(FEATURES[:3])].eq(100.0).all().all()
    assert informed.X[FEATURES[4]].eq(0.4).all()
    assert informed.X[FEATURES[-1]].to_numpy() == pytest.approx(np.full(len(informed.X), 20.0))
    assert not informed.X.eq(999999).any().any()
    assert (informed.rows.weather_available_at <= informed.rows.issue_time).all()
    assert informed.rows.weather_reference_time.dt.hour.eq(0).all()
    assert len(informed.rows) == 192 and informed.origin_failures.empty


def fitted() -> LightGBMForecaster:
    pytest.importorskip("lightgbm")
    model = LightGBMForecaster(
        LightGBMConfig(n_estimators=5, min_child_samples=1), feature_set="history+nwp"
    )
    dataset = build_training_dataset(
        pv(),
        spec(),
        fit_time="2025-06-13T08Z",
        feature_set="history+nwp",
        weather_provider=store(["2025-06-10", "2025-06-11"]),
    )
    model.fit(dataset)
    return model


def test_mandatory_weather_prediction_leakage_canary_full_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = fitted()
    captured = []
    booster = model._booster
    assert booster is not None

    class RecordingBooster:
        def feature_name(self) -> list[str]:
            assert booster is not None
            return [str(name) for name in booster.feature_name()]

        def predict(self, X: pd.DataFrame, **kwargs: Any) -> Any:
            assert booster is not None
            captured.append(X.copy())
            return booster.predict(X, **kwargs)

    monkeypatch.setattr(model, "_booster", RecordingBooster())
    schedule = BacktestSpec(
        pd.DatetimeIndex(["2025-06-13T09Z"]),
        availability_policy=EventTimeCutoff("30min"),
        context_window="14D",
    )
    result = backtest(model=model, data=pv(), spec=schedule, weather_provider=store(["2025-06-13"]))
    assert result.origins.status.tolist() == ["ok"]
    assert len(captured) == 1 and captured[0].shape == (96, 20)
    assert not captured[0].eq(999999).any().any()
    assert captured[0][FEATURES[0]].eq(100).all()


@pytest.mark.parametrize("missing", ["run", "time", "variable"])
def test_missing_training_weather_explicit_origin_accounting(missing: str) -> None:
    weather = store(["2025-06-10", "2025-06-11"], sentinel=False)
    if missing == "run":
        weather = replace(weather, runs=weather.runs[1:])
    else:
        run = weather.runs[0]
        frame = run.data
        if missing == "time":
            frame = frame.loc[frame.dswrf_end_hour != 35]
        else:
            frame.loc[frame.dswrf_end_hour == 35, "temperature_2m_k"] = np.nan
        altered = ForecastRun(run.reference_time, frame, version="missing", metadata=run.metadata)
        weather = replace(weather, runs=(altered, weather.runs[1]))
    data = build_training_dataset(
        pv(),
        replace_spec_continue(),
        fit_time="2025-06-13T08Z",
        feature_set="history+nwp",
        weather_provider=weather,
    )
    assert len(data.rows) == 96 and data.n_requested == 192
    assert len(data.origin_failures) == 1
    with pytest.raises(WeatherError):
        build_training_dataset(
            pv(),
            spec(),
            fit_time="2025-06-13T08Z",
            feature_set="history+nwp",
            weather_provider=weather,
        )


def replace_spec_continue() -> BacktestSpec:
    original = spec()
    return BacktestSpec(
        original.issue_times,
        availability_policy=original.availability_policy,
        context_window="14D",
        on_error="continue",
    )


@pytest.mark.parametrize("day,count", [("2026-03-28", 92), ("2026-06-10", 96), ("2026-10-24", 100)])
def test_weather_model_dst_predictions(day: str, count: int) -> None:
    model = fitted()
    issue = pd.Timestamp(day + "T11:00", tz="Europe/Brussels").tz_convert("UTC")
    schedule = BacktestSpec(pd.DatetimeIndex([issue]), availability_policy=EventTimeCutoff("30min"))
    history = pv((issue.floor("D") - pd.Timedelta(days=10)).isoformat())
    result = backtest(model=model, data=history, spec=schedule, weather_provider=store([day]))
    assert len(result.predictions) == count
    assert result.predictions.prediction_mw.notna().all()


def test_history_only_regression_ignores_weather_and_matches_original_recipe() -> None:
    pytest.importorskip("lightgbm")
    dataset = build_training_dataset(pv(), spec(), fit_time="2025-06-13T08Z")
    original, explicit = LightGBMForecaster(), LightGBMForecaster(feature_set="history")
    original.fit(dataset)
    explicit.fit(dataset)
    issue = pd.Timestamp("2025-06-13T09Z")
    request = ForecastRequest(issue, spec().target_policy(issue))
    context = _build_context(_prepare_data(pv()), request, spec())
    with_weather = _build_context(_prepare_data(pv()), request, spec(), store(["2025-06-13"]))
    assert_frame_equal(lightgbm_features(context), lightgbm_features(with_weather))
    assert_frame_equal(
        original.predict(context).predictions, explicit.predict(with_weather).predictions
    )
    assert original.model_id == "lightgbm"


def test_fit_rejects_future_weather_audit_and_missing_feature_values() -> None:
    model = fitted()
    dataset = build_training_dataset(
        pv(),
        spec(),
        fit_time="2025-06-13T08Z",
        feature_set="history+nwp",
        weather_provider=store(["2025-06-10", "2025-06-11"]),
    )
    rows = dataset.rows.copy()
    rows["weather_available_at"] = rows.issue_time + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="historical issue"):
        model.fit(replace(dataset, rows=rows))
    X = dataset.X.copy()
    X.loc[0, FEATURES[0]] = np.nan
    with pytest.raises(WeatherError, match="complete"):
        validate_feature_matrix(X, "history+nwp")
