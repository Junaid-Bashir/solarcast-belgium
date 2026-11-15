"""Offline NWP semantics, coverage and mandatory future-vintage leakage canary."""

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest, day_ahead_local_day
from solarcast.data.weather import (
    FEATURES,
    ForecastRun,
    NWPAvailability,
    SpatialWeights,
    WeatherError,
    WeatherStore,
    aligned_cells,
    hourly_radiation,
    select_nwp_vintage,
    weather_features,
)
from solarcast.data.weather.types import PRODUCT, PROVIDER, WeatherCovariates
from solarcast.models import Forecast, ForecastContext, ForecastRequest

CELLS = ((50.5, 3.0), (50.5, 4.5), (50.5, 5.5))


@pytest.fixture
def weights() -> SpatialWeights:
    return SpatialWeights(
        pd.DataFrame(
            {"latitude": [50.5] * 3, "longitude": [3.0, 4.5, 5.5], "area_m2": [1.0, 2.0, 1.0]}
        ),
        expected_cells=CELLS,
        boundary_id="synthetic-three-cell-fixture",
    )


def native(
    reference: str = "2025-06-10T00Z",
    *,
    radiation: float = 100,
    steps: range = range(18, 49),
    version: str = "fixture",
) -> ForecastRun:
    """Realistic wide normalized provider fixture with six-hour running averages."""
    ref = pd.Timestamp(reference)
    rows = []
    for step in steps:
        start = ((step - 1) // 6) * 6
        for i, (latitude, longitude) in enumerate(CELLS):
            rows.append(
                {
                    "reference_time": ref,
                    "valid_time": ref + pd.Timedelta(hours=step),
                    "forecast_lead": pd.Timedelta(hours=step),
                    "latitude": latitude,
                    "longitude": longitude,
                    "dswrf_start_hour": start,
                    "dswrf_end_hour": step,
                    "dswrf_average_w_m2": radiation + i * 100,
                    "cloud_percent": 20.0 + i * 20,
                    "temperature_2m_k": 273.15 + i * 10,
                }
            )
    return ForecastRun(
        ref,
        pd.DataFrame(rows),
        version=version,
        metadata={"provider": PROVIDER, "product": PRODUCT, "fixture": True},
    )


def changed(
    run: ForecastRun,
    frame: pd.DataFrame,
    *,
    revision: bool = False,
    observed: str | datetime | None = None,
    version_available_at: str | datetime | None = None,
) -> ForecastRun:
    return ForecastRun(
        run.reference_time,
        frame,
        version=run.version,
        metadata=run.metadata,
        revision=revision,
        observed_available_at=observed,
        version_available_at=version_available_at,
    )


def tomorrow() -> pd.DatetimeIndex:
    return pd.date_range("2025-06-11T12Z", periods=4, freq="15min")


def test_native_times_leads_and_owned_data() -> None:
    run = native()
    assert str(run.data.reference_time.dt.tz) == str(run.data.valid_time.dt.tz) == "UTC"
    assert (run.data.valid_time - run.reference_time).equals(run.data.forecast_lead)
    assert run.observed_available_at is None
    copy = run.data
    copy["temperature_2m_k"] = 999999
    assert run.data.temperature_2m_k.max() < 300
    metadata = run.metadata
    metadata["provider"] = "reanalysis"
    assert run.metadata["provider"] == PROVIDER


@pytest.mark.parametrize("column", ["reference_time", "valid_time"])
def test_reject_naive_native_times(column: str) -> None:
    run = native()
    frame = run.data
    frame[column] = frame[column].dt.tz_localize(None)
    with pytest.raises(WeatherError, match="timezone"):
        changed(run, frame)
    with pytest.raises(WeatherError, match="timezone"):
        ForecastRun("2025-06-10", run.data, version="x", metadata=run.metadata)


@pytest.mark.parametrize("offset,eligible", [("-1ns", True), ("0ns", True), ("1ns", False)])
@pytest.mark.parametrize("mode", ["assumed", "observed"])
def test_availability_inclusive_boundaries(
    weights: SpatialWeights, offset: str, eligible: bool, mode: str
) -> None:
    run = native()
    available = pd.Timestamp("2025-06-10T06Z")
    policy = NWPAvailability() if mode == "assumed" else NWPAvailability("observed")
    if mode == "observed":
        run = changed(run, run.data, observed=available)
    issue = available - pd.Timedelta(offset)
    if eligible:
        selected = select_nwp_vintage([run], issue, tomorrow(), weights=weights, policy=policy)
        assert selected.available_at == available
        assert selected.availability_basis == mode
        assert selected.policy_name == policy.name
    else:
        with pytest.raises(WeatherError, match="no eligible"):
            select_nwp_vintage([run], issue, tomorrow(), weights=weights, policy=policy)


def test_mandatory_weather_leakage_canary_through_backtester(weights: SpatialWeights) -> None:
    older = native(version="older")
    newer = native("2025-06-10T06Z", radiation=999999, version="newer")
    # Every radiation cell has the sentinel (rather than only a national mean).
    new_frame = newer.data
    new_frame["dswrf_average_w_m2"] = 999999.0
    newer = changed(newer, new_frame)
    targets = tomorrow()
    store = WeatherStore((newer, older), weights)
    issue = pd.Timestamp("2025-06-10T09Z")
    selected = store.for_issue(issue, targets)
    assert selected.reference_time == older.reference_time
    assert selected.run_version == "older"
    assert not (selected.features[list(FEATURES)] == 999999).any().any()
    assert selected.features[list(FEATURES)].max().max() <= 300
    later = store.for_issue(pd.Timestamp("2025-06-10T12Z"), targets)
    assert later.reference_time == newer.reference_time
    assert (later.features[list(FEATURES[:3])] == 999999).all().all()

    class Recorder:
        model_id = "weather-context-recorder"

        def predict(self, context: ForecastContext) -> Forecast:
            assert context.weather is not None
            assert context.weather.run_version == "older"
            assert context.weather.issue_time == context.request.issue_time
            assert pd.DatetimeIndex(context.weather.features.target_time).equals(targets)
            assert (context.weather.features[list(FEATURES)].to_numpy() < 999999).all()
            return Forecast(
                context.request,
                pd.DataFrame({"target_time": targets, "prediction_mw": 1.0}),
                model_id=self.model_id,
            )

    history = pd.DataFrame(
        {
            "target_time": pd.date_range("2025-06-10T07Z", periods=4, freq="15min"),
            "pv_generation_mw": 1.0,
        }
    )
    result = backtest(
        model=Recorder(),
        data=history,
        weather_provider=store,
        spec=BacktestSpec(
            pd.DatetimeIndex([issue]),
            availability_policy=EventTimeCutoff("30min"),
            target_policy=lambda _: targets,
        ),
    )
    assert result.origins.status.tolist() == ["ok"]
    assert result.origins.weather_run_version.tolist() == ["older"]
    assert result.origins.weather_availability_basis.tolist() == ["assumed"]


def test_observed_mode_and_revision_never_backdated(weights: SpatialWeights) -> None:
    run = native()
    issue = pd.Timestamp("2025-06-10T09Z")
    with pytest.raises(WeatherError, match="no eligible"):
        select_nwp_vintage(
            [run], issue, tomorrow(), weights=weights, policy=NWPAvailability("observed")
        )
    observed = changed(run, run.data, observed="2025-06-10T08Z")
    selection = select_nwp_vintage(
        [observed], issue, tomorrow(), weights=weights, policy=NWPAvailability("observed")
    )
    assert selection.available_at == pd.Timestamp("2025-06-10T08Z")
    assert selection.availability_basis == "observed"
    for corrected in (
        changed(run, run.data, revision=True),
        changed(run, run.data, revision=True, version_available_at="2025-06-10T10Z"),
    ):
        with pytest.raises(WeatherError, match="no eligible"):
            select_nwp_vintage([corrected], issue, tomorrow(), weights=weights)
    with pytest.raises(WeatherError, match="ambiguous"):
        select_nwp_vintage([run, observed], issue, tomorrow(), weights=weights)


def test_radiation_running_average_energy_reset_and_no_cross_run() -> None:
    run = native(steps=range(22, 27))
    frame = run.data
    # A(18,22)=100; A(18,23)=120 -> E(22,23)=720000 J/m2 -> mean 200.
    # A(18,24)=150 -> E(23,24)=1080000 -> 300. Reset A(24,25)=50.
    means = {22: 100.0, 23: 120.0, 24: 150.0, 25: 50.0, 26: 75.0}
    frame["dswrf_average_w_m2"] = frame.dswrf_end_hour.map(means)
    output = hourly_radiation(changed(run, frame))
    assert output.loc[output.dswrf_end_hour == 22, "ghi_hourly_w_m2"].isna().all()
    for end, mean in ((23, 200.0), (24, 300.0), (25, 50.0), (26, 100.0)):
        assert output.loc[output.dswrf_end_hour == end, "ghi_hourly_w_m2"].eq(mean).all()
    other = native("2025-06-10T06Z", steps=range(24, 25), radiation=999999)
    assert hourly_radiation(other).ghi_hourly_w_m2.isna().all()
    assert (output.loc[output.dswrf_end_hour == 23, "ghi_hourly_w_m2"] * 3600).eq(720000).all()


def test_negative_radiation_is_reported_not_clipped() -> None:
    run = native(steps=range(22, 24))
    frame = run.data
    frame.loc[frame.dswrf_end_hour == 23, "dswrf_average_w_m2"] = 0.0
    with pytest.raises(WeatherError, match="negative"):
        hourly_radiation(changed(run, frame))


def test_temporal_flux_energy_and_state_interval_means(weights: SpatialWeights) -> None:
    run = native(steps=range(24, 26))
    frame = run.data
    frame["cloud_percent"] = np.where(frame.dswrf_end_hour == 24, 0.0, 80.0)
    frame["temperature_2m_k"] = np.where(frame.dswrf_end_hour == 24, 273.15, 281.15)
    targets = pd.date_range("2025-06-11T00Z", periods=4, freq="15min")
    aligned = aligned_cells(changed(run, frame), targets, weights)
    first = aligned.loc[aligned.longitude == 3.0]
    assert first.ghi_w_m2.tolist() == [100.0] * 4
    assert (first.ghi_w_m2 * 900).sum() == 100 * 3600
    assert first.cloud_fraction.tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7])
    assert first.temperature_2m_c.tolist() == pytest.approx([1.0, 3.0, 5.0, 7.0])


def test_spatial_weighted_features_and_defensive_copy(weights: SpatialWeights) -> None:
    run = native()
    features = weather_features(run, tomorrow(), weights)
    assert features.iloc[0][list(FEATURES)].to_numpy(dtype=float) == pytest.approx(
        [100.0, 200.0, 300.0, np.sqrt(5000), 0.4, np.sqrt(0.02), 10.0]
    )
    selected = select_nwp_vintage([run], "2025-06-10T09Z", tomorrow(), weights=weights)
    copy = selected.features
    copy.loc[:, list(FEATURES)] = 999999.0
    assert_frame_equal(selected.features, features)


@pytest.mark.parametrize(
    "problem", ["time", "cell", "cloud", "temperature", "radiation", "predecessor"]
)
def test_missing_weather_fails_without_filling(weights: SpatialWeights, problem: str) -> None:
    run = native()
    frame = run.data
    if problem == "time":
        frame = frame.loc[frame.dswrf_end_hour != 37]
    elif problem == "cell":
        frame = frame.loc[~((frame.dswrf_end_hour == 37) & (frame.longitude == 5.5))]
    elif problem == "predecessor":
        # Target hour37 needs average ending37 plus38 (same start36).
        frame = frame.loc[frame.dswrf_end_hour != 37]
    else:
        column = {
            "cloud": "cloud_percent",
            "temperature": "temperature_2m_k",
            "radiation": "dswrf_average_w_m2",
        }[problem]
        frame.loc[frame.dswrf_end_hour == 37, column] = np.nan
    targets = tomorrow() + (pd.Timedelta(hours=1) if problem == "predecessor" else pd.Timedelta(0))
    with pytest.raises(WeatherError, match="no eligible complete"):
        select_nwp_vintage([changed(run, frame)], "2025-06-10T09Z", targets, weights=weights)


def test_whole_run_fallback_never_stitches_newer_partial_run(weights: SpatialWeights) -> None:
    older = native(version="old")
    newer = native("2025-06-10T06Z", radiation=999999, version="partial")
    newer = changed(newer, newer.data.loc[newer.data.dswrf_end_hour != 31])
    selected = select_nwp_vintage([older, newer], "2025-06-10T12Z", tomorrow(), weights=weights)
    assert selected.run_version == "old"
    assert selected.selection_notes and "missing native" in selected.selection_notes[0]
    assert selected.features[FEATURES[0]].eq(100).all()


@pytest.mark.parametrize("fault", ["duplicate", "lead", "cloud", "unit", "missing-variable"])
def test_native_invalid_contracts(fault: str) -> None:
    run = native()
    frame = run.data
    if fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif fault == "lead":
        frame["forecast_lead"] += pd.Timedelta(hours=1)
    elif fault == "cloud":
        frame["cloud_percent"] = 101.0
    elif fault == "unit":
        frame["temperature_2m_k"] = -1.0
    else:
        frame = frame.drop(columns="cloud_percent")
    with pytest.raises(WeatherError):
        changed(run, frame)


def test_context_rejects_misaligned_weather_and_weather_is_optional(
    weights: SpatialWeights,
) -> None:
    issue = pd.Timestamp("2025-06-10T09Z")
    request = ForecastRequest(issue, tomorrow())
    history = pd.DataFrame(
        {
            "target_time": pd.date_range("2025-06-10T07Z", periods=2, freq="15min"),
            "pv_generation_mw": 1.0,
        }
    )
    context = ForecastContext(request, history, history_cutoff=issue)
    assert context.weather is None
    weather = select_nwp_vintage([native()], issue, tomorrow(), weights=weights)
    assert (
        ForecastContext(request, history, history_cutoff=issue, weather=weather).weather is weather
    )
    with pytest.raises(ValueError, match="match"):
        ForecastContext(
            ForecastRequest(issue, tomorrow()[:2]), history, history_cutoff=issue, weather=weather
        )
    with pytest.raises(WeatherError, match="availability"):
        WeatherCovariates(
            weather.features,
            issue_time=issue,
            reference_time=weather.reference_time,
            available_at=issue + pd.Timedelta(1, "ns"),
            availability_basis="assumed",
            policy_name="x",
            run_version="x",
            weights_id="x",
        )


@pytest.mark.parametrize(
    "issue,count", [("2025-03-29T09Z", 92), ("2025-06-10T09Z", 96), ("2025-10-25T09Z", 100)]
)
def test_weather_alignment_dst_calendar_day(
    weights: SpatialWeights, issue: str, count: int
) -> None:
    stamp = pd.Timestamp(issue)
    targets = day_ahead_local_day(stamp)
    run = native(stamp.floor("D").isoformat())
    selected = select_nwp_vintage([run], stamp, targets, weights=weights)
    assert len(selected.features) == count
    assert pd.DatetimeIndex(selected.features.target_time).equals(targets)


def test_invalid_policy_and_naive_selection(weights: SpatialWeights) -> None:
    with pytest.raises(WeatherError):
        replace(NWPAvailability(), delay=pd.Timedelta(hours=-1))
    with pytest.raises(WeatherError, match="timezone"):
        select_nwp_vintage([native()], "2025-06-10", tomorrow(), weights=weights)
    with pytest.raises(WeatherError, match="timezone"):
        weather_features(native(), tomorrow().tz_localize(None), weights)
    run = native()
    with pytest.raises(WeatherError, match="timezone"):
        changed(run, run.data, observed="2025-06-10T08:00")
    with pytest.raises(WeatherError, match="declared revision"):
        changed(run, run.data, version_available_at="2025-06-10T10Z")


def test_missing_weather_fails_origin_before_model_call(weights: SpatialWeights) -> None:
    class MustNotRun:
        model_id = "must-not-receive-incomplete-weather"

        def predict(self, context: ForecastContext) -> Forecast:
            raise AssertionError("model was invoked despite missing weather")

    issue = pd.Timestamp("2025-06-10T09Z")
    history = pd.DataFrame(
        {
            "target_time": pd.date_range("2025-06-10T07Z", periods=2, freq="15min"),
            "pv_generation_mw": 1.0,
        }
    )
    result = backtest(
        model=MustNotRun(),
        data=history,
        weather_provider=WeatherStore((), weights),
        spec=BacktestSpec(
            pd.DatetimeIndex([issue]),
            availability_policy=EventTimeCutoff("30min"),
            target_policy=lambda _: tomorrow(),
            on_error="continue",
        ),
    )
    assert result.origins.error_type.tolist() == ["WeatherError"]
    assert len(result.predictions) == len(tomorrow())
    assert result.predictions.prediction_mw.isna().all()


def test_real_cropped_provider_fixture_is_normalized_and_transformable(
    weights: SpatialWeights,
) -> None:
    fixture = json.loads(
        Path("tests/fixtures/weather/gfs_20250610_00_f036_f038.json").read_text(encoding="utf8")
    )
    frame = pd.DataFrame(fixture["data"])
    for name in ("reference_time", "valid_time"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    frame["forecast_lead"] = pd.to_timedelta(frame.forecast_lead)
    run = ForecastRun(
        fixture["reference_time"],
        frame,
        version=fixture["source_version"],
        metadata={"provider": fixture["provider"], "product": fixture["product"]},
    )
    features = weather_features(
        run, pd.date_range("2025-06-11T12Z", periods=8, freq="15min"), weights
    )
    assert len(features) == 8
    assert features[list(FEATURES[:3])].min().min() > 700
    assert features[FEATURES[-1]].between(15, 25).all()
