"""Offline feature contracts, DST and information-cutoff regression tests."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solarcast.features import (
    BELGIUM_REFERENCE_LOCATION,
    SolarLocation,
    calendar_features,
    pv_lag_features,
    pv_rolling_features,
    solar_features,
)


def history(stamps: pd.DatetimeIndex, values: list[float | None] | None = None) -> pd.DataFrame:
    """Minimal normalized fixture, including a capacity column that must be ignored."""
    return pd.DataFrame(
        {
            "target_time": stamps,
            "pv_generation_mw": pd.array(
                values if values is not None else list(range(len(stamps))), dtype="Float64"
            ),
            "monitored_capacity_mw": 1000.0,
        }
    )


def test_calendar_clock_fields_and_offset() -> None:
    times = pd.DatetimeIndex(["2026-01-02T00:45Z", "2026-07-01T23:15Z"])
    result = calendar_features(times)
    assert list(result.local_hour) == [1, 1]
    assert list(result.local_minute) == [45, 15]
    assert list(result.quarter_hour_in_hour) == [3, 1]
    assert list(result.day_of_week) == [4, 3]  # Monday = 0
    assert list(result.day_of_year) == [2, 183]
    assert list(result.month) == [1, 7]
    assert list(result.utc_offset_minutes) == [60, 120]
    assert pd.DatetimeIndex(result.target_time).equals(times)


def test_clock_cycles_are_continuous_at_midnight() -> None:
    times = pd.DatetimeIndex(["2026-01-01T22:45Z", "2026-01-01T23:00Z"])
    result = calendar_features(times)
    assert result.time_of_day_sin.iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert result.time_of_day_cos.iloc[1] == pytest.approx(1.0)
    distance = np.hypot(
        result.time_of_day_sin.diff().iloc[1], result.time_of_day_cos.diff().iloc[1]
    )
    assert distance < 0.07


@pytest.mark.parametrize("year,days", [(2024, 366), (2025, 365)])
def test_year_phase_accounts_for_leap_year(year: int, days: int) -> None:
    result = calendar_features(pd.DatetimeIndex([f"{year}-07-01T00:00:00+02:00"]))
    phase = 2 * np.pi * (result.day_of_year.iloc[0] - 1) / days
    assert result.day_of_year_sin.iloc[0] == pytest.approx(np.sin(phase))
    assert result.day_of_year_cos.iloc[0] == pytest.approx(np.cos(phase))


@pytest.mark.parametrize(
    "day,end,count",
    [
        ("2026-03-29", "2026-03-30", 92),
        ("2026-10-25", "2026-10-26", 100),
    ],
)
def test_dst_calendar_and_solar_preserve_all_unique_intervals(
    day: str,
    end: str,
    count: int,
) -> None:
    times = pd.date_range(
        pd.Timestamp(day, tz="Europe/Brussels"),
        pd.Timestamp(end, tz="Europe/Brussels"),
        freq="15min",
        inclusive="left",
    )
    calendar = calendar_features(times)
    solar = solar_features(times)
    assert len(calendar) == len(solar) == count
    assert calendar.target_time.is_unique
    assert calendar.target_time.equals(solar.target_time)
    if count == 92:
        assert 2 not in set(calendar.local_hour)
    else:
        repeated = calendar[calendar.local_hour == 2]
        assert len(repeated) == 8
        assert set(repeated.utc_offset_minutes) == {60, 120}
        assert list(repeated.local_minute) == [0, 15, 30, 45] * 2


def test_solar_and_clearsky_are_physically_sensible_without_modifying_target() -> None:
    times = pd.DatetimeIndex(["2026-06-21T00:00Z", "2026-06-21T11:45Z"])
    result = solar_features(times)
    night, noon = result.iloc[0], result.iloc[1]
    assert night.solar_apparent_elevation_deg < 0
    assert not night.is_daylight
    assert night.clearsky_ghi_w_m2 == 0
    assert 55 < noon.solar_apparent_elevation_deg < 70
    assert noon.is_daylight
    assert 500 < noon.clearsky_ghi_w_m2 < 1200
    assert np.allclose(result.solar_apparent_elevation_deg + result.solar_apparent_zenith_deg, 90)
    assert result.solar_azimuth_deg.between(0, 360).all()
    assert result.clearsky_ghi_w_m2.ge(0).all()


def test_solar_location_is_explicit_and_configurable() -> None:
    assert BELGIUM_REFERENCE_LOCATION.latitude_deg == 50.85
    assert BELGIUM_REFERENCE_LOCATION.longitude_deg == 4.35
    times = pd.DatetimeIndex(["2026-06-21T12:00Z"])
    north = solar_features(times, location=SolarLocation(60, 4.35))
    default = solar_features(times)
    assert north.solar_apparent_elevation_deg.iloc[0] < default.solar_apparent_elevation_deg.iloc[0]


@pytest.mark.parametrize(
    "location", [(91, 0, 0), (0, 181, 0), (float("nan"), 0, 0), (0, 0, float("inf"))]
)
def test_invalid_location_fails(location: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        SolarLocation(*location)


def test_exact_lags_with_gaps_and_unordered_history() -> None:
    stamps = pd.DatetimeIndex(
        [
            "2026-06-01T12:00Z",
            "2026-06-06T12:00Z",
            "2026-06-07T12:00Z",
            "2026-06-07T12:30Z",
        ]
    )
    source = history(stamps, [7, 48, 24, 999]).iloc[::-1]
    targets = pd.DatetimeIndex(["2026-06-08T12:00Z", "2026-06-08T12:15Z"])
    result = pv_lag_features(source, targets, history_cutoff="2026-06-08T00:00Z")
    assert result.pv_lag_24h_mw.iloc[0] == 24
    assert result.pv_lag_48h_mw.iloc[0] == 48
    assert result.pv_lag_7d_mw.iloc[0] == 7
    assert result.iloc[1, 1:].isna().all()  # Gap is not a shift to the next available row.


def test_lag_cutoff_excludes_incomplete_interval_and_later_history() -> None:
    source = history(pd.DatetimeIndex(["2026-06-07T11:45Z", "2026-06-07T12:00Z"]), [1, 2])
    targets = pd.DatetimeIndex(["2026-06-08T11:45Z", "2026-06-08T12:00Z"])
    result = pv_lag_features(source, targets, history_cutoff="2026-06-07T12:00Z")
    assert result.pv_lag_24h_mw.iloc[0] == 1
    assert pd.isna(result.pv_lag_24h_mw.iloc[1])
    changed = source.copy(deep=True)
    changed.loc[1, "pv_generation_mw"] = 99999
    assert_frame_equal(
        result, pv_lag_features(changed, targets, history_cutoff="2026-06-07T12:00Z")
    )


@pytest.mark.parametrize(
    "target,elapsed,local_previous",
    [
        ("2026-03-29T10:00Z", "2026-03-28T10:00Z", "2026-03-28T11:00Z"),
        ("2026-10-25T11:00Z", "2026-10-24T11:00Z", "2026-10-24T10:00Z"),
    ],
)
def test_24h_is_elapsed_time_not_previous_local_clock(
    target: str,
    elapsed: str,
    local_previous: str,
) -> None:
    source = history(pd.DatetimeIndex([elapsed, local_previous]), [10, 20])
    result = pv_lag_features(source, pd.DatetimeIndex([target]), history_cutoff=target)
    assert result.pv_lag_24h_mw.iloc[0] == 10


def test_null_history_and_empty_history_leave_missing_lags() -> None:
    source = history(pd.DatetimeIndex(["2026-06-01T12:00Z"]), [None])
    targets = pd.DatetimeIndex(["2026-06-02T12:00Z"])
    for data in (source, source.iloc[:0]):
        result = pv_lag_features(data, targets, history_cutoff="2026-06-02T00:00Z")
        assert result.iloc[:, 1:].isna().all().all()
        assert all(str(dtype) == "Float64" for dtype in result.dtypes.iloc[1:])


def test_rolling_exact_window_excludes_current_and_future_observations() -> None:
    times = pd.date_range("2026-06-01T00:00Z", periods=194, freq="15min")
    source = history(times)
    targets = pd.DatetimeIndex(["2026-06-02T00:00Z"])
    result = pv_rolling_features(source, targets, history_cutoff="2026-06-03T00:00Z")
    assert result.pv_trailing_24h_mean_mw.iloc[0] == pytest.approx(47.5)
    assert result.pv_trailing_24h_max_mw.iloc[0] == 95
    changed = source.copy(deep=True)
    changed.loc[changed.target_time >= targets[0], "pv_generation_mw"] = 999999
    assert_frame_equal(
        result, pv_rolling_features(changed, targets, history_cutoff="2026-06-03T00:00Z")
    )


def test_rolling_future_targets_are_anchored_at_cutoff() -> None:
    source = history(pd.date_range("2026-06-01T00:00Z", periods=192, freq="15min"))
    targets = pd.DatetimeIndex(["2026-06-02T01:00Z", "2026-06-02T23:45Z"])
    result = pv_rolling_features(source, targets, history_cutoff="2026-06-02T00:00Z")
    assert list(result.pv_trailing_24h_mean_mw) == [47.5, 47.5]
    assert list(result.pv_trailing_24h_max_mw) == [95, 95]
    changed = source.copy(deep=True)
    changed.loc[96:, "pv_generation_mw"] = -999999
    assert_frame_equal(
        result, pv_rolling_features(changed, targets, history_cutoff="2026-06-02T00:00Z")
    )


@pytest.mark.parametrize("missing", ["gap", "null", "insufficient"])
def test_rolling_requires_a_complete_nonnull_elapsed_window(missing: str) -> None:
    source = history(pd.date_range("2026-06-01T00:00Z", periods=96, freq="15min"))
    if missing == "gap":
        source = source.drop(index=24)
    elif missing == "null":
        source.loc[24, "pv_generation_mw"] = pd.NA
    else:
        source = source.iloc[1:]
    result = pv_rolling_features(
        source, pd.DatetimeIndex(["2026-06-02T00:00Z"]), history_cutoff="2026-06-02T00:00Z"
    )
    assert result.iloc[:, 1:].isna().all().all()


def test_inputs_are_immutable_and_output_deterministic() -> None:
    source = history(pd.date_range("2026-06-01T00:00Z", periods=700, freq="15min"))
    original = source.copy(deep=True)
    times = pd.DatetimeIndex(source.target_time.iloc[-4:])
    for function in (calendar_features, solar_features):
        assert_frame_equal(function(times), function(times))
    for historical_function in (pv_lag_features, pv_rolling_features):
        a = historical_function(source, times, history_cutoff="2026-06-08T00:00Z")
        b = historical_function(source, times, history_cutoff="2026-06-08T00:00Z")
        assert_frame_equal(a, b)
        assert "monitored_capacity_mw" not in a
        assert "pv_generation_mw" not in a
    assert_frame_equal(source, original)


@pytest.mark.parametrize("end", ["2026-03-30", "2026-10-26"])
def test_rolling_uses_elapsed_24h_across_dst(end: str) -> None:
    target = pd.Timestamp(end, tz="Europe/Brussels").tz_convert("UTC")
    stamps = pd.date_range(target - pd.Timedelta(hours=24), target, freq="15min", inclusive="left")
    result = pv_rolling_features(history(stamps), pd.DatetimeIndex([target]), history_cutoff=target)
    assert result.pv_trailing_24h_mean_mw.iloc[0] == 47.5
    assert result.pv_trailing_24h_max_mw.iloc[0] == 95


@pytest.mark.parametrize(
    "times",
    [
        pd.DatetimeIndex(["2026-01-01"]),
        pd.DatetimeIndex(["2026-01-01T00:01Z"]),
        pd.DatetimeIndex(["2026-01-01T00:00Z", "2026-01-01T00:00Z"]),
        pd.DatetimeIndex([pd.NaT], tz="UTC"),
    ],
)
def test_bad_target_times_rejected(times: pd.DatetimeIndex) -> None:
    for function in (calendar_features, solar_features):
        with pytest.raises(ValueError):
            function(times)


@pytest.mark.parametrize("cutoff", ["2026-06-01", "2026-06-01T00:01Z", "not a date"])
def test_history_cutoff_must_be_explicit_and_aligned(cutoff: str) -> None:
    times = pd.DatetimeIndex(["2026-06-02T00:00Z"])
    with pytest.raises(ValueError):
        pv_lag_features(history(times), times, history_cutoff=cutoff)


def test_duplicate_naive_and_non_numeric_history_rejected() -> None:
    times = pd.DatetimeIndex(["2026-06-01T00:00Z"])
    source = history(times)
    invalid = [
        pd.concat([source, source]),
        source.assign(target_time=times.tz_localize(None)),
        source.assign(pv_generation_mw="bad"),
        source.assign(pv_generation_mw=float("inf")),
    ]
    for frame in invalid:
        with pytest.raises(ValueError):
            pv_lag_features(frame, times, history_cutoff="2026-06-02T00:00Z")


def test_empty_targets_produce_typed_empty_tables() -> None:
    times = pd.DatetimeIndex([], tz="UTC")
    for function in (calendar_features, solar_features):
        result = function(times)
        assert result.empty
        assert str(result.target_time.dtype) == "datetime64[ns, UTC]"
    for historical_function in (pv_lag_features, pv_rolling_features):
        result = historical_function(history(times), times, history_cutoff="2026-06-02T00:00Z")
        assert result.empty
        assert str(result.target_time.dtype) == "datetime64[ns, UTC]"
