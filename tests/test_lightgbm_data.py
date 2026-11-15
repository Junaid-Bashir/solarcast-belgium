"""Origin-specific supervised construction; does not require optional LightGBM."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, _build_context, _prepare_data
from solarcast.models import ForecastRequest
from solarcast.models.lightgbm_data import FEATURE_NAMES, build_training_dataset, lightgbm_features


def source() -> pd.DataFrame:
    times = pd.date_range("2026-01-01T00:00Z", periods=12 * 96, freq="15min")
    return pd.DataFrame(
        {
            "target_time": times,
            "pv_generation_mw": pd.array(range(len(times)), dtype="Float64"),
            "monitored_capacity_mw": 999999.0,
            "elia_forecast": 999999.0,
        }
    )


def next_hour(issue: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(issue + pd.Timedelta(hours=1), periods=2, freq="15min")


def spec() -> BacktestSpec:
    return BacktestSpec(
        pd.DatetimeIndex(["2026-01-09T09:00Z", "2026-01-10T09:00Z"]),
        availability_policy=EventTimeCutoff("30min"),
        target_policy=next_hour,
        context_window="8D",
    )


def test_builder_reconstructs_features_at_each_origin_and_keeps_audit_rows() -> None:
    data = source()
    original = data.copy(deep=True)
    dataset = build_training_dataset(data, spec(), fit_time="2026-01-11T00:00Z")
    assert tuple(dataset.X.columns) == FEATURE_NAMES
    assert len(dataset.X) == len(dataset.y) == len(dataset.rows) == 4
    assert dataset.n_requested == 4 and dataset.n_missing_labels == 0
    assert list(dataset.X.horizon_hours) == [1, 1.25] * 2
    assert dataset.X.pv_lag_24h_mw.notna().all()
    assert not dataset.X.eq(999999).any().any()
    context = _build_context(
        _prepare_data(data),
        ForecastRequest(spec().issue_times[0], next_hour(spec().issue_times[0])),
        spec(),
    )
    assert_frame_equal(dataset.X.iloc[:2].reset_index(drop=True), lightgbm_features(context))
    assert (dataset.rows.history_cutoff <= dataset.rows.issue_time).all()
    assert_frame_equal(data, original)


def test_mandatory_future_pv_and_target_leakage_canary() -> None:
    data = source()
    one = BacktestSpec(
        spec().issue_times[:1],
        availability_policy=EventTimeCutoff("30min"),
        target_policy=next_hour,
        context_window="8D",
    )
    ordinary = build_training_dataset(data, one, fit_time="2026-01-11T00:00Z")
    changed = data.copy(deep=True)
    changed.loc[changed.target_time >= pd.Timestamp("2026-01-09T08:30Z"), "pv_generation_mw"] = (
        999999
    )
    canary = build_training_dataset(changed, one, fit_time="2026-01-11T00:00Z")
    assert list(canary.y) == [999999, 999999]  # Future target is allowed as y after it matures.
    assert_frame_equal(ordinary.X, canary.X)  # But no changed observation enters X at issue.
    assert not canary.X.eq(999999).any().any()


def test_missing_targets_are_counted_and_missing_lags_remain_nan() -> None:
    data = source()
    data.loc[data.target_time == pd.Timestamp("2026-01-09T10:00Z"), "pv_generation_mw"] = pd.NA
    short = BacktestSpec(
        spec().issue_times,
        availability_policy=EventTimeCutoff("30min"),
        target_policy=next_hour,
        context_window="1h",
    )
    dataset = build_training_dataset(data, short, fit_time="2026-01-11T00:00Z")
    assert dataset.n_missing_labels == 1 and dataset.n_requested == 4
    assert len(dataset.y) == 3
    assert dataset.X.pv_lag_24h_mw.isna().all()
    assert dataset.X.pv_lag_48h_mw.isna().all()
    assert dataset.X.pv_lag_7d_mw.isna().all()
    assert dataset.X.pv_trailing_24h_mean_mw.isna().all()
    assert np.isfinite(dataset.y).all()


def test_training_window_filters_origins_without_moving_context_semantics() -> None:
    dataset = build_training_dataset(
        source(), spec(), fit_time="2026-01-11T00:00Z", training_window="1D"
    )
    assert set(dataset.rows.issue_time) == {pd.Timestamp("2026-01-10T09:00Z")}
    assert dataset.n_requested == 2


@pytest.mark.parametrize("boundary", ["2026-01-09T09:00Z", "2026-01-10T10:00Z"])
def test_training_origins_or_targets_cannot_cross_fit_boundary(boundary: str) -> None:
    with pytest.raises(ValueError, match="fit_time|label cutoff"):
        build_training_dataset(source(), spec(), fit_time=boundary)


def test_lazy_optional_model_import_and_config_do_not_import_lightgbm() -> None:
    code = """
import sys
sys.modules['lightgbm'] = None
import solarcast.models
from solarcast.models.lightgbm import LightGBMConfig, LightGBMForecaster
assert LightGBMForecaster(LightGBMConfig()).model_id == 'lightgbm'
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_optional_extra_has_actionable_fit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from solarcast.models.lightgbm import LightGBMForecaster

    dataset = build_training_dataset(source(), spec(), fit_time="2026-01-11T00:00Z")
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(ImportError, match="dev,lightgbm"):
        LightGBMForecaster().fit(dataset)
