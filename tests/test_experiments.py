"""Offline experiment configuration, snapshot integrity and common-support diagnostics."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from solarcast.evaluation import compare_models
from solarcast.experiments import (
    ExperimentConfig,
    daily_origins,
    daylight,
    grouped_diagnostics,
    load_snapshot,
    quality_summary,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/experiments/lightgbm_v1.toml"


def normalized() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_time": pd.date_range("2025-01-01T00:00Z", periods=4, freq="15min"),
            "pv_generation_mw": pd.array([0, None, -1, 3], dtype="Float64"),
            "monitored_capacity_mw": pd.array([10] * 4, dtype="Float64"),
            "source_dataset": "ods032",
            "source_target_field": "measured",
            "retrieved_at": pd.Timestamp("2026-11-14T12:00Z"),
        }
    )


def test_real_config_and_dst_schedule() -> None:
    config = ExperimentConfig.load(CONFIG)
    assert len(config.origins("test")) == 245
    assert config.model.n_estimators == 100
    assert daily_origins("2025-03-29", "2025-03-31", "11:00").hour.tolist() == [10, 9]
    with pytest.raises(ValueError):
        daily_origins("2025-01-01", "2025-01-02", "11:00+01:00")
    with pytest.raises(ValueError):
        daily_origins("2025-03-30", "2025-03-31", "02:30")


@pytest.mark.parametrize(
    "changes",
    [
        {"fit_time": "2024-12-30T10:00Z"},
        {"validation_end": "2025-03-01"},
        {"data_start": "2024-01-01T00:00Z"},
        {"data_end": "2025-10-01T00:00Z"},
        {"reference_model": "nonexistent"},
        {"ranking_metric": "mae"},
        {"baseline_lags": ["24h", "1D", "48h", "7D"]},
        {"representative_days": ["2026-01-01"]},
        {"safety_lag": "-1min"},
    ],
)
def test_invalid_split_or_policy_fails(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(ExperimentConfig.load(CONFIG), **changes).validate()


def test_missingness_audit_keeps_nulls_and_negatives_distinct() -> None:
    data = normalized().drop(index=0)
    quality = quality_summary(
        data, pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2025-01-01T01:00Z")
    )
    assert quality["missing_interval_rows"] == 1
    assert quality["null_pv_observations"] == 1
    assert quality["negative_pv_observations"] == 1
    with pytest.raises(ValueError, match="unique"):
        quality_summary(
            pd.concat([data, data]),
            pd.Timestamp("2025-01-01T00:00Z"),
            pd.Timestamp("2025-01-01T01:00Z"),
        )


def test_cache_reuses_hash_checked_data_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_fetch(self: object, *, start: object, end: object) -> pd.DataFrame:
        calls.append((start, end))
        return normalized()

    monkeypatch.setattr("solarcast.experiments.EliaPVClient.fetch_history", fake_fetch)
    config = replace(
        ExperimentConfig.load(CONFIG), data_start="2025-01-01T00:00Z", data_end="2025-01-01T01:00Z"
    )
    first, manifest = load_snapshot(config, tmp_path)
    second, again = load_snapshot(config, tmp_path)
    assert len(calls) == 1 and manifest == again
    pd.testing.assert_series_equal(first.pv_generation_mw, second.pv_generation_mw)
    cached = tmp_path / manifest[0]["file"]
    cached.write_text(cached.read_text() + "corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/bounds"):
        load_snapshot(config, tmp_path)
    assert len(calls) == 1


def test_network_failure_preserves_valid_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(self: object, *, start: object, end: object) -> pd.DataFrame:
        raise RuntimeError("network down")

    monkeypatch.setattr("solarcast.experiments.EliaPVClient.fetch_history", failed)
    config = replace(
        ExperimentConfig.load(CONFIG), data_start="2025-01-01T00:00Z", data_end="2025-01-01T01:00Z"
    )
    with pytest.raises(RuntimeError, match="network down"):
        load_snapshot(config, tmp_path)
    assert not list((tmp_path / config.cache_dir).glob("*.csv"))


def test_diagnostic_groups_keep_common_support_and_metrics() -> None:
    issue = pd.Timestamp("2025-03-01T10:00Z")
    times = pd.date_range("2025-03-01T23:00Z", periods=96, freq="15min")
    one = pd.DataFrame(
        {
            "issue_time": issue,
            "target_time": times,
            "model_id": "a",
            "actual_mw": np.arange(96, dtype=float),
            "prediction_mw": 0.0,
            "horizon": times - issue,
            "status": "ok",
        }
    )
    two = one.assign(model_id="b", prediction_mw=1.0)
    two.loc[0, "prediction_mw"] = np.nan
    aligned = compare_models([one, two], reference_model="a").aligned_predictions
    daily = grouped_diagnostics(aligned, "a", "local_date")
    assert daily.n_scored.tolist() == [95, 95]
    groups = grouped_diagnostics(aligned, "a", "horizon_6h")
    assert groups.groupby("model_id").n_scored.sum().tolist() == [95, 95]
    flags = daylight(one[["issue_time", "target_time", "horizon"]])
    assert flags.dtype == bool and 0 < flags.sum() < 96


def test_unknown_config_key_rejected(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(CONFIG.read_text() + "\nunknown = 2\n", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        ExperimentConfig.load(config)
