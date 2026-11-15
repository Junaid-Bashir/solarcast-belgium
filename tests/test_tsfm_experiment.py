"""Offline benchmark identity, fair scoring, diagnostic and timing checks."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest
from solarcast.evaluation import compare_models
from solarcast.experiments import grouped_diagnostics, sha256, write_json
from solarcast.models import Forecast, ForecastContext
from solarcast.tsfm_experiment import (
    MODEL_IDS,
    TimedForecaster,
    TSFMExperimentConfig,
    cache_identity,
    compute_summary,
    daily_wins,
    error_correlations,
    load_cache,
    save_cache,
    selected_cases,
    verify_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_root(tmp_path: Path) -> tuple[TSFMExperimentConfig, Path]:
    # No ignored local data/artifacts are required in CI.
    parent = tmp_path / "configs/experiments/lightgbm_v1.toml"
    parent.parent.mkdir(parents=True)
    parent.write_bytes((ROOT / "configs/experiments/lightgbm_v1.toml").read_bytes())
    manifest = tmp_path / "artifacts/experiments/lightgbm_baseline_v1/snapshot.json"
    manifest.parent.mkdir(parents=True)
    write_json(manifest, [{"file": "data.csv", "sha256": "test", "start": "a", "end": "b"}])
    path = tmp_path / "config.toml"
    original = (ROOT / "configs/experiments/tsfm_zero_shot_v1.toml").read_text()
    path.write_text(
        original.replace(
            "98e0cd1f701a60a51f405d1bc6ff415952c9513c28a27fc38b75dca6cc3156e7", sha256(manifest)
        )
    )
    return TSFMExperimentConfig.load(path, tmp_path), tmp_path


def test_frozen_config_parent_schedule_and_timing_validation(
    config_root: tuple[TSFMExperimentConfig, Path],
) -> None:
    config, root = config_root
    parent = config.parent(root)
    assert config.models == MODEL_IDS and config.context_slots == 1344
    assert len(parent.origins("test")) == 245
    assert len(config.timing_spec(root).issue_times) == 2
    assert config.timing_spec(root).issue_times.isin(parent.origins("validation")).all()
    assert parent.model.random_state == 0 and parent.model.n_estimators == 100


@pytest.mark.parametrize(
    "changes",
    [
        {"models": ["chronos"]},
        {"context_slots": 672},
        {"point_policy": "mean"},
        {"device": "cuda"},
        {"torch_threads": True},
        {"seed": -1},
        {"tie_atol_mw": -1.0},
        {"timing_origin_dates": ["2025-06-15"]},
        {"chronos": {"checkpoint": "other", "revision": "main"}},
        {"t0": {"checkpoint": "other"}},
        {"output_dir": "../outside"},
        {"normal_plot_days": ["2025-06-16"]},
    ],
)
def test_config_rejects_methodology_drift(
    config_root: tuple[TSFMExperimentConfig, Path], changes: dict[str, Any]
) -> None:
    config, root = config_root
    with pytest.raises(ValueError):
        replace(config, **changes).validate(root)


def test_parent_and_manifest_are_hash_pinned(
    config_root: tuple[TSFMExperimentConfig, Path],
) -> None:
    config, root = config_root
    parent = root / config.parent_config
    parent.write_text(parent.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="parent"):
        config.validate(root)
    with pytest.raises(ValueError, match="snapshot"):
        replace(config, parent_sha256=sha256(parent), snapshot_sha256="wrong").validate(root)


def test_cache_identity_covers_every_nested_scientific_setting() -> None:
    payload: dict[str, Any] = {
        "model": {"revision": "abc", "context": 1344},
        "period": "Mar-Oct",
        "snapshot": "hash",
        "runtime": "0.6.1",
        "sources": {"adapter": "hash"},
        "stage": "test",
    }
    original = cache_identity(payload)
    assert original == cache_identity(dict(reversed(list(payload.items()))))
    for key, value in [
        ("model", {"revision": "different", "context": 1344}),
        ("model", {"revision": "abc", "context": 672}),
        ("period", "June"),
        ("snapshot", "changed"),
        ("runtime", "other"),
        ("sources", {}),
        ("stage", "timing"),
    ]:
        assert cache_identity({**payload, key: value}) != original


class FakeModel:
    def __init__(self, model_id: str, fail_second: bool = False) -> None:
        self.model_id = model_id
        self.fail_second = fail_second
        self.n = 0

    def predict(self, context: ForecastContext) -> Forecast:
        self.n += 1
        if self.fail_second and self.n == 2:
            raise RuntimeError("intentional failure")
        values = np.arange(len(context.request.target_times), dtype=float) - self.n
        return Forecast(
            context.request,
            pd.DataFrame(
                {
                    "target_time": context.request.target_times,
                    "prediction_mw": values,
                }
            ),
            model_id=self.model_id,
        )


def test_cached_failures_keep_population_and_common_support(tmp_path: Path) -> None:
    times = pd.date_range("2025-01-01T00:00Z", "2025-01-06T00:00Z", freq="15min")
    data = pd.DataFrame({"target_time": times, "pv_generation_mw": 100.0})
    spec = BacktestSpec(
        pd.date_range("2025-01-02T10:00Z", periods=2, freq="D"),
        availability_policy=EventTimeCutoff("30min"),
        on_error="continue",
    )
    frames = []
    for model in [FakeModel("a"), FakeModel("b", fail_second=True)]:
        timed = TimedForecaster(model)
        result = backtest(model=timed, data=data, spec=spec)
        calls = pd.DataFrame(timed.calls)
        save_cache(
            tmp_path,
            "identity",
            model.model_id,
            result.predictions,
            result.origins,
            calls,
            compute_summary(calls),
        )
        cached, origins, _, compute = load_cache(tmp_path, "identity", model.model_id)
        assert len(origins) == 2 and len(cached) == 192
        assert compute["successful_calls"] == (1 if model.fail_second else 2)
        frames.append(cached)
    evaluation = compare_models(frames, reference_model="a")
    assert evaluation.summary.n_scored.tolist() == [96, 96]
    assert evaluation.summary.forecast_availability.tolist() == [1.0, 0.5]
    daily = grouped_diagnostics(evaluation.aligned_predictions, "a", "local_date")
    assert len(daily) == 2  # One common day, two models; failures not filled.
    assert error_correlations(evaluation.aligned_predictions).shape == (2, 2)
    with pytest.raises(ValueError, match="identity"):
        load_cache(tmp_path, "different", "a")
    path = tmp_path / "predictions_a.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        load_cache(tmp_path, "identity", "a")


def test_snapshot_refuses_missing_or_changed_data_without_network(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("target_time,pv\na,1\n")
    shard = {"file": "data.csv", "sha256": sha256(csv), "start": "a", "end": "b"}
    write_json(csv.with_suffix(".json"), shard)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, [shard])
    assert verify_snapshot(tmp_path, manifest) == [shard]
    csv.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        verify_snapshot(tmp_path, manifest)


def test_daily_fractional_ties_and_rule_based_cases() -> None:
    daily = pd.DataFrame(
        [
            {"local_date": day, "model_id": model, "rmse_mw": value}
            for day, values in [("d1", [1, 1, 3]), ("d2", [3, 1, 2]), ("d3", [9, 8, 7])]
            for model, value in zip(["a", "b", "c"], values, strict=True)
        ]
    )
    detail, wins = daily_wins(daily, 1e-9)
    assert detail.winners.tolist() == ["a|b", "b", "c"]
    np.testing.assert_allclose(wins.fractional_days_won, [0.5, 1.5, 1.0])
    assert wins.percent_days_won.sum() == pytest.approx(100)
    cases = selected_cases(daily, 1e-9)
    assert cases.loc[cases.selection.eq("difficult"), "local_date"].tolist() == ["d3"]
    assert "a" not in cases.loc[cases.selection.eq("strongest_sole_win"), "model_id"].tolist()
    broken = daily.copy()
    broken.loc[0, "rmse_mw"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        daily_wins(broken, 1e-9)


def test_compute_summary_excludes_failures_from_success_mean() -> None:
    calls = pd.DataFrame(
        {
            "status": ["ok", "forecast_failed", "ok"],
            "predict_seconds": [1.0, 100.0, 3.0],
            "inference_seconds": [0.5, np.nan, 1.5],
        }
    )
    summary = compute_summary(calls)
    assert summary["attempted_calls"] == 3 and summary["successful_calls"] == 2
    assert summary["mean_predict_seconds"] == 2
    assert summary["mean_native_inference_seconds"] == 1
    assert summary["total_predict_seconds_all_attempts"] == 104
    assert compute_summary(calls.iloc[:0])["mean_predict_seconds"] is None
