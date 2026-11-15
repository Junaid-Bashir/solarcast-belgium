"""Exercise common-support weather diagnostics end to end on offline fake forecasts."""

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from solarcast.backtesting import day_ahead_local_day
from solarcast.data.weather import FEATURES
from solarcast.data.weather.types import WeatherCovariates
from solarcast.experiments import sha256, write_json
from solarcast.weather_experiment import WeatherExperimentConfig


def runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "weather_benchmark_runner", Path("scripts/benchmark_weather_lightgbm.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_support_diagnostics_preserve_negatives_and_use_forecast_only_regimes(
    experiment: tuple[Path, WeatherExperimentConfig],
    monkeypatch: Any,
) -> None:
    root, original = experiment
    parent_path = root / original.parent_config
    text = parent_path.read_text().replace('test_end = "2025-10-31"', 'test_end = "2025-03-02"')
    text = text.replace(
        'representative_days = ["2025-03-15", "2025-06-15", "2025-09-15"]',
        'representative_days = ["2025-03-01", "2025-03-02"]',
    )
    parent_path.write_text(text)
    config = replace(original, parent_sha256=sha256(parent_path))
    parent = config.parent(root)
    keys = [
        (issue, target) for issue in parent.origins("test") for target in day_ahead_local_day(issue)
    ]
    frame = pd.DataFrame(keys, columns=["issue_time", "target_time"])
    phase = np.arange(len(frame)) * np.pi / 48
    frame["actual_mw"] = np.maximum(0, np.sin(phase)) * 1000
    frame["horizon"] = frame.target_time - frame.issue_time
    frame["status"] = "ok"
    weather_id = "lightgbm_history_nwp"
    frames = [
        frame.assign(model_id=name, prediction_mw=frame.actual_mw + scale * np.cos(phase))
        for name, scale in (
            (config.reference_model, 150),
            ("lightgbm", 100),
            (weather_id, 50),
            ("t0", 80),
            ("chronos", 90),
        )
    ]
    output = root / config.output_dir
    output.mkdir(parents=True)
    write_json(output / "freeze.json", {"identity": "offline-fixture"})
    write_json(output / "preparation.json", {"wall_seconds": 2.0})
    for name in (config.reference_model, "t0", "chronos"):
        write_json(output / f"reused_{name}.json", {"model_id": name, "load_seconds": 0.1})
    module = runner()
    monkeypatch.setattr(module, "load_frames", lambda *args: (frames, weather_id))
    monkeypatch.setattr(
        module, "load_cache", lambda *args: (None, None, None, {"load_seconds": 0.0})
    )
    inventory = pd.DataFrame(
        {
            "status": ["ok", "ok"],
            "transferred_bytes": [100, 100],
            "native_cache_bytes": [50, 50],
            "elapsed_seconds": [1.0, 1.0],
        }
    )
    monkeypatch.setattr(module, "weather_inventory", lambda *args: inventory.copy())

    class ForecastWeather:
        def __init__(self, *args: Any) -> None:
            pass

        def for_issue(self, issue: pd.Timestamp, targets: pd.DatetimeIndex) -> WeatherCovariates:
            assert targets.unit == "ns"  # Same canonical boundary as ForecastRequest.
            weather = pd.DataFrame(
                {"target_time": targets, **{name: np.ones(len(targets)) for name in FEATURES}}
            )
            weather[FEATURES[4]] = np.linspace(0.1, 0.9, len(targets))
            weather[FEATURES[5]] = np.linspace(0.05, 0.25, len(targets))
            return WeatherCovariates(
                weather,
                issue_time=issue,
                reference_time=issue.floor("D"),
                available_at=issue.floor("D") + pd.Timedelta(hours=6),
                availability_basis="assumed",
                policy_name=config.nwp_policy,
                run_version="synthetic",
                weights_id="synthetic",
            )

    monkeypatch.setattr(module, "PreparedWeather", ForecastWeather)
    plotted = []
    monkeypatch.setattr(module, "plots", lambda *args: plotted.append(args))
    module.evaluate(config, root)
    leaderboard = pd.read_csv(output / "leaderboard.csv")
    assert leaderboard.model_id.iloc[0] == weather_id
    assert leaderboard.n_scored.eq(192).all()
    assert pd.read_csv(output / "aligned_predictions.csv").prediction_mw.min() < 0
    regimes = pd.read_csv(output / "weather_regimes.csv")
    assert set(regimes.regime) == {
        "low_cloud",
        "medium_cloud",
        "high_cloud",
        "spatially_uniform",
        "spatially_variable",
    }
    assert set(regimes.daylight_only) == {False, True}
    assert pd.read_csv(output / "daily_win_summary.csv").days_in_common.eq(2).all()
    assert len(pd.read_csv(output / "information_value.csv")) == 3
    assert (output / "results_manifest.json").exists() and len(plotted) == 1


def test_analysis_amendment_cannot_allow_changed_model_or_configuration_sources(
    tmp_path: Path,
) -> None:
    module = runner()
    frozen = {
        "scripts/benchmark_weather_lightgbm.py": "old",
        "src/solarcast/models/lightgbm.py": "same",
    }
    current = dict(frozen, **{"scripts/benchmark_weather_lightgbm.py": "reporting-only-fix"})
    module.check_analysis_sources(frozen, frozen, tmp_path, "original-inference")
    with pytest.raises(ValueError, match="authenticated"):
        module.check_analysis_sources(frozen, current, tmp_path, "original-inference")
    archive = tmp_path / "analysis_source_snapshot.zip"
    archive.write_bytes(b"offline recorded source archive")
    write_json(
        tmp_path / "analysis_amendment.json",
        {
            "inference_identity": "original-inference",
            "original_sources": frozen,
            "analysis_sources": current,
            "refit_or_reinference": False,
            "analysis_archive_sha256": sha256(archive),
            "prediction_files": {},
        },
    )
    module.check_analysis_sources(frozen, current, tmp_path, "original-inference")
    altered = dict(current, **{"src/solarcast/models/lightgbm.py": "changed"})
    with pytest.raises(ValueError, match="sources changed"):
        module.check_analysis_sources(frozen, altered, tmp_path, "original-inference")
