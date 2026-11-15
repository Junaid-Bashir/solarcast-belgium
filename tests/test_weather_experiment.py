"""Offline frozen-plan, local-covariate and reused-prediction cache integrity."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from solarcast.backtesting import day_ahead_local_day
from solarcast.data.weather import FEATURES, WeatherError
from solarcast.experiments import sha256, write_json
from solarcast.tsfm_experiment import TSFMExperimentConfig, save_cache
from solarcast.weather_experiment import (
    PreparedWeather,
    WeatherExperimentConfig,
    cached_comparators,
    choose_weather,
    feature_entry,
    plan,
)


def test_frozen_config_and_plan_reject_methodology_or_parent_drift(
    experiment: tuple[Path, WeatherExperimentConfig],
) -> None:
    root, config = experiment
    initial = plan(config, root)
    assert initial == plan(config, root)
    updates: tuple[dict[str, Any], ...] = (
        {"classification": "untouched holdout"},
        {"nwp_policy": "zero delay"},
        {"radiation_policy": "clip everything"},
        {"weather_features": []},
        {"primary_tsfm": "different"},
        {"workers": 100},
        {"output_dir": "../outside"},
        {"selection_rule": "lowest test RMSE"},
    )
    for update in updates:
        with pytest.raises(ValueError):
            replace(config, **update).validate(root)
    parent = root / config.parent_config
    parent.write_text(parent.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="parent hash"):
        plan(config, root)


def test_validation_selection_excludes_history_and_uses_only_predeclared_variants() -> None:
    frame = pd.DataFrame(
        {
            "feature_set": [
                "history",
                "history+radiation",
                "history+radiation+cloud",
                "history+nwp",
            ],
            "rmse_mw": [1.0, 3.0, 2.0, 2.0],
            "n_scored": 96,
        }
    )
    assert choose_weather(frame) == "history+radiation+cloud"
    frame.loc[1:, "rmse_mw"] = np.nan
    with pytest.raises(ValueError, match="validation"):
        choose_weather(frame)


def test_local_weather_requires_plan_checksum_exact_targets_and_declared_policy(
    experiment: tuple[Path, WeatherExperimentConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = experiment
    provider = PreparedWeather(root, config)
    issue = pd.Timestamp("2025-03-01T10Z")
    targets = day_ahead_local_day(issue)
    with pytest.raises(WeatherError, match="not been prepared"):
        provider.for_issue(issue, targets)
    entry = feature_entry(root, config, issue)
    entry.mkdir(parents=True)
    binary = entry / "features.parquet"
    binary.write_bytes(b"offline payload; parquet engine mocked at this boundary")
    frame = pd.DataFrame(
        {"target_time": targets, **{name: np.ones(len(targets)) for name in FEATURES}}
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: frame.copy())
    status: dict[str, Any] = {
        "plan_identity": plan(config, root)["identity"],
        "issue_time": issue.isoformat(),
        "reference_time": "2025-03-01T00Z",
        "available_at": "2025-03-01T06Z",
        "status": "ok",
        "availability_basis": "assumed",
        "policy_name": config.nwp_policy,
        "run_version": "fixture",
        "weights_id": "fixture",
        "radiation_policy": config.radiation_policy,
        "radiation_adjusted_cell_hours": 0,
        "features_sha256": sha256(binary),
    }
    write_json(entry / "status.json", status)
    selected = provider.for_issue(issue, targets)
    assert selected.reference_time.hour == 0 and len(selected.features) == 96
    with pytest.raises(WeatherError, match="exactly"):
        provider.for_issue(issue, targets[:2])
    altered = dict(status, policy_name="unapproved")
    write_json(entry / "status.json", altered)
    with pytest.raises(WeatherError, match="methodology"):
        provider.for_issue(issue, targets)
    write_json(entry / "status.json", status)
    binary.write_bytes(b"corrupted")
    with pytest.raises(WeatherError, match="checksum"):
        provider.for_issue(issue, targets)


def test_original_comparators_require_exact_schedule_and_checkpoint_metadata(
    experiment: tuple[Path, WeatherExperimentConfig],
) -> None:
    root, config = experiment
    output = root / config.tsfm_output_dir
    freeze = json.loads((output / "freeze.json").read_text())
    parent = config.parent(root)
    keys = [
        (issue, target) for issue in parent.origins("test") for target in day_ahead_local_day(issue)
    ]
    frame = pd.DataFrame(keys, columns=["issue_time", "target_time"])
    frame["horizon"] = frame.target_time - frame.issue_time
    frame["prediction_mw"], frame["actual_mw"], frame["status"] = 1.0, 1.0, "ok"
    origins = pd.DataFrame({"issue_time": parent.origins("test"), "status": "ok"})
    calls = pd.DataFrame({"issue_time": parent.origins("test"), "predict_seconds": 0.1})
    prior = TSFMExperimentConfig.load(root / config.tsfm_config, root)
    for model in (parent.reference_model, "lightgbm", "t0", "chronos"):
        compute = (
            {**(prior.t0 if model == "t0" else prior.chronos), "context_cap_slots": 1344}
            if model in ("t0", "chronos")
            else {}
        )
        save_cache(
            output, freeze["identity"], model, frame.assign(model_id=model), origins, calls, compute
        )
    assert len(cached_comparators(config, root)) == 4
    path = output / "cache_t0.json"
    original = json.loads(path.read_text())
    original["compute"]["context_cap_slots"] = 100
    write_json(path, original)
    with pytest.raises(ValueError, match="checkpoint/context"):
        cached_comparators(config, root)
    original["compute"]["context_cap_slots"] = 1344
    write_json(path, original)
    prediction_path = output / "predictions_t0.csv"
    altered = pd.read_csv(prediction_path).iloc[1:]
    altered.to_csv(prediction_path, index=False)
    original["files"][prediction_path.name] = sha256(prediction_path)
    write_json(path, original)
    with pytest.raises(ValueError, match="schedule"):
        cached_comparators(config, root)
