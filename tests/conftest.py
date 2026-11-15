"""Offline frozen-experiment fixture; no downloaded artifacts or network required."""

import shutil
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from solarcast.experiments import sha256, write_json
from solarcast.tsfm_experiment import TSFMExperimentConfig, cache_identity
from solarcast.weather_experiment import WeatherExperimentConfig


@pytest.fixture
def experiment(tmp_path: Path) -> tuple[Path, WeatherExperimentConfig]:
    root = Path.cwd()
    (tmp_path / "configs/experiments").mkdir(parents=True)
    parent_path = "configs/experiments/lightgbm_v1.toml"
    shutil.copyfile(root / parent_path, tmp_path / parent_path)
    snapshot = tmp_path / "artifacts/experiments/lightgbm_baseline_v1/snapshot.json"
    snapshot.parent.mkdir(parents=True)
    write_json(snapshot, [])
    source = root / "configs/experiments/tsfm_zero_shot_v1.toml"
    text = source.read_text().replace(
        "98e0cd1f701a60a51f405d1bc6ff415952c9513c28a27fc38b75dca6cc3156e7", sha256(snapshot)
    )
    tsfm_path = tmp_path / "configs/experiments/tsfm_zero_shot_v1.toml"
    tsfm_path.write_text(text)
    prior = TSFMExperimentConfig.load(tsfm_path, tmp_path)
    output = tmp_path / prior.output_dir
    output.mkdir(parents=True)
    payload = {"configuration": asdict(prior), "stage": "test"}
    freeze = {"identity": cache_identity(payload), "identity_payload": payload}
    write_json(output / "freeze.json", freeze)
    write_json(
        output / "results_manifest.json",
        {"identity": freeze["identity"], "files": {"freeze.json": sha256(output / "freeze.json")}},
    )
    with (root / "configs/experiments/weather_lightgbm_v1.toml").open("rb") as stream:
        raw = tomllib.load(stream)
    config = replace(
        WeatherExperimentConfig(**raw),
        tsfm_config_sha256=sha256(tsfm_path),
        tsfm_freeze_sha256=sha256(output / "freeze.json"),
        tsfm_results_sha256=sha256(output / "results_manifest.json"),
    )
    config.validate(tmp_path)
    return tmp_path, config
