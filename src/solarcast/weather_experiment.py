"""Frozen weather-LightGBM development experiment, local covariates and cache checks."""

import json
import time
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import pandas as pd

from solarcast.backtesting import day_ahead_local_day
from solarcast.data.weather import (
    FEATURES,
    GFSClient,
    NWPAvailability,
    WeatherCache,
    WeatherCovariates,
    WeatherError,
    belgium_weights,
    select_nwp_vintage,
)
from solarcast.data.weather.cache import digest
from solarcast.data.weather.types import PRODUCT
from solarcast.experiments import ExperimentConfig, sha256, write_json
from solarcast.models.lightgbm_data import FeatureSet, feature_names
from solarcast.tsfm_experiment import TSFMExperimentConfig, cache_identity, load_cache


@dataclass(frozen=True)
class WeatherExperimentConfig:
    name: str
    classification: str
    parent_config: str
    parent_sha256: str
    tsfm_config: str
    tsfm_config_sha256: str
    tsfm_output_dir: str
    tsfm_freeze_sha256: str
    tsfm_results_sha256: str
    output_dir: str
    weather_cache_dir: str
    weather_feature_dir: str
    workers: int
    request_interval: float
    nwp_policy: str
    nwp_product: str
    radiation_policy: str
    weather_features: list[str]
    ablations: list[str]
    selection_rule: str
    regime_cloud_thresholds: list[float]
    variability_threshold: float
    representative_rule: str
    difficult_rule: str
    reference_model: str
    primary_tsfm: str
    ranking_metric: str

    @classmethod
    def load(cls, path: Path, root: Path) -> "WeatherExperimentConfig":
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        if set(raw) != {f.name for f in fields(cls)}:
            raise ValueError("Weather experiment has missing/unknown configuration keys")
        config = cls(**raw)
        config.validate(root)
        return config

    def parent(self, root: Path) -> ExperimentConfig:
        if sha256(root / self.parent_config) != self.parent_sha256:
            raise ValueError("Weather experiment parent hash mismatch")
        return ExperimentConfig.load(root / self.parent_config)

    def validate(self, root: Path) -> None:
        parent = self.parent(root)
        for relative in (
            self.parent_config,
            self.tsfm_config,
            self.tsfm_output_dir,
            self.output_dir,
            self.weather_cache_dir,
            self.weather_feature_dir,
        ):
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
                raise ValueError("Weather experiment paths must stay inside repository")
        if (
            self.classification != "development: prior test results known"
            or self.nwp_policy != NWPAvailability().name
            or self.nwp_product != PRODUCT
            or self.radiation_policy != "grib_packing_bound_v1"
            or self.weather_features != list(FEATURES)
            or self.workers not in (1, 2, 4, 8)
            or not 0.3 <= self.request_interval <= 3
            or self.reference_model != parent.reference_model
            or self.primary_tsfm != "t0"
            or self.ranking_metric != "rmse"
            or self.selection_rule
            != "lowest validation common-support RMSE; fewer features on exact tie"
            or self.regime_cloud_thresholds != [0.25, 0.75]
            or self.variability_threshold != 0.15
            or self.representative_rule != "15 March, 15 June, 15 September; no outcome selection"
            or self.difficult_rule
            != "largest daylight mean forecast cloud spatial standard deviation"
        ):
            raise ValueError("Unsupported frozen weather methodology")
        expected = ["history", "history+radiation", "history+radiation+cloud", "history+nwp"]
        if self.ablations != expected:
            raise ValueError("Require the four predeclared feature ablations")
        tsfm = root / self.tsfm_output_dir
        for path, expected_hash in (
            (root / self.tsfm_config, self.tsfm_config_sha256),
            (tsfm / "freeze.json", self.tsfm_freeze_sha256),
            (tsfm / "results_manifest.json", self.tsfm_results_sha256),
        ):
            if sha256(path) != expected_hash:
                raise ValueError("Pinned prior TSFM experiment hash mismatch")
        prior = TSFMExperimentConfig.load(root / self.tsfm_config, root)
        if prior.parent_sha256 != self.parent_sha256:
            raise ValueError("TSFM schedule/history configuration differs")
        if len({self.output_dir, parent.output_dir, self.tsfm_output_dir}) != 3:
            raise ValueError("Never overwrite previous experiment artifacts")


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf8")))


def plan(config: WeatherExperimentConfig, root: Path) -> dict[str, Any]:
    """Freeze methodology before acquisition/validation, independently of source edits."""
    payload = {"config": asdict(config), "expanded_parent": asdict(config.parent(root))}
    return {"identity": cache_identity(payload), "payload": payload}


def feature_entry(root: Path, config: WeatherExperimentConfig, issue: pd.Timestamp) -> Path:
    return (
        root
        / config.weather_feature_dir
        / str(plan(config, root)["identity"])
        / issue.strftime("%Y%m%dT%H%MZ")
    )


def prepare_origin(
    root_text: str, config: WeatherExperimentConfig, issue_text: str
) -> dict[str, Any]:
    """Process-safe bounded acquisition followed by eligible-vintage selection.

    Persist success or failure for every planned origin. Failures are never silently
    retried into a different experimental population when resuming the same plan.
    """
    root, issue = Path(root_text), pd.Timestamp(issue_text)
    entry = feature_entry(root, config, issue)
    manifest = entry / "status.json"
    if manifest.exists():
        status = read_json(manifest)
        if (
            status["issue_time"] != issue.isoformat()
            or status["plan_identity"] != plan(config, root)["identity"]
        ):
            raise ValueError("Prepared weather issue/plan identity mismatch")
        if (
            status["status"] == "ok"
            and sha256(entry / "features.parquet") != status["features_sha256"]
        ):
            raise ValueError("Prepared weather checksum mismatch")
        return status
    if entry.exists():
        raise ValueError(f"Incomplete prepared weather entry: {entry}")
    entry.mkdir(parents=True)
    targets = day_ahead_local_day(issue)
    reference = (issue - pd.Timedelta(hours=6)).floor("6h")
    first = int((targets[0].floor("h") - reference) / pd.Timedelta(hours=1))
    last = int(
        ((targets[-1] + pd.Timedelta(minutes=15)).ceil("h") - reference) / pd.Timedelta(hours=1)
    )
    start = time.perf_counter()
    status = {
        "issue_time": issue.isoformat(),
        "reference_time": reference.isoformat(),
        "plan_identity": plan(config, root)["identity"],
        "status": "failed",
        "n_targets": len(targets),
        "transferred_bytes": 0,
    }
    try:
        cache = WeatherCache(root / config.weather_cache_dir)
        with GFSClient(cache=cache, request_interval=config.request_interval) as client:
            run = client.fetch_forecast_run(reference, steps=list(range(first, last + 1)))
        status["transferred_bytes"] = run.metadata["transferred_bytes"]
        status["run_version"] = run.version
        status["native_identity"] = run.metadata["identity"]
        status["native_cache_bytes"] = sum(
            p.stat().st_size
            for p in (root / config.weather_cache_dir / digest(run.metadata["identity"])).iterdir()
        )
        retrieval_seconds = time.perf_counter() - start
        weather = select_nwp_vintage(
            [run],
            issue,
            targets,
            weights=belgium_weights(),
            radiation_policy=config.radiation_policy,
        )
        weather.features.to_parquet(entry / "features.parquet", index=False, engine="pyarrow")
        status.update(
            {
                "status": "ok",
                "available_at": weather.available_at.isoformat(),
                "availability_basis": weather.availability_basis,
                "policy_name": weather.policy_name,
                "weights_id": weather.weights_id,
                "features_sha256": sha256(entry / "features.parquet"),
                "retrieval_seconds": retrieval_seconds,
                "preprocessing_seconds": time.perf_counter() - start - retrieval_seconds,
                "radiation_policy": weather.radiation_policy,
                "radiation_adjusted_cell_hours": weather.radiation_adjusted_cell_hours,
            }
        )
    except WeatherError as error:
        status.update({"error_type": type(error).__name__, "error_message": str(error)})
    status["elapsed_seconds"] = time.perf_counter() - start
    pending = manifest.with_suffix(".partial")
    write_json(pending, status)
    pending.replace(manifest)
    return status


class PreparedWeather:
    """Experiment-only local feature cache; never performs network access at fit/predict."""

    def __init__(self, root: Path, config: WeatherExperimentConfig) -> None:
        self.root, self.config = root, config

    def for_issue(self, issue_time: pd.Timestamp, targets: pd.DatetimeIndex) -> WeatherCovariates:
        entry = feature_entry(self.root, self.config, issue_time)
        if not (entry / "status.json").exists():
            raise WeatherError("NWP origin has not been prepared")
        status = read_json(entry / "status.json")
        if (
            status["issue_time"] != issue_time.isoformat()
            or status["plan_identity"] != plan(self.config, self.root)["identity"]
        ):
            raise WeatherError("NWP prepared issue/plan identity mismatch")
        if status["status"] != "ok":
            raise WeatherError("NWP preparation failed: " + status.get("error_message", "unknown"))
        if (
            status["policy_name"] != self.config.nwp_policy
            or status["availability_basis"] != "assumed"
            or status["radiation_policy"] != self.config.radiation_policy
            or pd.Timestamp(status["reference_time"])
            != (issue_time - pd.Timedelta(hours=6)).floor("6h")
        ):
            raise WeatherError("Prepared NWP policy/reference differs from frozen methodology")
        if sha256(entry / "features.parquet") != status["features_sha256"]:
            raise WeatherError("Prepared weather checksum mismatch")
        frame = pd.read_parquet(entry / "features.parquet", engine="pyarrow")
        if not pd.DatetimeIndex(frame.target_time).equals(targets):
            raise WeatherError("Prepared weather targets do not match the request exactly")
        return WeatherCovariates(
            frame,
            issue_time=issue_time,
            reference_time=status["reference_time"],
            available_at=status["available_at"],
            availability_basis=status["availability_basis"],
            policy_name=status["policy_name"],
            run_version=status["run_version"],
            weights_id=status["weights_id"],
            radiation_policy=status["radiation_policy"],
            radiation_adjusted_cell_hours=status["radiation_adjusted_cell_hours"],
        )


def cached_comparators(
    config: WeatherExperimentConfig, root: Path
) -> list[tuple[pd.DataFrame, dict[str, Any]]]:
    """Reuse original immutable outputs, not old-source guards against current source.

    Authenticate original freeze/config/artifacts, exact checkpoints and full UTC
    schedule. Current source is explicitly different and is frozen separately.
    """
    output = root / config.tsfm_output_dir
    freeze = read_json(output / "freeze.json")
    prior = TSFMExperimentConfig.load(root / config.tsfm_config, root)
    if (
        freeze["identity"] != cache_identity(freeze["identity_payload"])
        or freeze["identity_payload"]["configuration"] != asdict(prior)
        or freeze["identity_payload"]["stage"] != "test"
    ):
        raise ValueError("Original TSFM freeze identity mismatch")
    manifest = read_json(output / "results_manifest.json")
    if manifest["identity"] != freeze["identity"] or any(
        sha256(output / name) != value for name, value in manifest["files"].items()
    ):
        raise ValueError("Original TSFM artifacts differ from recorded results")
    parent = config.parent(root)
    expected = pd.MultiIndex.from_tuples(
        [
            (issue, target)
            for issue in parent.origins("test")
            for target in day_ahead_local_day(issue)
        ],
        names=["issue_time", "target_time"],
    )
    result = []
    for model in (parent.reference_model, "lightgbm", "t0", "chronos"):
        predictions, origins, _, compute = load_cache(output, freeze["identity"], model)
        if not pd.MultiIndex.from_frame(predictions[["issue_time", "target_time"]]).equals(
            expected
        ):
            raise ValueError("Cached comparator schedule differs from weather experiment")
        if not pd.DatetimeIndex(pd.to_datetime(origins.issue_time, utc=True)).equals(
            parent.origins("test")
        ):
            raise ValueError("Cached comparator origin population mismatch")
        if model in ("t0", "chronos"):
            checkpoint = prior.t0 if model == "t0" else prior.chronos
            if (
                compute.get("checkpoint") != checkpoint["checkpoint"]
                or compute.get("revision") != checkpoint["revision"]
                or compute.get("context_cap_slots") != prior.context_slots
            ):
                raise ValueError("Cached TSFM checkpoint/context metadata mismatch")
        result.append(
            (
                predictions,
                {
                    "model_id": model,
                    **compute,
                    "timing_source": "tsfm_zero_shot_v1 cached original CPU run",
                },
            )
        )
    return result


def choose_weather(ablation: pd.DataFrame) -> FeatureSet:
    """Validation only; no inference from a test leaderboard is accepted."""
    weather = ablation.loc[ablation.feature_set.ne("history")].copy()
    if weather.empty or weather.rmse_mw.isna().any() or weather.n_scored.le(0).any():
        raise ValueError("No complete common-support validation ablation")
    weather["feature_count"] = [
        len(feature_names(cast(FeatureSet, name))) for name in weather.feature_set
    ]
    return cast(FeatureSet, weather.sort_values(["rmse_mw", "feature_count"]).feature_set.iloc[0])
