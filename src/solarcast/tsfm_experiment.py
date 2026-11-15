"""Focused controlled-experiment validation, cache integrity and diagnostics."""

import hashlib
import json
import math
import time
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solarcast.backtesting import BacktestSpec, EventTimeCutoff
from solarcast.experiments import ExperimentConfig, daily_origins, sha256, write_json
from solarcast.models import Forecast, ForecastContext, Forecaster
from solarcast.models.chronos import CHECKPOINT as CHRONOS_CHECKPOINT
from solarcast.models.chronos import REVISION as CHRONOS_REVISION
from solarcast.models.t0 import CHECKPOINT as T0_CHECKPOINT
from solarcast.models.t0 import REVISION as T0_REVISION

MODEL_IDS = ["seasonal_persistence_48h", "lightgbm", "chronos", "t0"]


@dataclass(frozen=True)
class TSFMExperimentConfig:
    name: str
    parent_config: str
    parent_sha256: str
    snapshot_manifest: str
    snapshot_sha256: str
    output_dir: str
    models: list[str]
    context_slots: int
    point_policy: str
    dtype: str
    device: str
    torch_threads: int
    seed: int
    tie_atol_mw: float
    timing_origin_dates: list[str]
    normal_plot_days: list[str]
    difficult_day_rule: str
    all_struggle_rule: str
    win_example_rule: str
    context_selection: str
    zero_shot: str
    chronos: dict[str, str]
    t0: dict[str, str]
    packages: dict[str, str]

    @classmethod
    def load(cls, path: Path, root: Path) -> "TSFMExperimentConfig":
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        if set(raw) != {f.name for f in fields(cls)}:
            raise ValueError("TSFM configuration has missing or unknown keys")
        config = cls(**raw)
        config.validate(root)
        return config

    def parent(self, root: Path) -> ExperimentConfig:
        path = root / self.parent_config
        if sha256(path) != self.parent_sha256:
            raise ValueError("Frozen parent configuration hash mismatch")
        return ExperimentConfig.load(path)

    def validate(self, root: Path) -> None:
        parent = self.parent(root)
        if sha256(root / self.snapshot_manifest) != self.snapshot_sha256:
            raise ValueError("Frozen snapshot manifest hash mismatch")
        if (
            self.models != MODEL_IDS
            or not isinstance(self.context_slots, int)
            or self.context_slots != 1344
        ):
            raise ValueError("This primary experiment requires exactly four models and 1344 slots")
        if self.chronos != {"checkpoint": CHRONOS_CHECKPOINT, "revision": CHRONOS_REVISION}:
            raise ValueError("Chronos checkpoint differs from the approved selection")
        if self.t0 != {
            "checkpoint": T0_CHECKPOINT,
            "revision": T0_REVISION,
            "adjustments": "all off",
        }:
            raise ValueError("t0 checkpoint/adjustments differ from the approved selection")
        if (self.device, self.dtype, self.point_policy) != ("cpu", "float32", "native P50"):
            raise ValueError("This CPU experiment requires float32 native P50")
        if (
            isinstance(self.torch_threads, bool)
            or not isinstance(self.torch_threads, int)
            or self.torch_threads < 1
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or not math.isfinite(self.tie_atol_mw)
            or self.tie_atol_mw < 0
        ):
            raise ValueError("Invalid threads, seed or tie tolerance")
        if parent.context_window != "14D" or parent.reference_model != MODEL_IDS[0]:
            raise ValueError("Require the original 14D window and 48h reference")
        if not self.timing_origin_dates or len(set(self.timing_origin_dates)) != len(
            self.timing_origin_dates
        ):
            raise ValueError("Timing origins must be nonempty and unique")
        allowed = parent.origins("validation")
        for day in self.timing_origin_dates:
            if (
                daily_origins(
                    day, str(pd.Timestamp(day).date() + pd.Timedelta("1D")), parent.issue_clock
                )[0]
                not in allowed
            ):
                raise ValueError("Timing origins must belong to reserved validation, not test")
        if self.normal_plot_days != parent.representative_days:
            raise ValueError("Preserve predeclared representative plot dates")
        if (
            self.difficult_day_rule != "largest mean model daily RMSE"
            or self.all_struggle_rule != "largest minimum model daily RMSE"
            or self.win_example_rule
            != "largest second-best minus winning daily RMSE; sole wins only"
        ):
            raise ValueError("Unknown diagnostic selection rule")
        if not self.context_selection or not self.zero_shot or not self.packages:
            raise ValueError("Record context selection, zero-shot definition and runtime pins")
        for path in (self.output_dir, self.parent_config, self.snapshot_manifest):
            resolved = (root / path).resolve()
            if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
                raise ValueError("Experiment paths must stay within the repository")
        if self.output_dir == parent.output_dir:
            raise ValueError("Never overwrite the original benchmark")

    def timing_spec(self, root: Path) -> BacktestSpec:
        parent = self.parent(root)
        issues = pd.DatetimeIndex(
            [
                pd.Timestamp(f"{day}T{parent.issue_clock}", tz=parent.timezone)
                for day in sorted(self.timing_origin_dates)
            ]
        )
        return BacktestSpec(
            issues,
            availability_policy=EventTimeCutoff(parent.safety_lag),
            context_window=parent.context_window,
            on_error="continue",
        )


def cache_identity(value: object) -> str:
    """Canonical entire configuration/provenance/runtime identity, not just model name."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def verify_snapshot(root: Path, manifest: Path) -> list[dict[str, Any]]:
    """Refuse missing/corrupt existing shards; this experiment never retrieves data."""
    shards: list[dict[str, Any]] = json.loads(manifest.read_text(encoding="utf-8"))
    if not shards:
        raise ValueError("Snapshot manifest is empty")
    for shard in shards:
        path = (root / shard["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != shard["sha256"]:
            raise ValueError("Recorded snapshot CSV hash mismatch")
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if any(sidecar.get(key) != shard[key] for key in ("sha256", "start", "end")):
            raise ValueError("Recorded snapshot sidecar mismatch")
    return shards


def save_cache(
    output: Path,
    identity: str,
    model: str,
    predictions: pd.DataFrame,
    origins: pd.DataFrame,
    calls: pd.DataFrame,
    compute: Mapping[str, object],
) -> None:
    """Save each completed model immediately, retaining failed-origin rows."""
    paths = [output / f"{kind}_{model}.csv" for kind in ("predictions", "origins", "calls")]
    for path, frame in zip(paths, (predictions, origins, calls), strict=True):
        frame.to_csv(path, index=False)
    write_json(
        output / f"cache_{model}.json",
        {
            "identity": identity,
            "model_id": model,
            "compute": dict(compute),
            "files": {p.name: sha256(p) for p in paths},
        },
    )


def load_cache(
    output: Path, identity: str, model: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    meta = json.loads((output / f"cache_{model}.json").read_text(encoding="utf-8"))
    expected = {f"{kind}_{model}.csv" for kind in ("predictions", "origins", "calls")}
    if meta["identity"] != identity or meta["model_id"] != model or set(meta["files"]) != expected:
        raise ValueError("Prediction cache identity mismatch")
    if any(sha256(output / name) != digest for name, digest in meta["files"].items()):
        raise ValueError("Prediction cache file hash mismatch")
    predictions, origins, calls = [
        pd.read_csv(output / name)
        for name in (f"predictions_{model}.csv", f"origins_{model}.csv", f"calls_{model}.csv")
    ]
    for column in ("issue_time", "target_time"):
        predictions[column] = pd.to_datetime(predictions[column], utc=True)
    predictions["horizon"] = pd.to_timedelta(predictions.horizon)
    for column in ("prediction_mw", "actual_mw"):
        predictions[column] = predictions[column].astype("Float64")
    if not predictions.model_id.eq(model).all():
        raise ValueError("Prediction model identity mismatch")
    return predictions, origins, calls, meta["compute"]


class TimedForecaster:
    """Common predict-boundary wall timing, including feature/input/output translation."""

    def __init__(self, model: Forecaster) -> None:
        self.model = model
        self.calls: list[dict[str, object]] = []

    @property
    def model_id(self) -> str:
        return self.model.model_id

    def predict(self, context: ForecastContext) -> Forecast:
        start = time.perf_counter()
        status = "forecast_failed"
        try:
            result = self.model.predict(context)
            status = "ok"
            return result
        finally:
            row: dict[str, object] = {
                "model_id": self.model_id,
                "issue_time": context.request.issue_time,
                "status": status,
                "predict_seconds": time.perf_counter() - start,
                "history_rows": len(context.history),
            }
            info = getattr(self.model, "last_inference", None)
            if status == "ok" and info is not None:
                row.update(asdict(info))
            self.calls.append(row)
            if len(self.calls) % 25 == 0:
                print(f"{self.model_id}: {len(self.calls)} forecast calls completed", flush=True)


def compute_summary(calls: pd.DataFrame) -> dict[str, float | int | None]:
    if calls.empty:
        return {
            "attempted_calls": 0,
            "successful_calls": 0,
            "mean_predict_seconds": None,
            "mean_native_inference_seconds": None,
        }
    valid = calls.loc[calls.status.eq("ok")]
    return {
        "attempted_calls": len(calls),
        "successful_calls": len(valid),
        "mean_predict_seconds": float(valid.predict_seconds.mean()) if len(valid) else None,
        "total_predict_seconds_all_attempts": float(calls.predict_seconds.sum()),
        "mean_native_inference_seconds": (
            float(valid.inference_seconds.mean())
            if len(valid) and "inference_seconds" in valid
            else None
        ),
    }


def daily_wins(daily: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal day weights; absolute MW tolerance, fractional credit for ties."""
    matrix = daily.pivot(index="local_date", columns="model_id", values="rmse_mw")
    if matrix.empty or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Daily wins require finite common-support metrics for every model/day")
    winners = matrix.sub(matrix.min(axis=1), axis=0).le(tolerance)
    credit = winners.div(winners.sum(axis=1), axis=0)
    detail = pd.DataFrame(
        {
            "local_date": matrix.index,
            "winners": ["|".join(matrix.columns[row].tolist()) for row in winners.to_numpy()],
            "n_tied_winners": winners.sum(axis=1).to_numpy(),
        }
    )
    summary = pd.DataFrame(
        {
            "model_id": matrix.columns,
            "days_in_common": len(matrix),
            "sole_wins": (winners & winners.sum(axis=1).eq(1).to_numpy()[:, None]).sum().to_numpy(),
            "fractional_days_won": credit.sum().to_numpy(),
            "percent_days_won": credit.mean().to_numpy() * 100,
        }
    )
    return detail, summary


def selected_cases(daily: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    matrix = daily.pivot(index="local_date", columns="model_id", values="rmse_mw").sort_index()
    rows: list[dict[str, object]] = [
        {
            "selection": "difficult",
            "model_id": "all",
            "local_date": matrix.mean(axis=1).idxmax(),
            "criterion_mw": float(matrix.mean(axis=1).max()),
        },
        {
            "selection": "all_struggle",
            "model_id": "all",
            "local_date": matrix.min(axis=1).idxmax(),
            "criterion_mw": float(matrix.min(axis=1).max()),
        },
    ]
    for model in matrix.columns:
        margin = matrix.drop(columns=model).min(axis=1) - matrix[model]
        eligible = margin[margin > tolerance]
        if len(eligible):
            rows.append(
                {
                    "selection": "strongest_sole_win",
                    "model_id": model,
                    "local_date": eligible.idxmax(),
                    "criterion_mw": float(eligible.max()),
                }
            )
    return pd.DataFrame(rows)


def error_correlations(aligned: pd.DataFrame) -> pd.DataFrame:
    errors = aligned.assign(
        error_mw=aligned.prediction_mw.astype(float) - aligned.actual_mw.astype(float)
    )
    return errors.pivot(
        index=["issue_time", "target_time"], columns="model_id", values="error_mw"
    ).corr()
