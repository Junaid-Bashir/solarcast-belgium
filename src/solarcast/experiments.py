"""Small real-experiment utilities; no network or model fitting on import."""

import hashlib
import json
import logging
import tomllib
from dataclasses import dataclass, fields
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, _prepare_data, day_ahead_local_day
from solarcast.data import EliaPVClient
from solarcast.data._schema import CURRENT, OLD, TRANSITION, utc_timestamp
from solarcast.data.validation import validate_history
from solarcast.evaluation import compare_models
from solarcast.features import solar_features
from solarcast.models import SeasonalPersistenceForecaster
from solarcast.models.lightgbm import LightGBMConfig

logger = logging.getLogger(__name__)


def daily_origins(start: str, end: str, clock: str) -> pd.DatetimeIndex:
    """Brussels origin dates [start,end), rejecting ambiguous/nonexistent local clocks."""
    first, stop = date.fromisoformat(start), date.fromisoformat(end)
    local_clock = time.fromisoformat(clock)
    if first >= stop or local_clock.tzinfo is not None:
        raise ValueError("Require increasing origin dates and offset-free issue_clock")
    return (
        pd.DatetimeIndex(
            [
                datetime.combine(first + timedelta(days=i), local_clock)
                for i in range((stop - first).days)
            ]
        )
        .tz_localize("Europe/Brussels", ambiguous="raise", nonexistent="raise")
        .tz_convert("UTC")
    )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data_start: str
    data_end: str
    cache_dir: str
    output_dir: str
    train_start: str
    train_end: str
    fit_time: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    issue_clock: str
    timezone: str
    target_policy: str
    context_window: str
    safety_lag: str
    baseline_lags: list[str]
    comparison_lags: list[str]
    include_last_value: bool
    reference_model: str
    ranking_metric: str
    evaluation_subsets: list[str]
    representative_days: list[str]
    model: LightGBMConfig

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        if set(raw) != {field.name for field in fields(cls)}:
            raise ValueError("Configuration has missing or unknown keys")
        config = cls(**{**raw, "model": LightGBMConfig(**raw["model"])})
        config.validate()
        return config

    def origins(self, split: str) -> pd.DatetimeIndex:
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        return daily_origins(
            getattr(self, f"{split}_start"), getattr(self, f"{split}_end"), self.issue_clock
        )

    def spec(self, split: str) -> BacktestSpec:
        return BacktestSpec(
            self.origins(split),
            availability_policy=EventTimeCutoff(self.safety_lag),
            context_window=self.context_window,
            on_error="continue",
        )

    def validate(self) -> None:
        if (
            self.timezone != "Europe/Brussels"
            or self.target_policy != "next_local_calendar_day"
            or self.ranking_metric != "rmse"
            or self.evaluation_subsets != ["all_intervals", "reference_point_daylight"]
        ):
            raise ValueError(
                "This experiment requires Brussels next-day targets, RMSE and both subsets"
            )
        first, stop = (
            utc_timestamp(self.data_start, name="data_start"),
            utc_timestamp(self.data_end, name="data_end"),
        )
        if first >= stop or any(t != t.floor("15min") for t in (first, stop)):
            raise ValueError("Data bounds must increase and align to quarter hours")
        train, validation, test = (self.origins(s) for s in ("train", "validation", "test"))
        fit = utc_timestamp(self.fit_time, name="fit_time")
        policy = EventTimeCutoff(self.safety_lag)
        label_cutoff = (fit - policy.safety_lag).floor("15min")
        if not (train[-1] < fit <= validation[0] < test[0]):
            raise ValueError("Require train < fit <= validation < test")
        if day_ahead_local_day(train[-1])[-1] + pd.Timedelta("15min") > label_cutoff:
            raise ValueError("Training labels cross fit cutoff")
        if day_ahead_local_day(validation[-1])[-1] + pd.Timedelta("15min") > test[0]:
            raise ValueError("Validation labels overlap test origins")
        window = self.spec("train").context_window
        assert window is not None
        if first > train[0] - window or stop < day_ahead_local_day(test[-1])[-1] + pd.Timedelta(
            "15min"
        ):
            raise ValueError("Data period does not cover context and test targets")
        if not self.comparison_lags or not set(self.comparison_lags) <= set(self.baseline_lags):
            raise ValueError("Comparison lags must be a nonempty subset of baseline lags")
        identities = [SeasonalPersistenceForecaster(lag).model_id for lag in self.baseline_lags]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate baseline identities")
        compared = [SeasonalPersistenceForecaster(lag).model_id for lag in self.comparison_lags]
        if self.reference_model not in compared:
            raise ValueError("Reference must be an included comparison baseline")
        for day in self.representative_days:
            target = pd.Timestamp(day, tz=self.timezone).tz_convert("UTC")
            if not day_ahead_local_day(test[0])[0] <= target <= day_ahead_local_day(test[-1])[-1]:
                raise ValueError("Representative day outside test targets")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )


def quality_summary(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    """Validate without filling; distinguish absent rows from explicit null targets."""
    prepared = _prepare_data(data)
    validate_history(prepared, start, end)
    expected = pd.date_range(start, end, freq="15min", inclusive="left")
    times = pd.DatetimeIndex(prepared.target_time)
    return {
        "start_inclusive": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "first_observation": str(times.min()),
        "last_observation": str(times.max()),
        "expected_intervals": len(expected),
        "interval_rows": len(data),
        "missing_interval_rows": len(expected.difference(times)),
        "null_pv_observations": int(prepared.pv_generation_mw.isna().sum()),
        "duplicate_timestamps": int(data.target_time.duplicated().sum()),
        "misaligned_timestamps": int((times != times.floor("15min")).sum()),
        "negative_pv_observations": int((prepared.pv_generation_mw < 0).sum()),
        "source_counts": prepared.source_dataset.value_counts().to_dict(),
        "source_target_fields": sorted(prepared.source_target_field.unique().tolist()),
        "crosses_schema_transition": bool(start < TRANSITION < end),
    }


def load_snapshot(
    config: ExperimentConfig, root: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Monthly resumable immutable cache, hash-checked before use; no synthetic fallback.

    Each normalized CSV/sidecar pair is promoted only after a successful fetch and
    validation. Existing valid shards are never refreshed automatically. Partial
    or corrupt pairs raise; they never overwrite a previously valid snapshot.
    """
    first, stop = (
        pd.Timestamp(config.data_start).tz_convert("UTC"),
        pd.Timestamp(config.data_end).tz_convert("UTC"),
    )
    cache = root / config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    boundaries = [first]
    boundary = (first.tz_localize(None).to_period("M") + 1).start_time.tz_localize("UTC")
    while boundary < stop:
        boundaries.append(boundary)
        boundary = boundary + pd.offsets.MonthBegin()
    boundaries.append(stop)
    frames, metadata = [], []
    with EliaPVClient() as client:
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
            stem = f"{lower.strftime('%Y%m%dT%H%MZ')}_{upper.strftime('%Y%m%dT%H%MZ')}"
            csv, sidecar = cache / f"{stem}.csv", cache / f"{stem}.json"
            if csv.exists() != sidecar.exists():
                raise ValueError(f"Incomplete cache pair: {csv}; inspect before retrying")
            if csv.exists():
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if (
                    meta.get("sha256") != sha256(csv)
                    or meta.get("start") != str(lower)
                    or meta.get("end") != str(upper)
                ):
                    raise ValueError(f"Cache hash/bounds mismatch: {csv}")
                frame = pd.read_csv(csv)
                for column in ("target_time", "retrieved_at"):
                    stamps = [pd.Timestamp(v) for v in frame[column]]
                    if any(pd.isna(v) or v.tzinfo is None for v in stamps):
                        raise ValueError(f"Cache needs explicit UTC offsets: {csv}")
                    frame[column] = pd.to_datetime(stamps, utc=True).astype("datetime64[ns, UTC]")
                for column in ("pv_generation_mw", "monitored_capacity_mw"):
                    frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Float64")
                quality_summary(frame, lower, upper)
                logger.info("Using snapshot %s", csv)
            else:
                logger.info("Retrieving real Elia [%s, %s)", lower, upper)
                frame = client.fetch_history(start=lower, end=upper)
                quality = quality_summary(frame, lower, upper)
                pending = csv.with_suffix(".csv.partial")
                frame.to_csv(pending, index=False)
                meta = {
                    "start": str(lower),
                    "end": str(upper),
                    "sha256": sha256(pending),
                    "retrieval_first": str(frame.retrieved_at.min()),
                    "retrieval_last": str(frame.retrieved_at.max()),
                    "quality": quality,
                    "attribution": "Elia Transmission Belgium SA",
                    "license_url": "https://opendata.elia.be/pages/licence/",
                    "source_urls": [
                        f"https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{s}"
                        for s in sorted(frame.source_dataset.unique())
                    ],
                    "normalization": (
                        "National interval-start MW; no filling/clipping; no forecast fields"
                    ),
                }
                pending_json = sidecar.with_suffix(".json.partial")
                write_json(pending_json, meta)
                pending.replace(csv)
                pending_json.replace(sidecar)
            # Verify source mapping as well as timestamps, including cache reads.
            for source in (OLD, CURRENT):
                rows = frame.loc[frame.source_dataset == source.dataset]
                if not rows.source_target_field.eq(source.target_field).all():
                    raise ValueError("Unexpected cached target field")
            if not frame.source_dataset.isin([OLD.dataset, CURRENT.dataset]).all():
                raise ValueError("Unexpected cached source dataset")
            frames.append(frame)
            metadata.append({"file": csv.relative_to(root).as_posix(), **meta})
    combined = pd.concat(frames, ignore_index=True)
    quality_summary(combined, first, stop)
    return combined, metadata


def daylight(keys: pd.DataFrame) -> pd.Series:
    flags = solar_features(pd.DatetimeIndex(keys.target_time.unique())).set_index("target_time")
    return keys.target_time.map(flags.is_daylight).astype(bool)


def grouped_diagnostics(aligned: pd.DataFrame, reference: str, by: str) -> pd.DataFrame:
    """Group already common-aligned rows, reusing established metric definitions."""
    frame = aligned.copy()
    local = frame.target_time.dt.tz_convert("Europe/Brussels")
    if by == "horizon_6h":
        frame[by] = (frame.horizon.dt.total_seconds() // 21600).astype(int) * 6
    elif by == "local_date":
        frame[by] = local.dt.strftime("%Y-%m-%d")
    else:
        raise ValueError("Unsupported diagnostic grouping")
    tables = []
    for label, group in frame.groupby(by, sort=True):
        inputs = [rows.drop(columns=by) for _, rows in group.groupby("model_id", sort=True)]
        table = compare_models(inputs, reference_model=reference).summary
        table.insert(0, by, label)
        tables.append(table)
    if not tables:
        raise ValueError("No common support for diagnostics")
    return pd.concat(tables, ignore_index=True)
