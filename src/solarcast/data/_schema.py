"""Elia-specific mappings, kept outside the public data representation."""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from solarcast.data.exceptions import EliaSchemaError, PVDataValidationError

logger = logging.getLogger(__name__)

# 26 October 2026 midnight Europe/Brussels, verified in docs/data/elia.md.
TRANSITION = datetime(2026, 10, 25, 23, tzinfo=UTC)
INTERVAL = pd.Timedelta(minutes=15)


@dataclass(frozen=True)
class Source:
    """Only the differences between the two verified historical schemas."""

    dataset: str
    target_field: str
    grid_total: bool = False

    @property
    def fields(self) -> str:
        fields = ["datetime", "region", self.target_field, "monitoredcapacity"]
        fields.append("gridconnectiontype" if self.grid_total else "resolutioncode")
        return ",".join(fields)

    @property
    def selection(self) -> str:
        return "region='Belgium'" + (" and gridconnectiontype='Total'" if self.grid_total else "")


OLD = Source("ods032", "measured")
CURRENT = Source("ods209", "measured_upscaled", grid_total=True)


def utc_timestamp(value: str | datetime, *, name: str) -> pd.Timestamp:
    """Parse an explicitly offset-aware timestamp without guessing a local zone."""
    if not isinstance(value, (str, datetime)):
        raise ValueError(f"{name} must be an offset-aware datetime or ISO timestamp")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must have an explicit timezone/UTC offset")
    return stamp.tz_convert("UTC")


def _number(value: object, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EliaSchemaError(f"{context}: expected a JSON number or null, got {value!r}")
    try:
        result = float(value)
    except OverflowError as error:
        raise PVDataValidationError(f"{context}: non-finite numeric value") from error
    if not math.isfinite(result):
        raise PVDataValidationError(f"{context}: non-finite numeric value {result!r}")
    return result


def normalize(
    records: Sequence[Mapping[str, object]], source: Source, retrieved_at: datetime
) -> pd.DataFrame:
    """Normalize one source, rejecting unexpected dimensions instead of aggregating."""
    rows: list[dict[str, object]] = []
    for record in records:
        missing = set(source.fields.split(",")) - record.keys()
        if missing:
            raise EliaSchemaError(f"{source.dataset}: missing fields {sorted(missing)}")
        if record["region"] != "Belgium" or (
            source.grid_total and record["gridconnectiontype"] != "Total"
        ):
            raise EliaSchemaError(f"{source.dataset}: response violated national filter")
        if not source.grid_total and record["resolutioncode"] != "PT15M":
            raise EliaSchemaError(f"{source.dataset}: expected resolutioncode='PT15M'")
        value = record["datetime"]
        if not isinstance(value, str):
            raise EliaSchemaError(f"{source.dataset}: datetime must be an offset-aware string")
        try:
            stamp = utc_timestamp(value, name="source datetime")
        except ValueError as error:
            raise EliaSchemaError(f"{source.dataset}: {error}") from error
        context = f"{source.dataset} at {stamp}"
        rows.append(
            {
                "target_time": stamp,
                "pv_generation_mw": _number(
                    record[source.target_field], context=f"{context} {source.target_field}"
                ),
                "monitored_capacity_mw": _number(
                    record["monitoredcapacity"], context=f"{context} monitoredcapacity"
                ),
                "source_dataset": source.dataset,
                "source_target_field": source.target_field,
                "retrieved_at": retrieved_at,
            }
        )
    frame = pd.DataFrame(rows, columns=list(DTYPES))
    frame = frame.astype(DTYPES)
    if not frame["target_time"].is_monotonic_increasing:
        logger.warning("%s: source rows were not ordered; sorting by UTC target", source.dataset)
    return frame.sort_values("target_time", ignore_index=True)


DTYPES = {
    "target_time": "datetime64[ns, UTC]",
    "pv_generation_mw": "Float64",
    "monitored_capacity_mw": "Float64",
    "source_dataset": "string",
    "source_target_field": "string",
    "retrieved_at": "datetime64[ns, UTC]",
}
