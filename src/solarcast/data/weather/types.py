"""Small GFS-specific, UTC-only contracts; no optional binary dependencies."""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

PROVIDER = "NOAA/NCEP"
PRODUCT = "gfs.pgrb2.0p25.operational"
TRANSFORM = "gfs_belgium_v1"
FEATURES = (
    "nwp_ghi_west_mean_w_m2",
    "nwp_ghi_central_mean_w_m2",
    "nwp_ghi_east_mean_w_m2",
    "nwp_ghi_belgium_spatial_std_w_m2",
    "nwp_cloud_belgium_mean_fraction",
    "nwp_cloud_belgium_spatial_std_fraction",
    "nwp_temperature_2m_belgium_mean_c",
)
NATIVE_COLUMNS = (
    "reference_time",
    "valid_time",
    "forecast_lead",
    "latitude",
    "longitude",
    "dswrf_start_hour",
    "dswrf_end_hour",
    "dswrf_average_w_m2",
    "cloud_percent",
    "temperature_2m_k",
)


class WeatherError(ValueError):
    """Weather contract/coverage failure, never an instruction to fill values."""


def utc(value: str | datetime, name: str = "time") -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise WeatherError(f"{name} must have an explicit timezone")
    return stamp.tz_convert("UTC").as_unit("ns")


def targets_utc(targets: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(targets, pd.DatetimeIndex) or targets.tz is None:
        raise WeatherError("targets must be timezone-aware")
    result = targets.tz_convert("UTC").as_unit("ns")
    if (
        result.empty
        or result.hasnans
        or not result.is_unique
        or not result.is_monotonic_increasing
        or not result.equals(result.floor("15min"))
    ):
        raise WeatherError("targets must be nonempty, ordered, unique quarter-hour starts")
    return result.copy()


@dataclass(frozen=True)
class BelgiumGrid:
    """Output centres, not a claim that all bounding-box cells are Belgian."""

    north: float = 51.75
    west: float = 2.25
    south: float = 49.25
    east: float = 6.75
    spacing: float = 0.25

    @property
    def cells(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (float(y), float(x))
            for y in np.arange(self.south, self.north + 0.01, self.spacing)
            for x in np.arange(self.west, self.east + 0.01, self.spacing)
        )


GRID = BelgiumGrid()


@dataclass(frozen=True, init=False, eq=False)
class ForecastRun:
    """Owned native averages/states and provenance for one immutable source version.

    Missing rows/values remain representable; selection requires complete requested
    coverage. Duplicate keys and inconsistent time semantics are always errors.
    """

    reference_time: pd.Timestamp
    version: str
    observed_available_at: pd.Timestamp | None
    revision: bool
    version_available_at: pd.Timestamp | None
    _data: pd.DataFrame = field(repr=False)
    _metadata: dict[str, Any] = field(repr=False)

    def __init__(
        self,
        reference_time: str | datetime,
        data: pd.DataFrame,
        *,
        version: str,
        metadata: dict[str, Any],
        observed_available_at: str | datetime | None = None,
        revision: bool = False,
        version_available_at: str | datetime | None = None,
    ) -> None:
        reference = utc(reference_time, "reference_time")
        if reference.hour not in (0, 6, 12, 18) or reference != reference.floor("h"):
            raise WeatherError("GFS reference_time must be a 00/06/12/18 UTC cycle")
        if not version or not data.columns.is_unique or set(data.columns) != set(NATIVE_COLUMNS):
            raise WeatherError("native schema/version mismatch")
        frame = data.loc[:, list(NATIVE_COLUMNS)].copy(deep=True)
        for name in ("reference_time", "valid_time"):
            if not isinstance(frame[name].dtype, pd.DatetimeTZDtype):
                raise WeatherError(f"{name} must be timezone-aware")
            frame[name] = frame[name].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
        if frame.empty or not frame.reference_time.eq(reference).all():
            raise WeatherError("native rows must belong to exactly one reference_time")
        if frame.duplicated(["valid_time", "latitude", "longitude"]).any():
            raise WeatherError("duplicate native valid_time/grid keys")
        if not pd.api.types.is_timedelta64_dtype(frame.forecast_lead):
            raise WeatherError("forecast_lead must be an elapsed duration")
        frame["forecast_lead"] = frame.forecast_lead.astype("timedelta64[ns]")
        if not (frame.valid_time - reference).eq(frame.forecast_lead).all():
            raise WeatherError("forecast lead differs from valid_time - reference_time")
        for name in NATIVE_COLUMNS[3:]:
            if not pd.api.types.is_numeric_dtype(frame[name]):
                raise WeatherError(f"{name} must be numeric")
            frame[name] = frame[name].astype(float)
            if np.isinf(frame[name].to_numpy()).any():
                raise WeatherError(f"{name} cannot contain infinity")
        start, end = frame.dswrf_start_hour, frame.dswrf_end_hour
        if (
            start.isna().any()
            or end.isna().any()
            or (start < 0).any()
            or (end <= start).any()
            or (end > 120).any()
            or (start % 1 != 0).any()
            or (end % 1 != 0).any()
            or not frame.forecast_lead.eq(pd.to_timedelta(end, unit="h")).all()
        ):
            raise WeatherError("invalid hourly radiation bounds/lead")
        if not frame.valid_time.eq(frame.valid_time.dt.floor("h")).all():
            raise WeatherError("native valid times must be hourly")
        if (
            frame.cloud_percent.dropna().lt(0).any()
            or frame.cloud_percent.dropna().gt(100).any()
            or frame.temperature_2m_k.dropna().le(0).any()
        ):
            raise WeatherError("invalid percent cloud / Kelvin temperature")
        if frame[["latitude", "longitude"]].isna().any().any():
            raise WeatherError("missing coordinates")
        stamps = [
            None if v is None else utc(v, n)
            for v, n in (
                (observed_available_at, "observed_available_at"),
                (version_available_at, "version_available_at"),
            )
        ]
        if any(v is not None and v < reference for v in stamps):
            raise WeatherError("availability cannot precede initialization")
        if stamps[1] is not None and not revision:
            raise WeatherError("version availability requires an explicitly declared revision")
        if metadata.get("provider") != PROVIDER or metadata.get("product") != PRODUCT:
            raise WeatherError("only the selected operational GFS product is supported")
        for name, value in (
            ("reference_time", reference),
            ("version", version),
            ("observed_available_at", stamps[0]),
            ("revision", revision),
            ("version_available_at", stamps[1]),
            (
                "_data",
                frame.sort_values(["valid_time", "latitude", "longitude"]).reset_index(drop=True),
            ),
            ("_metadata", deepcopy(metadata)),
        ):
            object.__setattr__(self, name, value)

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy(deep=True)

    @property
    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._metadata)


@dataclass(frozen=True, init=False, eq=False)
class WeatherCovariates:
    """Complete numerical features plus the selected run's availability audit."""

    issue_time: pd.Timestamp
    reference_time: pd.Timestamp
    available_at: pd.Timestamp
    availability_basis: str
    policy_name: str
    run_version: str
    weights_id: str
    selection_notes: tuple[str, ...]
    radiation_policy: str
    radiation_adjusted_cell_hours: int
    _features: pd.DataFrame = field(repr=False)

    def __init__(
        self,
        features: pd.DataFrame,
        *,
        issue_time: str | datetime,
        reference_time: str | datetime,
        available_at: str | datetime,
        availability_basis: str,
        policy_name: str,
        run_version: str,
        weights_id: str,
        selection_notes: tuple[str, ...] = (),
        radiation_policy: str = "strict",
        radiation_adjusted_cell_hours: int = 0,
    ) -> None:
        issue, reference, available = (utc(issue_time), utc(reference_time), utc(available_at))
        if not reference <= available <= issue:
            raise WeatherError("weather availability must satisfy reference <= available <= issue")
        if not features.columns.is_unique or set(features.columns) != {"target_time", *FEATURES}:
            raise WeatherError("weather requires exactly the seven approved features and targets")
        times = targets_utc(pd.DatetimeIndex(features.target_time))
        if (times <= issue).any():
            raise WeatherError("weather targets must follow the issue")
        frame = features.loc[:, ["target_time", *FEATURES]].copy(deep=True)
        frame["target_time"] = times
        for name in FEATURES:
            if (
                not pd.api.types.is_numeric_dtype(frame[name])
                or pd.api.types.is_bool_dtype(frame[name])
                or pd.api.types.is_complex_dtype(frame[name])
            ):
                raise WeatherError("weather features must be real numeric values")
            frame[name] = frame[name].astype(float)
        if not np.isfinite(frame[list(FEATURES)].to_numpy(dtype=float)).all():
            raise WeatherError("missing/nonfinite requested weather")
        if availability_basis not in ("assumed", "observed"):
            raise WeatherError("availability basis must be explicit")
        if any(not value.strip() for value in (policy_name, run_version, weights_id)):
            raise WeatherError("weather provenance identifiers must be nonempty")
        if (
            radiation_policy not in ("strict", "grib_packing_bound_v1")
            or type(radiation_adjusted_cell_hours) is not int
            or radiation_adjusted_cell_hours < 0
        ):
            raise WeatherError("invalid weather radiation precision audit")
        for name, value in (
            ("issue_time", issue),
            ("reference_time", reference),
            ("available_at", available),
            ("availability_basis", availability_basis),
            ("policy_name", policy_name),
            ("run_version", run_version),
            ("weights_id", weights_id),
            ("selection_notes", tuple(selection_notes)),
            ("radiation_policy", radiation_policy),
            ("radiation_adjusted_cell_hours", radiation_adjusted_cell_hours),
            ("_features", frame.reset_index(drop=True)),
        ):
            object.__setattr__(self, name, value)

    @property
    def features(self) -> pd.DataFrame:
        return self._features.copy(deep=True)
