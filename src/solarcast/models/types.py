"""Small validated request, input context and deterministic result objects."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from solarcast.data.weather.types import WeatherCovariates
from solarcast.features._validation import cutoff_timestamp, pv_history, target_index


def _aware_time(value: str | datetime, name: str) -> pd.Timestamp:
    if not isinstance(value, (str, datetime)):
        raise ValueError(f"{name} must be an aware datetime or offset-aware timestamp string")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {name}") from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must have an explicit timezone/offset")
    return stamp.tz_convert("UTC")


def _table_times(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    if not frame.columns.is_unique or "target_time" not in frame:
        raise ValueError(f"{name} needs unique columns including target_time")
    if not isinstance(frame["target_time"].dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name}.target_time must be timezone-aware datetime data")
    return target_index(pd.DatetimeIndex(frame["target_time"]))


@dataclass(frozen=True, init=False, eq=False)
class ForecastRequest:
    """A single issue and explicit ordered future targets, canonicalized to UTC.

    Targets are quarter-hour interval starts, strictly after the issue; they need
    not be consecutive or have a fixed count. Issue time need not be quarter-hour aligned.
    """

    issue_time: pd.Timestamp
    _target_times: pd.DatetimeIndex = field(repr=False)

    def __init__(self, issue_time: str | datetime, target_times: pd.DatetimeIndex) -> None:
        issue = _aware_time(issue_time, "issue_time")
        times = target_index(target_times)
        if times.empty:
            raise ValueError("target_times must not be empty")
        if not times.is_monotonic_increasing:
            raise ValueError("target_times must be ordered in increasing UTC time")
        if (times <= issue).any():
            raise ValueError("All target_times must be strictly after issue_time")
        object.__setattr__(self, "issue_time", issue)
        object.__setattr__(self, "_target_times", times.copy(deep=True))

    @property
    def target_times(self) -> pd.DatetimeIndex:
        """A defensive copy of the explicit target grid."""
        return self._target_times.copy(deep=True)

    @property
    def horizons(self) -> pd.TimedeltaIndex:
        """Elapsed target-minus-issue durations, not ordinal step numbers."""
        return self._target_times - self.issue_time


@dataclass(frozen=True, init=False, eq=False)
class ForecastContext:
    """Inputs for exactly one request; no implicit fetching or feature generation.

    History contains only completed PV intervals at/before the explicit cutoff.
    This is a temporal check, not evidence of historical publication availability.
    Extra ingestion columns are deliberately excluded from model history.
    """

    request: ForecastRequest
    history_cutoff: pd.Timestamp
    _history: pd.DataFrame = field(repr=False)
    _features: pd.DataFrame | None = field(repr=False)
    weather: WeatherCovariates | None

    def __init__(
        self,
        request: ForecastRequest,
        history: pd.DataFrame,
        *,
        history_cutoff: str | datetime,
        features: pd.DataFrame | None = None,
        weather: WeatherCovariates | None = None,
    ) -> None:
        cutoff = cutoff_timestamp(history_cutoff)
        if weather is not None:
            if not isinstance(weather, WeatherCovariates):
                raise ValueError("weather must be normalized WeatherCovariates")
            if weather.issue_time != request.issue_time or not pd.DatetimeIndex(
                weather.features.target_time
            ).equals(request.target_times):
                raise ValueError("weather must match the request issue and targets exactly")
        if cutoff > request.issue_time:
            raise ValueError("history_cutoff must be at or before issue_time")
        series = pv_history(history, cutoff)
        if len(series) != len(history):
            raise ValueError(
                "history contains incomplete/after-cutoff intervals; "
                "filter explicitly before constructing ForecastContext"
            )
        normalized = pd.DataFrame({"target_time": series.index, "pv_generation_mw": series.array})
        feature_copy = None
        if features is not None:
            times = _table_times(features, "features")
            if not times.equals(request.target_times):
                raise ValueError("features must match requested targets exactly, in order")
            if {"pv_generation_mw", "monitored_capacity_mw"} & set(features.columns):
                raise ValueError("features must not contain PV labels or unverified capacity")
            feature_copy = features.copy(deep=True).reset_index(drop=True)
            feature_copy["target_time"] = times
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "history_cutoff", cutoff)
        object.__setattr__(self, "_history", normalized)
        object.__setattr__(self, "_features", feature_copy)
        object.__setattr__(self, "weather", weather)

    @property
    def history(self) -> pd.DataFrame:
        """Copied target/PV columns, with missing observations preserved."""
        return self._history.copy(deep=True)

    @property
    def features(self) -> pd.DataFrame | None:
        """Copied precomputed target-aligned features, or None.

        Their derivation/availability remains the caller's responsibility.
        """
        return None if self._features is None else self._features.copy(deep=True)


@dataclass(frozen=True, init=False, eq=False)
class Forecast:
    """Validated point forecasts plus constant metadata; no automatic clipping.

    Missing/NaN/infinite predictions are errors. A model that cannot forecast must
    fail explicitly instead of silently omitting targets or fabricating values.
    """

    request: ForecastRequest
    model_id: str
    model_version: str | None
    target_variable: str
    unit: str
    _predictions: pd.DataFrame = field(repr=False)

    def __init__(
        self,
        request: ForecastRequest,
        predictions: pd.DataFrame,
        *,
        model_id: str,
        model_version: str | None = None,
        target_variable: str = "pv_generation",
        unit: str = "MW",
    ) -> None:
        for name, value in (
            ("model_id", model_id),
            ("target_variable", target_variable),
            ("unit", unit),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if model_version is not None and (
            not isinstance(model_version, str) or not model_version.strip()
        ):
            raise ValueError("model_version must be a nonempty string or None")
        times = _table_times(predictions, "predictions")
        if set(predictions.columns) != {"target_time", "prediction_mw"}:
            raise ValueError("predictions must contain exactly target_time and prediction_mw")
        if not times.equals(request.target_times):
            raise ValueError("Predictions must match requested targets exactly, in order")
        values = predictions["prediction_mw"]
        if (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
        ):
            raise ValueError("prediction_mw must contain real numeric values")
        numeric = values.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(numeric).all():
            raise ValueError("prediction_mw must be finite and non-missing")
        if unit != "MW":
            raise ValueError("prediction_mw requires unit='MW'; no unit conversion is performed")
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "target_variable", target_variable)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(
            self,
            "_predictions",
            pd.DataFrame({"target_time": times, "prediction_mw": numeric.copy()}),
        )

    @property
    def issue_time(self) -> pd.Timestamp:
        """Issue time inherited from the validated request."""
        return self.request.issue_time

    @property
    def predictions(self) -> pd.DataFrame:
        """A copied two-column table; constant metadata stays on this object."""
        return self._predictions.copy(deep=True)

    def to_frame(self) -> pd.DataFrame:
        """Export a flat copy with issue/horizon/model/variable/unit metadata."""
        frame = self.predictions
        frame["issue_time"] = self.issue_time
        frame["horizon"] = self.request.horizons
        frame["model_id"] = self.model_id
        frame["model_version"] = self.model_version
        frame["target_variable"] = self.target_variable
        frame["unit"] = self.unit
        return frame
