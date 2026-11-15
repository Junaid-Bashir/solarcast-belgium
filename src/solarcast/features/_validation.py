"""Feature input validation independent of Elia-specific source schemas."""

from datetime import datetime

import numpy as np
import pandas as pd


def target_index(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return unique, quarter-hour UTC timestamps in caller order, never localize naive input."""
    if not isinstance(times, pd.DatetimeIndex) or times.tz is None:
        raise ValueError("times must be a timezone-aware pandas DatetimeIndex")
    utc = times.tz_convert("UTC").as_unit("ns")
    if utc.hasnans or not utc.is_unique:
        raise ValueError("times must be non-null and unique")
    if not utc.equals(utc.floor("15min")):
        raise ValueError("times must be aligned to 15-minute interval starts")
    return utc


def cutoff_timestamp(value: str | datetime) -> pd.Timestamp:
    """Require an explicit, quarter-hour cutoff; it is not a publication-time assertion."""
    if not isinstance(value, (str, datetime)):
        raise ValueError("history_cutoff must be a timezone-aware timestamp")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Invalid history_cutoff") from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("history_cutoff must have an explicit timezone/offset")
    stamp = stamp.tz_convert("UTC")
    if stamp != stamp.floor("15min"):
        raise ValueError("history_cutoff must be aligned to a quarter hour")
    return stamp


def pv_history(history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Copy eligible completed PV intervals, preserving nulls and negative values."""
    if not history.columns.is_unique or not {"target_time", "pv_generation_mw"} <= set(
        history.columns
    ):
        raise ValueError("history needs unique columns target_time and pv_generation_mw")
    dtype = history["target_time"].dtype
    if not isinstance(dtype, pd.DatetimeTZDtype):
        raise ValueError("history target_time must be timezone-aware datetime data")
    times = target_index(pd.DatetimeIndex(history["target_time"]))
    values = history["pv_generation_mw"]
    if (
        not pd.api.types.is_numeric_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_complex_dtype(values.dtype)
    ):
        raise ValueError("pv_generation_mw must be real numeric and nullable")
    numeric = values.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numeric).any():
        raise ValueError("pv_generation_mw contains infinite values")
    series = pd.Series(pd.array(numeric, dtype="Float64"), index=times)
    # target_time labels interval START. Even with zero publication delay, its
    # average is not complete until 15 minutes later. Actual delays remain unknown.
    return series.loc[times + pd.Timedelta(minutes=15) <= cutoff].sort_index()
