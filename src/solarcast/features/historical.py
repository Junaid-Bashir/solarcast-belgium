"""Historical-observed features with explicit cutoff and elapsed-time semantics."""

from datetime import datetime

import pandas as pd

from solarcast.features._validation import cutoff_timestamp, pv_history, target_index


def pv_lag_features(
    history: pd.DataFrame,
    times: pd.DatetimeIndex,
    *,
    history_cutoff: str | datetime,
) -> pd.DataFrame:
    """Look up exact elapsed 24h/48h/7d lags, subject to completed-interval cutoff.

    Missing/unavailable lags stay missing. The caller must additionally enforce
    true publication/vintage availability; this function cannot infer it.
    """
    utc = target_index(times)
    series = pv_history(history, cutoff_timestamp(history_cutoff))
    result = pd.DataFrame({"target_time": utc})
    for name, duration in (("24h", "24h"), ("48h", "48h"), ("7d", "7D")):
        result[f"pv_lag_{name}_mw"] = (
            series.reindex(utc - pd.Timedelta(duration)).astype("Float64").array
        )
    return result


def pv_rolling_features(
    history: pd.DataFrame,
    times: pd.DatetimeIndex,
    *,
    history_cutoff: str | datetime,
) -> pd.DataFrame:
    """Mean/max in [anchor-24h, anchor), where anchor=min(target, history_cutoff).

    Require all 96 elapsed quarter-hours to be present and non-null. For future
    targets the window stops at the explicit cutoff rather than moving into future
    observations. No current-target interval or incomplete cutoff interval is used.
    """
    utc = target_index(times)
    cutoff = cutoff_timestamp(history_cutoff)
    series = pv_history(history, cutoff)
    anchors = pd.DatetimeIndex([min(stamp, cutoff) for stamp in utc], tz="UTC")
    # Several future targets share the cutoff anchor; compute that window once.
    index = series.index.union(anchors.unique()).sort_values()
    rolling = series.reindex(index).rolling("24h", closed="left", min_periods=96)
    return pd.DataFrame(
        {
            "target_time": utc,
            "pv_trailing_24h_mean_mw": rolling.mean().reindex(anchors).astype("Float64").array,
            "pv_trailing_24h_max_mw": rolling.max().reindex(anchors).astype("Float64").array,
        }
    )
