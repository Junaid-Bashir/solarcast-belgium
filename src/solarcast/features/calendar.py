"""Future-known Belgian calendar features, derived only from aware target times."""

import numpy as np
import pandas as pd

from solarcast.features._validation import target_index


def calendar_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep UTC keys; derive Brussels clock fields and two cyclical pairs.

    Day-of-year phase uses fractional local calendar day divided by 365/366 for
    that calendar year. Clock-time phase is not a position in a 96-interval day.
    """
    utc = target_index(times)
    local = utc.tz_convert("Europe/Brussels")
    minute_of_day = (local.hour * 60 + local.minute).to_numpy(dtype="float64")
    daily_phase = 2 * np.pi * minute_of_day / 1440
    yearly_phase = (
        2
        * np.pi
        * (local.day_of_year.to_numpy(dtype="float64") - 1 + minute_of_day / 1440)
        / np.where(local.is_leap_year, 366, 365)
    )
    offsets: list[int] = []
    for stamp in local:
        offset = stamp.utcoffset()
        assert offset is not None  # target_index requires timezone-aware input.
        offsets.append(int(offset.total_seconds() / 60))
    return pd.DataFrame(
        {
            "target_time": utc,
            "local_hour": local.hour.astype("int64"),
            "local_minute": local.minute.astype("int64"),
            "quarter_hour_in_hour": (local.minute // 15).astype("int64"),
            "day_of_week": local.day_of_week.astype("int64"),
            "day_of_year": local.day_of_year.astype("int64"),
            "month": local.month.astype("int64"),
            "utc_offset_minutes": np.array(offsets, dtype="int64"),
            "time_of_day_sin": np.sin(daily_phase),
            "time_of_day_cos": np.cos(daily_phase),
            "day_of_year_sin": np.sin(yearly_phase),
            "day_of_year_cos": np.cos(yearly_phase),
        }
    )
