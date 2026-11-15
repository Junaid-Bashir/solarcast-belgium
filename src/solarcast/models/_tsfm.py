"""Small shared UTC input/alignment helpers; no model runtime imports."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from solarcast.models import ForecastContext, ForecastDataError

INTERVAL = pd.Timedelta("15min")


@dataclass(frozen=True)
class ContextGrid:
    times: pd.DatetimeIndex
    values: npt.NDArray[np.float32]


def context_grid(context: ForecastContext, length: int, label: str) -> ContextGrid:
    """Bound elapsed slots; preserve interior/trailing gaps without extra history."""
    history = context.history
    cutoff = context.history_cutoff
    history = history.loc[history.target_time >= cutoff - length * INTERVAL]
    if history.empty:
        raise ForecastDataError(f"{label} has no history inside its configured context window")
    grid = pd.date_range(history.target_time.min(), cutoff, freq="15min", inclusive="left")
    series = history.set_index("target_time").pv_generation_mw.reindex(grid)
    with np.errstate(over="ignore", invalid="ignore"):
        values = series.to_numpy(dtype=np.float32, na_value=np.nan)
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ForecastDataError(f"{label} needs finite float32 observations; NaN gaps are allowed")
    return ContextGrid(grid, values)


def bridge(context: ForecastContext, limit: int, label: str) -> tuple[npt.NDArray[np.int64], int]:
    """Roll out from cutoff, then select exact requested times, including sparse sets."""
    offsets = np.asarray(
        (context.request.target_times - context.history_cutoff) // INTERVAL, dtype=np.int64
    )
    steps = int(offsets[-1]) + 1
    if steps > limit:
        raise ForecastDataError(f"{label} rollout {steps} exceeds the supported {limit}-step block")
    return offsets, steps
