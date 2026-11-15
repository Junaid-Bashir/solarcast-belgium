"""Non-destructive validation of normalized historical PV tables."""

import logging

import pandas as pd

from solarcast.data._schema import CURRENT, INTERVAL, OLD, TRANSITION
from solarcast.data.exceptions import PVDataValidationError

logger = logging.getLogger(__name__)


def validate_history(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Raise for structural failures; log missing/negative values without correction.

    A crossing request requires the two observations adjacent to the transition.
    Temporal continuity does not prove methodological continuity.
    """
    times = frame["target_time"]
    if str(times.dtype) != "datetime64[ns, UTC]" or times.isna().any():
        raise PVDataValidationError("target_time must be non-null timezone-aware UTC")
    if (times != times.dt.floor("15min")).any():
        raise PVDataValidationError("target_time is not aligned to 15-minute intervals")
    if not times.is_monotonic_increasing:
        raise PVDataValidationError("target_time must be sorted")
    if frame.duplicated(["target_time", "source_dataset"]).any() or times.duplicated().any():
        raise PVDataValidationError("Duplicate/overlapping national target timestamps")
    if ((times < start) | (times >= end)).any():
        raise PVDataValidationError(f"Source returned timestamps outside [{start}, {end})")
    old = frame["source_dataset"] == OLD.dataset
    current = frame["source_dataset"] == CURRENT.dataset
    if ((old & (times >= TRANSITION)) | (current & (times < TRANSITION))).any():
        raise PVDataValidationError("Source data violates the documented transition boundary")
    if start < TRANSITION < end:
        before = times[old]
        after = times[current]
        if (
            before.empty
            or after.empty
            or before.iloc[-1] != TRANSITION - INTERVAL
            or after.iloc[0] != TRANSITION
        ):
            raise PVDataValidationError("Gap or unexpected ordering at ods032/ods209 seam")
    expected = int((end - start) / INTERVAL)
    if len(frame) != expected:
        logger.warning("[%s, %s): %d missing interval rows", start, end, expected - len(frame))
    for column in ("pv_generation_mw", "monitored_capacity_mw"):
        values = frame[column]
        # Numeric decoding rejects non-finite values before nullable conversion.
        if values.isin([float("inf"), float("-inf")]).any():
            raise PVDataValidationError(f"{column}: non-finite values")
        if values.isna().any():
            logger.warning("%s: %d null values preserved", column, int(values.isna().sum()))
        if (values < 0).any():
            logger.warning("%s: %d negative values preserved", column, int((values < 0).sum()))
