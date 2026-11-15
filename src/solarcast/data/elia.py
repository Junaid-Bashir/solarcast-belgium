"""Bounded historical Elia PV retrieval using the verified Explore API contract."""

import logging
import math
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import cast

import httpx
import pandas as pd

from solarcast.data._schema import CURRENT, OLD, TRANSITION, Source, normalize, utc_timestamp
from solarcast.data.exceptions import EliaAPIError, EliaSchemaError, PVDataValidationError
from solarcast.data.validation import validate_history

logger = logging.getLogger(__name__)
API_BASE = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class EliaPVClient:
    """Fetch national historical estimates; use as a context manager to close HTTP.

    Inputs must be timezone-aware datetimes or timestamp strings with explicit
    offsets, aligned to quarter hours. Ranges are half-open [start, end).
    No observation availability, cache, forecasts, or near-real-time fallback is inferred.
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        request_interval: float = 0.25,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if not math.isfinite(request_interval) or request_interval < 0:
            raise ValueError("request_interval must be finite and nonnegative")
        self._http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)
        self._max_retries = max_retries
        self._request_interval = request_interval
        self._last_request: float | None = None

    def __enter__(self) -> "EliaPVClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def fetch_history(self, *, start: str | datetime, end: str | datetime) -> pd.DataFrame:
        """Return UTC observations and per-row provenance, sorted by target time.

        Structural errors raise; missing interval rows and negative/null observations
        are logged and preserved. A gap/overlap at the schema seam raises.
        """
        first = utc_timestamp(start, name="start")
        stop = utc_timestamp(end, name="end")
        if first >= stop:
            raise ValueError("start must be earlier than end")
        if first != first.floor("15min") or stop != stop.floor("15min"):
            raise ValueError("start and end must be aligned to 15-minute intervals")
        frames: list[pd.DataFrame] = []
        for source, lower, upper in (
            (OLD, first, min(stop, pd.Timestamp(TRANSITION))),
            (CURRENT, max(first, pd.Timestamp(TRANSITION)), stop),
        ):
            if lower >= upper:
                continue
            logger.info("Fetching %s [%s, %s)", source.dataset, lower, upper)
            window = lower
            while window < upper:
                window_end = min(window + timedelta(days=1), upper)
                records = self._records(source, window, window_end)
                try:
                    frame = normalize(records, source, datetime.now(UTC))
                    validate_history(frame, window, window_end)
                except (EliaSchemaError, PVDataValidationError) as error:
                    raise type(error)(
                        f"{source.dataset} [{window}, {window_end}): {error}"
                    ) from error
                frames.append(frame)
                window = window_end
        combined = pd.concat(frames, ignore_index=True).sort_values(
            "target_time", ignore_index=True
        )
        validate_history(combined, first, stop)
        return combined

    def _records(
        self, source: Source, start: pd.Timestamp, end: pd.Timestamp
    ) -> list[dict[str, object]]:
        context = f"{source.dataset} [{start}, {end})"
        params = {
            "select": source.fields,
            "where": (
                f"{source.selection} and datetime>=date'{start.isoformat()}' "
                f"and datetime<date'{end.isoformat()}'"
            ),
            "order_by": "datetime",
            "limit": "100",
            "offset": "0",
        }
        response = self._get(API_BASE + source.dataset + "/records", params, context)
        try:
            payload = response.json()
        except ValueError as error:
            raise EliaAPIError(f"{context}: response was not valid JSON") from error
        if not isinstance(payload, dict):
            raise EliaAPIError(f"{context}: expected a JSON object")
        count = payload.get("total_count")
        records = payload.get("results")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or not isinstance(records, list)
            or any(not isinstance(row, dict) for row in records)
        ):
            raise EliaAPIError(f"{context}: invalid total_count/results envelope")
        # One UTC day has at most 96 unique national quarter-hours. A larger count
        # is a filter/duplicate/schema failure, never a reason to paginate blindly.
        if count > 96:
            raise PVDataValidationError(f"{context}: more than 96 rows in a UTC-day window")
        if len(records) != count:
            raise EliaAPIError(f"{context}: incomplete response ({len(records)} of {count} rows)")
        return cast(list[dict[str, object]], records)

    def _get(self, url: str, params: dict[str, str], context: str) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            if self._last_request is not None:
                delay = self._request_interval - (time.monotonic() - self._last_request)
                if delay > 0:
                    time.sleep(delay)
            self._last_request = time.monotonic()
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError as error:
                if attempt == self._max_retries:
                    raise EliaAPIError(
                        f"{context}: transport failure after {attempt + 1} attempts"
                    ) from error
                delay = min(2.0**attempt, 30.0)
            else:
                if response.status_code == 200:
                    return response
                if response.status_code not in RETRY_STATUSES or attempt == self._max_retries:
                    raise EliaAPIError(f"{context}: HTTP {response.status_code}")
                delay = self._retry_delay(response, attempt, context)
            logger.warning("%s: retry %d after %.2fs", context, attempt + 1, delay)
            time.sleep(delay)
        raise AssertionError("Unreachable retry state")

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int, context: str) -> float:
        header = response.headers.get("Retry-After")
        if header is None:
            return min(2.0**attempt, 30.0)
        try:
            delay = float(header)
        except ValueError:
            try:
                stamp = parsedate_to_datetime(header)
                if stamp.tzinfo is None:
                    raise ValueError("Retry-After date lacks timezone")
                delay = (stamp - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError) as error:
                raise EliaAPIError(f"{context}: invalid Retry-After header") from error
        if not math.isfinite(delay) or delay > 60:
            # Stop rather than retry earlier than instructed or block indefinitely.
            raise EliaAPIError(f"{context}: Retry-After exceeds the 60s retry wait budget")
        return max(0.0, delay)
