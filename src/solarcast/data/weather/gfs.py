"""Bounded public NODD index/range retrieval for operational hourly GFS only."""

import hashlib
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import httpx
import pandas as pd

from ._grib import decode_field
from .cache import WeatherCache, digest
from .types import GRID, PRODUCT, PROVIDER, TRANSFORM, ForecastRun, WeatherError, utc

ROOT = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
VARIABLES = {"DSWRF": "surface", "TCDC": "entire atmosphere", "TMP": "2 m above ground"}
Decoder = Callable[[bytes, str, pd.Timestamp, int, int], pd.DataFrame]


@dataclass(frozen=True)
class IndexField:
    variable: str
    start_hour: int
    end_hour: int
    first_byte: int
    last_byte: int


def parse_index(text: str, reference: pd.Timestamp, step: int) -> tuple[IndexField, ...]:
    """Exact level/statistic matching excludes average/convective TCDC duplicates."""
    records: list[tuple[int, str, str, str]] = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            raise WeatherError("malformed GFS index")
        try:
            number, offset = int(parts[0]), int(parts[1])
        except ValueError as error:
            raise WeatherError("invalid GFS index offsets") from error
        if number != len(records) + 1 or offset < 0 or (records and offset <= records[-1][0]):
            raise WeatherError("unordered/duplicate GFS index offsets")
        if parts[2] != f"d={reference.strftime('%Y%m%d%H')}":
            raise WeatherError("index reference_time mismatch")
        records.append((offset, parts[3], parts[4], parts[5]))
    fields = []
    for variable, level in VARIABLES.items():
        matches = []
        for i, (offset, var, lev, statistic) in enumerate(records):
            if var != variable or lev != level:
                continue
            pattern = r"(\d+)-(\d+) hour ave fcst" if variable == "DSWRF" else r"(\d+) hour fcst"
            match = re.fullmatch(pattern, statistic)
            if match is None:
                continue
            start = int(match[1])
            end = int(match[2]) if variable == "DSWRF" else start
            if end != step or start > end or (variable == "DSWRF" and start == end):
                raise WeatherError("index forecast-step/statistic mismatch")
            if i + 1 == len(records):
                raise WeatherError("selected last index record has no bounded end offset")
            matches.append(IndexField(variable, start, end, offset, records[i + 1][0] - 1))
        if len(matches) != 1:
            raise WeatherError(f"expected exactly one {variable}/{level} field, got {len(matches)}")
        fields.append(matches[0])
    return tuple(fields)


def retrieval_identity(reference_time: str | datetime, steps: Sequence[int]) -> dict[str, Any]:
    reference = utc(reference_time, "reference_time")
    ordered = list(steps)
    if (
        reference < pd.Timestamp("2024-01-01T00Z")
        or reference != reference.floor("h")
        or reference.hour not in (0, 6, 12, 18)
    ):
        raise WeatherError("supported GFS archive: 2024+ 00/06/12/18 UTC initialization")
    if (
        not ordered
        or len(ordered) > 49
        or any(type(s) is not int for s in ordered)
        or ordered != sorted(set(ordered))
        or ordered[0] < 1
        or ordered[-1] > 120
        or ordered != list(range(ordered[0], ordered[-1] + 1))
    ):
        raise WeatherError("steps must be 1..120 consecutive integer hours, at most 49 files")
    return {
        "provider": PROVIDER,
        "product": PRODUCT,
        "reference_time": reference.isoformat(),
        "steps": ordered,
        "variables": {
            "DSWRF": "surface/avg/W m-2",
            "TCDC": "entire atmosphere/instant/%",
            "TMP": "2m/instant/K",
        },
        "grid": asdict(GRID),
        "transform": TRANSFORM,
        "decoder_contract": "eccodes-gfs-v2-packing",
    }


class GFSClient:
    """Three global message byte ranges per step, cropped immediately to Belgium.

    Never request a complete GRIB object. Reject HTTP 200 before reading its body.
    Bounds apply to every streamed response and total attempted response bytes.
    """

    def __init__(
        self,
        *,
        cache: WeatherCache | None = None,
        timeout: float = 60,
        max_retries: int = 2,
        request_interval: float = 0.15,
        max_message_bytes: int = 8_000_000,
        max_total_bytes: int = 128_000_000,
        transport: httpx.BaseTransport | None = None,
        decoder: Decoder = decode_field,
    ) -> None:
        if (
            timeout <= 0
            or max_retries not in range(5)
            or request_interval < 0
            or max_message_bytes <= 0
            or max_total_bytes <= 0
        ):
            raise WeatherError("invalid GFS HTTP bounds")
        self.cache, self.decoder = cache, decoder
        self.max_retries, self.request_interval = max_retries, request_interval
        self.max_message_bytes, self.max_total_bytes = max_message_bytes, max_total_bytes
        self._bytes, self._last_request = 0, 0.0
        self._client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)

    def __enter__(self) -> "GFSClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get(self, url: str, *, field: IndexField | None = None) -> tuple[bytes, dict[str, str]]:
        limit = 512_000 if field is None else field.last_byte - field.first_byte + 1
        if field is not None and limit > self.max_message_bytes:
            raise WeatherError("GRIB range exceeds configured message bound")
        headers = (
            {"Range": f"bytes={field.first_byte}-{field.last_byte}", "Accept-Encoding": "identity"}
            if field is not None
            else {}
        )
        for attempt in range(self.max_retries + 1):
            wait = self.request_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                with self._client.stream("GET", url, headers=headers) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        response.raise_for_status()
                    if response.status_code != (200 if field is None else 206):
                        raise WeatherError(
                            f"unexpected HTTP {response.status_code} for bounded GFS request"
                        )
                    if response.headers.get("content-encoding", "identity") != "identity":
                        raise WeatherError("encoded range response is unsupported")
                    if field is not None:
                        content_range = response.headers.get("content-range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if (
                            match is None
                            or int(match[1]) != field.first_byte
                            or int(match[2]) != field.last_byte
                            or int(match[3]) <= field.last_byte
                        ):
                            raise WeatherError("invalid Content-Range")
                    length = response.headers.get("content-length")
                    if length is not None and (
                        int(length) > limit or (field is not None and int(length) != limit)
                    ):
                        raise WeatherError("invalid bounded Content-Length")
                    chunks, count = [], 0
                    for chunk in response.iter_bytes(chunk_size=65536):
                        count += len(chunk)
                        self._bytes += len(chunk)
                        if count > limit or self._bytes > self.max_total_bytes:
                            raise WeatherError("GFS response/total transfer bound exceeded")
                        chunks.append(chunk)
                    if field is not None and count != limit:
                        raise WeatherError("truncated GRIB range")
                    return b"".join(chunks), {
                        k: response.headers[k]
                        for k in ("etag", "last-modified", "content-range")
                        if k in response.headers
                    }
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                if attempt == self.max_retries:
                    raise WeatherError(
                        f"GFS request failed after {attempt + 1} attempts"
                    ) from error
                time.sleep(min(2**attempt, 4))
        raise AssertionError("unreachable")

    def fetch_forecast_run(
        self, reference_time: str | datetime, *, steps: Sequence[int]
    ) -> ForecastRun:
        identity = retrieval_identity(reference_time, steps)
        if self.cache is not None:
            cached = self.cache.load(identity)
            if cached is not None:
                return cached
        reference = utc(reference_time)
        frames, sources = [], []
        self._bytes = 0
        for step in steps:
            url = (
                f"{ROOT}/gfs.{reference:%Y%m%d}/{reference:%H}/atmos/"
                f"gfs.t{reference:%H}z.pgrb2.0p25.f{step:03d}"
            )
            index, index_headers = self._get(url + ".idx")
            try:
                fields = parse_index(index.decode("ascii"), reference, step)
            except UnicodeDecodeError as error:
                raise WeatherError("non-ASCII GFS index") from error
            merged: pd.DataFrame | None = None
            etag: str | None = None
            field_sources = []
            for selected in fields:
                raw, headers = self._get(url, field=selected)
                if "etag" not in headers:
                    raise WeatherError("GFS source version requires an ETag")
                if etag is not None and etag != headers["etag"]:
                    raise WeatherError("GFS object changed during range retrieval")
                etag = headers["etag"]
                decoded = self.decoder(
                    raw, selected.variable, reference, selected.start_hour, selected.end_hour
                )
                packing = decoded.attrs.get("packing")
                if (
                    set(decoded.columns) != {"latitude", "longitude", "value"}
                    or tuple(zip(decoded.latitude, decoded.longitude, strict=True)) != GRID.cells
                ):
                    raise WeatherError("decoder returned an unexpected Belgium grid")
                name = {
                    "DSWRF": "dswrf_average_w_m2",
                    "TCDC": "cloud_percent",
                    "TMP": "temperature_2m_k",
                }[selected.variable]
                decoded = decoded.rename(columns={"value": name})
                if selected.variable == "DSWRF":
                    decoded["dswrf_start_hour"] = selected.start_hour
                    decoded["dswrf_end_hour"] = selected.end_hour
                merged = (
                    decoded
                    if merged is None
                    else merged.merge(decoded, on=["latitude", "longitude"], validate="one_to_one")
                )
                field_sources.append(
                    {
                        **asdict(selected),
                        "packing": packing,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                        "headers": headers,
                    }
                )
            assert merged is not None
            merged["reference_time"] = reference
            merged["valid_time"] = reference + pd.Timedelta(hours=step)
            merged["forecast_lead"] = pd.Timedelta(hours=step)
            frames.append(merged)
            sources.append(
                {
                    "url": url,
                    "index_sha256": hashlib.sha256(index).hexdigest(),
                    "index_headers": index_headers,
                    "fields": field_sources,
                }
            )
        metadata = {
            "provider": PROVIDER,
            "product": PRODUCT,
            "identity": identity,
            "sources": sources,
            "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "availability": "unknown; use explicit policy, not retrieved_at/Last-Modified",
            "transferred_bytes": self._bytes,
        }
        version = digest({"identity": identity, "sources": sources})
        run = ForecastRun(
            reference, pd.concat(frames, ignore_index=True), version=version, metadata=metadata
        )
        if self.cache is not None:
            self.cache.save(identity, run)
        return run
