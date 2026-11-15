"""No-network HTTP, GRIB boundary and immutable cache tests."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solarcast.data.weather import GRID, GFSClient, WeatherCache, WeatherError, retrieval_identity
from solarcast.data.weather._grib import decode_field
from solarcast.data.weather.cache import digest
from solarcast.data.weather.gfs import IndexField, parse_index
from solarcast.data.weather.spatial import BOUNDARY_SHA256, belgium_weights

REFERENCE = pd.Timestamp("2025-06-10T00Z")


def index_text(step: int = 25) -> str:
    start = ((step - 1) // 6) * 6
    return (
        f"1:0:d=2025061000:TMP:2 m above ground:{step} hour fcst:\n"
        f"2:10:d=2025061000:TCDC:entire atmosphere:{start}-{step} hour ave fcst:\n"
        f"3:20:d=2025061000:TCDC:entire atmosphere:{step} hour fcst:\n"
        f"4:30:d=2025061000:TCDC:convective cloud layer:{step} hour fcst:\n"
        f"5:40:d=2025061000:DSWRF:surface:{start}-{step} hour ave fcst:\n"
        f"6:50:d=2025061000:OTHER:surface:{step} hour fcst:\n"
    )


def fake_decode(
    raw: bytes, variable: str, reference: pd.Timestamp, start: int, end: int
) -> pd.DataFrame:
    assert len(raw) == 10 and reference == REFERENCE
    return pd.DataFrame(
        {
            "latitude": [v[0] for v in GRID.cells],
            "longitude": [v[1] for v in GRID.cells],
            "value": {"DSWRF": 100.0, "TCDC": 50.0, "TMP": 293.15}[variable],
        }
    )


def respond(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith(".idx"):
        step = int(request.url.path[-7:-4])
        return httpx.Response(200, text=index_text(step))
    byte_range = request.headers["range"].removeprefix("bytes=")
    return httpx.Response(
        206,
        content=b"0123456789",
        headers={"Content-Range": f"bytes {byte_range}/60", "ETag": '"source-object-v1"'},
    )


def test_exact_index_matching_excludes_cloud_averages_and_other_levels() -> None:
    fields = parse_index(index_text(), REFERENCE, 25)
    assert [(f.variable, f.first_byte, f.last_byte) for f in fields] == [
        ("DSWRF", 40, 49),
        ("TCDC", 20, 29),
        ("TMP", 0, 9),
    ]
    assert fields[0].start_hour == 24


@pytest.mark.parametrize("fault", ["reference", "step", "duplicate", "missing", "offset"])
def test_index_rejects_ambiguous_or_incorrect_source(fault: str) -> None:
    text = index_text()
    if fault == "reference":
        text = text.replace("d=2025061000", "d=2025061006")
    elif fault == "step":
        text = text.replace("25 hour fcst", "26 hour fcst")
    elif fault == "duplicate":
        text = text.replace("TCDC:convective cloud layer", "TCDC:entire atmosphere")
    elif fault == "missing":
        text = text.replace("DSWRF:", "WRONG:")
    else:
        text = text.replace("6:50:", "6:40:")
    with pytest.raises(WeatherError):
        parse_index(text, REFERENCE, 25)


def test_bounded_client_crops_only_three_fields_and_preserves_provenance() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    with GFSClient(
        transport=httpx.MockTransport(handler), decoder=fake_decode, request_interval=0
    ) as client:
        run = client.fetch_forecast_run(REFERENCE, steps=[24, 25])
    assert len(run.data) == 2 * 209
    assert len(requests) == 8
    assert sum("range" in r.headers for r in requests) == 6
    assert run.observed_available_at is None
    assert (
        run.metadata["sources"][0]["fields"][0]["sha256"]
        == hashlib.sha256(b"0123456789").hexdigest()
    )
    assert run.metadata["identity"]["steps"] == [24, 25]


class UnreadBody(httpx.SyncByteStream):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("full GRIB response body must not be read")
        yield b""  # pragma: no cover


def test_rejected_http_200_never_reads_giant_grib_body() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=UnreadBody())

    with GFSClient(transport=httpx.MockTransport(handler), request_interval=0) as client:
        with pytest.raises(WeatherError, match="HTTP 200"):
            client._get("https://example.test/gfs", field=IndexField("TMP", 25, 25, 0, 9))


@pytest.mark.parametrize("fault", ["range", "length", "truncated", "oversize", "encoding", "etag"])
def test_reject_bad_range_responses(fault: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".idx"):
            return respond(request)
        response = respond(request)
        if fault == "range":
            response.headers["Content-Range"] = "bytes 1-10/60"
        elif fault == "length":
            response.headers["Content-Length"] = "500000000"
        elif fault == "truncated":
            return httpx.Response(
                206,
                content=b"x",
                headers={
                    "Content-Range": response.headers["Content-Range"],
                    "Content-Length": "10",
                },
            )
        elif fault == "oversize":
            return httpx.Response(
                206,
                stream=httpx.ByteStream(b"x" * 11),
                headers={"Content-Range": response.headers["Content-Range"]},
            )
        elif fault == "encoding":
            response.headers["Content-Encoding"] = "gzip"
        else:
            del response.headers["etag"]
        return response

    with GFSClient(
        transport=httpx.MockTransport(handler), decoder=fake_decode, request_interval=0
    ) as client:
        with pytest.raises(WeatherError):
            client.fetch_forecast_run(REFERENCE, steps=[25])


def test_bounded_retry_and_total_transfer_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("solarcast.data.weather.gfs.time.sleep", lambda _: None)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls <= 2:
            return httpx.Response(503)
        return respond(request)

    with GFSClient(
        transport=httpx.MockTransport(handler),
        decoder=fake_decode,
        request_interval=0,
        max_retries=2,
    ) as client:
        assert len(client.fetch_forecast_run(REFERENCE, steps=[25]).data) == 209
    assert calls == 6
    with GFSClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(429)),
        request_interval=0,
        max_retries=1,
    ) as client:
        with pytest.raises(WeatherError, match="2 attempts"):
            client.fetch_forecast_run(REFERENCE, steps=[25])


def test_http_timeout_retries_are_bounded_and_rate_spacing_is_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("solarcast.data.weather.gfs.time.sleep", sleeps.append)
    monkeypatch.setattr("solarcast.data.weather.gfs.time.monotonic", lambda: 0.0)
    calls = 0

    def timeout(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert set(request.extensions["timeout"].values()) == {1.25}
        raise httpx.ReadTimeout("offline simulated timeout", request=request)

    with GFSClient(
        transport=httpx.MockTransport(timeout), timeout=1.25, max_retries=1, request_interval=0.25
    ) as client:
        with pytest.raises(WeatherError, match="2 attempts"):
            client.fetch_forecast_run(REFERENCE, steps=[25])
    assert calls == 2
    assert sleeps == [0.25, 1, 0.25]
    with GFSClient(
        transport=httpx.MockTransport(respond), request_interval=0, max_total_bytes=10
    ) as client:
        with pytest.raises(WeatherError, match="total transfer"):
            client.fetch_forecast_run(REFERENCE, steps=[25])
    with GFSClient(
        transport=httpx.MockTransport(respond), request_interval=0, max_message_bytes=5
    ) as client:
        with pytest.raises(WeatherError, match="message bound"):
            client.fetch_forecast_run(REFERENCE, steps=[25])


def test_source_change_between_ranges_is_an_error() -> None:
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        response = respond(request)
        if "range" in request.headers:
            count += 1
            response.headers["etag"] = str(count)
        return response

    with GFSClient(
        transport=httpx.MockTransport(handler), decoder=fake_decode, request_interval=0
    ) as client:
        with pytest.raises(WeatherError, match="changed"):
            client.fetch_forecast_run(REFERENCE, steps=[25])


@pytest.mark.parametrize(
    "steps", [[], [0], [121], [25, 24], [24, 26], [25, 25], list(range(1, 51))]
)
def test_retrieval_bounds_before_any_network(steps: list[int]) -> None:
    with pytest.raises(WeatherError):
        retrieval_identity(REFERENCE, steps)


def test_cache_identity_includes_run_steps_variables_grid_transform() -> None:
    identity = retrieval_identity(REFERENCE, [24, 25])
    assert digest(identity) != digest(retrieval_identity("2025-06-10T06Z", [24, 25]))
    assert digest(identity) != digest(retrieval_identity(REFERENCE, [25, 26]))
    for key in ("variables", "grid", "transform", "product"):
        altered = dict(identity)
        altered[key] = "different"
        assert digest(identity) != digest(altered)


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None, reason="optional weather Parquet extra"
)
def test_cache_roundtrip_is_offline_and_detects_corruption(tmp_path: Path) -> None:
    cache = WeatherCache(tmp_path)
    with GFSClient(
        cache=cache, transport=httpx.MockTransport(respond), decoder=fake_decode, request_interval=0
    ) as client:
        run = client.fetch_forecast_run(REFERENCE, steps=[24, 25])
    identity = retrieval_identity(REFERENCE, [24, 25])
    loaded = cache.load(identity)
    assert loaded is not None
    assert_frame_equal(loaded.data, run.data)
    assert loaded.version == run.version
    before = (tmp_path / digest(identity) / "manifest.json").read_bytes()
    cache.save(identity, run)
    assert (tmp_path / digest(identity) / "manifest.json").read_bytes() == before
    metadata = run.metadata
    metadata["sources"][0]["index_sha256"] = "different source version"
    version = digest({"identity": identity, "sources": metadata["sources"]})
    from solarcast.data.weather import ForecastRun

    conflicting = ForecastRun(REFERENCE, run.data, version=version, metadata=metadata)
    with pytest.raises(WeatherError, match="immutable cache"):
        cache.save(identity, conflicting)

    def no_network(_: httpx.Request) -> httpx.Response:
        raise AssertionError("cache hit must not fetch source")

    with GFSClient(cache=cache, transport=httpx.MockTransport(no_network)) as client:
        assert client.fetch_forecast_run(REFERENCE, steps=[24, 25]).version == run.version
    entry = tmp_path / digest(identity)
    manifest_path = entry / "manifest.json"
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest["identity"]["steps"] = [25, 26]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(WeatherError, match="identity"):
        cache.load(identity)
    manifest_path.write_text(original)
    with (entry / "native.parquet").open("ab") as file:
        file.write(b"corrupt")
    with pytest.raises(WeatherError, match="checksum"):
        cache.load(identity)


def test_partial_cache_entry_fails_explicitly(tmp_path: Path) -> None:
    identity = retrieval_identity(REFERENCE, [25])
    (tmp_path / digest(identity)).mkdir()
    with pytest.raises(WeatherError, match="invalid weather cache"):
        WeatherCache(tmp_path).load(identity)


@pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None or importlib.util.find_spec("pyproj") is None,
    reason="optional spatial extra",
)
def test_pinned_boundary_and_area_weights() -> None:
    weights = belgium_weights()
    assert len(weights.expected_cells) == 209
    assert len(weights.table) == 89
    assert weights.table.area_m2.sum() == pytest.approx(30_669_780_726.17, rel=1e-8)
    assert weights.identity == belgium_weights().identity
    raw = Path("src/solarcast/data/weather/belgium.geojson").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BOUNDARY_SHA256


def test_grib_rejects_unbounded_or_partial_messages_before_decode() -> None:
    if importlib.util.find_spec("eccodes") is None:
        with pytest.raises(ImportError, match="weather"):
            decode_field(b"", "TMP", REFERENCE, 25, 25)
    else:
        with pytest.raises(WeatherError, match="complete GRIB2"):
            decode_field(b"GRIB" + b"0" * 100, "TMP", REFERENCE, 25, 25)


def test_missing_value_is_not_turned_into_zero_by_downloader() -> None:
    def decoder(
        raw: bytes, variable: str, reference: pd.Timestamp, start: int, end: int
    ) -> pd.DataFrame:
        result = fake_decode(raw, variable, reference, start, end)
        if variable == "TMP":
            result.loc[0, "value"] = np.nan
        return result

    with GFSClient(
        transport=httpx.MockTransport(respond), decoder=decoder, request_interval=0
    ) as client:
        run = client.fetch_forecast_run(REFERENCE, steps=[25])
    assert run.data.temperature_2m_k.isna().sum() == 1


@pytest.mark.skipif(
    importlib.util.find_spec("eccodes") is None, reason="optional GRIB decoder extra"
)
@pytest.mark.parametrize("variable,value", [("DSWRF", 150.0), ("TCDC", 40.0), ("TMP", 293.15)])
def test_real_offline_grib_metadata_units_grid_and_values(variable: str, value: float) -> None:
    raw = Path(f"tests/fixtures/weather/constant_{variable}.grib2").read_bytes()
    start = 36 if variable == "DSWRF" else 38
    decoded = decode_field(raw, variable, REFERENCE, start, 38)
    assert len(decoded) == 209
    assert decoded.value.to_numpy() == pytest.approx(np.full(209, value), abs=1e-4)
    with pytest.raises(WeatherError, match="dataTime"):
        decode_field(raw, variable, REFERENCE + pd.Timedelta(hours=6), start, 38)
    with pytest.raises(WeatherError, match="startStep"):
        decode_field(raw, variable, REFERENCE, 35, 38)
    other = "TCDC" if variable == "TMP" else "TMP"
    with pytest.raises(WeatherError):
        decode_field(raw, other, REFERENCE, start, 38)


@pytest.mark.skipif(
    importlib.util.find_spec("eccodes") is None, reason="optional GRIB decoder extra"
)
def test_real_grib_bitmap_missing_cell_stays_nan() -> None:
    ec = importlib.import_module("eccodes")
    raw = Path("tests/fixtures/weather/constant_TMP.grib2").read_bytes()
    handle = ec.codes_new_from_message(raw)
    try:
        ec.codes_set(handle, "bitmapPresent", 1)
        ec.codes_set(handle, "missingValue", 9999.0)
        values = np.full(1440 * 721, 293.15)
        values[158 * 1440 + 12] = 9999.0  # 50.5 N, 3 E on north-to-south regular grid.
        ec.codes_set_values(handle, values)
        message = ec.codes_get_message(handle)
    finally:
        ec.codes_release(handle)
    decoded = decode_field(message, "TMP", REFERENCE, 38, 38)
    assert decoded.value.isna().sum() == 1
    assert (
        decoded.loc[(decoded.latitude == 50.5) & (decoded.longitude == 3.0), "value"].isna().all()
    )
