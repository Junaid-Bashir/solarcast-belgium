"""Synthetic schema fixtures and mocked HTTP: the suite never contacts Elia."""

import logging
from datetime import UTC, datetime

import httpx
import pandas as pd
import pytest

from solarcast.data import EliaAPIError, EliaPVClient, EliaSchemaError, PVDataValidationError
from solarcast.data._schema import CURRENT, OLD, TRANSITION, Source, normalize
from solarcast.data.validation import validate_history

RETRIEVED = datetime(2026, 11, 13, tzinfo=UTC)


def row(stamp: str, source: Source = OLD, value: object = 12.5) -> dict[str, object]:
    """Synthetic values; field names follow the documented Elia schema."""
    record: dict[str, object] = {
        "datetime": stamp,
        "region": "Belgium",
        source.target_field: value,
        "monitoredcapacity": 100.0,
    }
    record["gridconnectiontype" if source.grid_total else "resolutioncode"] = (
        "Total" if source.grid_total else "PT15M"
    )
    return record


def respond(records: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"total_count": len(records), "results": records})


@pytest.mark.parametrize("source", [OLD, CURRENT])
def test_mapping_nulls_offsets_and_provenance(source: Source) -> None:
    records = [
        row("2026-10-26T01:15:00+02:00", source, None),
        row("2026-10-26T01:00:00+02:00", source, 0),
    ]
    records[0]["monitoredcapacity"] = None
    frame = normalize(records, source, RETRIEVED)
    assert list(frame.columns) == [
        "target_time",
        "pv_generation_mw",
        "monitored_capacity_mw",
        "source_dataset",
        "source_target_field",
        "retrieved_at",
    ]
    assert str(frame.target_time.dtype) == "datetime64[ns, UTC]"
    assert str(frame.pv_generation_mw.dtype) == "Float64"
    assert str(frame.monitored_capacity_mw.dtype) == "Float64"
    assert str(frame.retrieved_at.dtype) == "datetime64[ns, UTC]"
    assert frame.target_time.iloc[0] == pd.Timestamp(TRANSITION)
    assert frame.pv_generation_mw.iloc[0] == 0
    assert pd.isna(frame.pv_generation_mw.iloc[1])
    assert pd.isna(frame.monitored_capacity_mw.iloc[1])
    assert set(frame.source_dataset) == {source.dataset}
    assert set(frame.source_target_field) == {source.target_field}
    assert (frame.retrieved_at == RETRIEVED).all()
    assert "available_at" not in frame


@pytest.mark.parametrize("field,value", [("region", "Hainaut"), ("gridconnectiontype", "DSO")])
def test_rejects_non_national_total_rows(field: str, value: str) -> None:
    record = row("2026-10-26T00:00Z", CURRENT)
    record[field] = value
    with pytest.raises(EliaSchemaError, match="national filter"):
        normalize([record], CURRENT, RETRIEVED)


@pytest.mark.parametrize(
    "field,value",
    [
        ("datetime", "2025-01-01T00:00"),
        ("datetime", None),
        ("resolutioncode", "PT60M"),
        ("measured", "1.5"),
        ("measured", True),
    ],
)
def test_schema_failures(field: str, value: object) -> None:
    record = row("2025-01-01T00:00Z")
    record[field] = value
    with pytest.raises(EliaSchemaError):
        normalize([record], OLD, RETRIEVED)


def test_missing_field_is_not_treated_as_null() -> None:
    record = row("2025-01-01T00:00Z")
    del record["measured"]
    with pytest.raises(EliaSchemaError, match="missing fields.*measured"):
        normalize([record], OLD, RETRIEVED)


@pytest.mark.parametrize("field", ["measured", "monitoredcapacity"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_fail_before_nullable_conversion(field: str, value: float) -> None:
    record = row("2025-01-01T00:00Z")
    record[field] = value
    with pytest.raises(PVDataValidationError, match="non-finite"):
        normalize([record], OLD, RETRIEVED)


def test_negative_and_missing_values_are_preserved_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = normalize(
        [row("2025-01-01T00:00Z", value=-1), row("2025-01-01T00:15Z", value=None)], OLD, RETRIEVED
    )
    with caplog.at_level(logging.WARNING):
        validate_history(
            frame, pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2025-01-01T00:45Z")
        )
    assert frame.pv_generation_mw.iloc[0] == -1
    assert pd.isna(frame.pv_generation_mw.iloc[1])
    assert "negative values preserved" in caplog.text
    assert "null values preserved" in caplog.text
    assert "1 missing interval rows" in caplog.text


@pytest.mark.parametrize(
    "stamps,match",
    [
        (["2025-01-01T00:00Z", "2025-01-01T00:00Z"], "Duplicate"),
        (["2025-01-01T00:01Z"], "aligned"),
        (["2025-01-02T00:00Z"], "outside"),
    ],
)
def test_structural_validation(stamps: list[str], match: str) -> None:
    frame = normalize([row(s) for s in stamps], OLD, RETRIEVED)
    with pytest.raises(PVDataValidationError, match=match):
        validate_history(
            frame, pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2025-01-02T00:00Z")
        )


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ("2025-01-01T00:00Z", "2025-01-01T00:15Z", [OLD.dataset]),
        ("2026-10-25T22:45Z", "2026-10-25T23:00Z", [OLD.dataset]),
        ("2026-10-25T23:00Z", "2026-10-25T23:15Z", [CURRENT.dataset]),
        ("2026-10-25T22:45Z", "2026-10-25T23:15Z", [OLD.dataset, CURRENT.dataset]),
    ],
)
def test_routing_and_query_contract(start: str, end: str, expected: list[str]) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        source = CURRENT if CURRENT.dataset in request.url.path else OLD
        seen.append(source.dataset)
        params = request.url.params
        assert params["select"] == source.fields
        assert params["order_by"] == "datetime"
        assert params["limit"] == "100"
        assert params["offset"] == "0"
        assert source.selection in params["where"]
        assert "forecast" not in params["select"]
        if source == OLD:
            assert "gridconnectiontype" not in params["where"]
        stamp = (
            pd.Timestamp(start)
            if source == OLD
            else max(pd.Timestamp(start), pd.Timestamp(TRANSITION))
        )
        return respond([row(stamp.isoformat(), source)])

    with EliaPVClient(transport=httpx.MockTransport(handler), request_interval=0) as client:
        frame = client.fetch_history(start=start, end=end)
    assert seen == expected
    assert list(frame.source_dataset) == expected


def test_adjacent_utc_windows_are_half_open_and_do_not_duplicate() -> None:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        where = request.url.params["where"]
        first, stop = [pd.Timestamp(part.split("'")[0]) for part in where.split("date'")[1:]]
        windows.append((first, stop))
        return respond(
            [row(t.isoformat()) for t in pd.date_range(first, stop, freq="15min", inclusive="left")]
        )

    with EliaPVClient(transport=httpx.MockTransport(handler), request_interval=0) as client:
        frame = client.fetch_history(start="2025-01-01T00:00Z", end="2025-01-03T00:00Z")
    assert len(windows) == 2
    assert windows[0][1] == windows[1][0]
    assert len(frame) == 192
    assert frame.target_time.is_unique
    assert frame.target_time.iloc[-1] == pd.Timestamp("2025-01-02T23:45Z")


@pytest.mark.parametrize(
    "day,next_day,count",
    [
        ("2026-03-29", "2026-03-30", 92),
        ("2026-10-25", "2026-10-26", 100),
    ],
)
def test_dst_local_day_ingestion(day: str, next_day: str, count: int) -> None:
    start = pd.Timestamp(day, tz="Europe/Brussels")
    end = pd.Timestamp(next_day, tz="Europe/Brussels")

    def handler(request: httpx.Request) -> httpx.Response:
        where = request.url.params["where"]
        lower, upper = [pd.Timestamp(p.split("'")[0]) for p in where.split("date'")[1:]]
        stamps = pd.date_range(lower, upper, freq="15min", inclusive="left")
        return respond([row(t.tz_convert("Europe/Brussels").isoformat()) for t in stamps])

    with EliaPVClient(transport=httpx.MockTransport(handler), request_interval=0) as client:
        frame = client.fetch_history(start=start, end=end)
    assert len(frame) == count
    assert frame.target_time.is_unique
    assert str(frame.target_time.dt.tz) == "UTC"


@pytest.mark.parametrize("mode", ["gap_old", "gap_new", "overlap", "wrong_side"])
def test_seam_failures_are_not_silently_resolved(mode: str) -> None:
    before = normalize([row("2026-10-25T22:45Z")], OLD, RETRIEVED)
    after = normalize([row("2026-10-25T23:00Z", CURRENT)], CURRENT, RETRIEVED)
    if mode == "gap_old":
        before = before.iloc[:0]
    elif mode == "gap_new":
        after = after.iloc[:0]
    elif mode == "overlap":
        after.loc[0, "target_time"] = before.target_time.iloc[0]
    else:
        before.loc[0, "target_time"] = pd.Timestamp(TRANSITION) + pd.Timedelta(minutes=15)
    frame = pd.concat([before, after], ignore_index=True).sort_values("target_time")
    with pytest.raises(PVDataValidationError):
        validate_history(
            frame, pd.Timestamp("2026-10-25T22:45Z"), pd.Timestamp("2026-10-25T23:30Z")
        )


@pytest.mark.parametrize(
    "start,end",
    [
        ("2025-01-01", "2025-01-02"),
        (datetime(2025, 1, 1), datetime(2025, 1, 2, tzinfo=UTC)),
        ("2025-01-01T00:01Z", "2025-01-02T00:00Z"),
        ("2025-01-01T00:00Z", "2025-01-01T00:00Z"),
        ("2025-01-02T00:00Z", "2025-01-01T00:00Z"),
    ],
)
def test_invalid_input_does_not_contact_network(start: str | datetime, end: str | datetime) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid arguments must be rejected before HTTP")

    with EliaPVClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            client.fetch_history(start=start, end=end)


def test_empty_response_has_stable_dtypes(caplog: pytest.LogCaptureFixture) -> None:
    with EliaPVClient(transport=httpx.MockTransport(lambda r: respond([]))) as client:
        frame = client.fetch_history(start="2025-01-01T00:00Z", end="2025-01-01T00:15Z")
    assert frame.empty
    assert str(frame.target_time.dtype) == "datetime64[ns, UTC]"
    assert str(frame.pv_generation_mw.dtype) == "Float64"
    assert "missing interval rows" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"total_count": 1, "results": []},
        {"total_count": True, "results": []},
        {"total_count": 1, "results": [1]},
    ],
)
def test_invalid_response_envelope(payload: object) -> None:
    with EliaPVClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(EliaAPIError, match="ods032"):
            client.fetch_history(start="2025-01-01T00:00Z", end="2025-01-01T00:15Z")


def test_oversized_response_fails_instead_of_paginating() -> None:
    payload = {"total_count": 97, "results": []}
    with EliaPVClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(PVDataValidationError, match="more than 96"):
            client.fetch_history(start="2025-01-01T00:00Z", end="2025-01-02T00:00Z")


@pytest.mark.parametrize("failure", ["429", "503", "timeout"])
def test_transient_failures_retry_then_succeed(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("solarcast.data.elia.time.sleep", sleeps.append)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("synthetic timeout", request=request)
            return httpx.Response(int(failure), headers={"Retry-After": "2"})
        return respond([row("2025-01-01T00:00Z")])

    with EliaPVClient(transport=httpx.MockTransport(handler), request_interval=0) as client:
        assert len(client.fetch_history(start="2025-01-01T00:00Z", end="2025-01-01T00:15Z")) == 1
    assert attempts == 2
    assert sleeps == ([1.0] if failure == "timeout" else [2.0])


@pytest.mark.parametrize("status,expected", [(503, 3), (400, 1), (401, 1)])
def test_retries_are_bounded_and_permanent_errors_fail_immediately(
    status: int,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("solarcast.data.elia.time.sleep", lambda delay: None)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status)

    with EliaPVClient(transport=httpx.MockTransport(handler), request_interval=0) as client:
        with pytest.raises(EliaAPIError, match=f"HTTP {status}"):
            client.fetch_history(start="2025-01-01T00:00Z", end="2025-01-01T00:15Z")
    assert attempts == expected


def test_long_retry_after_stops_without_retrying_early(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_sleep(delay: float) -> None:
        pytest.fail("Must stop rather than shorten the server's wait")

    monkeypatch.setattr("solarcast.data.elia.time.sleep", no_sleep)
    with EliaPVClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(429, headers={"Retry-After": "120"}))
    ) as client:
        with pytest.raises(EliaAPIError, match="retry wait budget"):
            client.fetch_history(start="2025-01-01T00:00Z", end="2025-01-01T00:15Z")


def test_invalid_json_has_useful_context() -> None:
    with EliaPVClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="not JSON"))
    ) as client:
        with pytest.raises(EliaAPIError, match="ods032.*not valid JSON"):
            client.fetch_history(start="2025-01-01T00:00Z", end="2025-01-01T00:15Z")
