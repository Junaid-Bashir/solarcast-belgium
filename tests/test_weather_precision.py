"""Pretraining GRIB-precision correction: default strict behavior stays unchanged."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solarcast.data.weather import ForecastRun, WeatherError, hourly_radiation
from solarcast.data.weather._grib import decode_field


def run_with_quantum(value: float = 49.99, quantum: float = 0.02) -> ForecastRun:
    fixture = json.loads(
        Path("tests/fixtures/weather/gfs_20250610_00_f036_f038.json").read_text(encoding="utf8")
    )
    frame = pd.DataFrame(fixture["data"])
    for name in ("reference_time", "valid_time"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    frame["forecast_lead"] = pd.to_timedelta(frame.forecast_lead)
    frame.loc[frame.dswrf_end_hour == 37, "dswrf_average_w_m2"] = 100.0
    frame.loc[frame.dswrf_end_hour == 38, "dswrf_average_w_m2"] = value
    metadata = {
        "provider": fixture["provider"],
        "product": fixture["product"],
        "sources": [
            {
                "fields": [
                    {
                        "variable": "DSWRF",
                        "end_hour": step,
                        "packing": {
                            "packing_type": "grid_complex_spatial_differencing",
                            "quantum_w_m2": quantum,
                        },
                    }
                    for step in (36, 37, 38)
                ]
            }
        ],
    }
    return ForecastRun(
        fixture["reference_time"], frame, version="precision-fixture", metadata=metadata
    )


def test_grib_bound_propagates_quantisation_and_records_every_adjustment() -> None:
    run = run_with_quantum()
    with pytest.raises(WeatherError, match="negative"):
        hourly_radiation(run)
    aligned = hourly_radiation(run, radiation_policy="grib_packing_bound_v1")
    ending38 = aligned.loc[aligned.dswrf_end_hour == 38]
    assert ending38.ghi_hourly_w_m2.eq(0).all()
    assert ending38.radiation_rounding_adjusted.all()
    assert ending38.radiation_rounding_bound_w_m2.to_numpy() == pytest.approx(np.full(3, 0.06))
    assert aligned.loc[aligned.dswrf_end_hour == 37, "ghi_hourly_w_m2"].eq(100).all()
    assert aligned.radiation_rounding_adjusted.sum() == 3
    assert run.data.loc[run.data.dswrf_end_hour == 38, "dswrf_average_w_m2"].eq(49.99).all()


def test_grib_bound_still_rejects_large_negative_and_unknown_evidence() -> None:
    with pytest.raises(WeatherError, match="negative"):
        hourly_radiation(run_with_quantum(40.0), radiation_policy="grib_packing_bound_v1")
    run = run_with_quantum()
    metadata = run.metadata
    metadata.pop("sources")
    missing = ForecastRun(run.reference_time, run.data, version="unknown", metadata=metadata)
    with pytest.raises(WeatherError, match="evidence"):
        hourly_radiation(missing, radiation_policy="grib_packing_bound_v1")


@pytest.mark.skipif(
    importlib.util.find_spec("eccodes") is None, reason="optional GRIB decoder extra"
)
def test_decoder_preserves_radiation_packing_evidence() -> None:
    raw = Path("tests/fixtures/weather/constant_DSWRF.grib2").read_bytes()
    decoded = decode_field(raw, "DSWRF", pd.Timestamp("2025-06-10T00Z"), 36, 38)
    assert decoded.attrs["packing"]["packing_type"] == "grid_simple"
    assert decoded.attrs["packing"]["bits_per_value"] == 0
    assert decoded.attrs["packing"]["quantum_w_m2"] == 0
