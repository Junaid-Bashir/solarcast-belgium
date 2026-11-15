"""Optional ecCodes boundary: validate the actual message, then crop its grid."""

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .types import GRID, WeatherError


def decode_field(
    raw: bytes, variable: str, reference: pd.Timestamp, start: int, end: int
) -> pd.DataFrame:
    """Decode exactly one GRIB2 message. Global messages never enter the cache."""
    try:
        ec = importlib.import_module("eccodes")
    except ImportError as error:
        raise ImportError("Install solarcast-belgium[weather] to decode GFS GRIB2") from error
    if (
        len(raw) < 20
        or raw[:4] != b"GRIB"
        or raw[7] != 2
        or raw[-4:] != b"7777"
        or int.from_bytes(raw[8:16], "big") != len(raw)
    ):
        raise WeatherError("range is not exactly one complete GRIB2 message")
    handle = ec.codes_new_from_message(raw)
    try:

        def get(key: str) -> Any:
            return ec.codes_get(handle, key)

        common = {
            "centre": "kwbc",
            "productionStatusOfProcessedData": 0,
            "typeOfProcessedData": "fc",
            "discipline": 0,
            "dataDate": int(reference.strftime("%Y%m%d")),
            "dataTime": reference.hour * 100,
            "startStep": start,
            "endStep": end,
            "stepUnits": 1,
            "gridType": "regular_ll",
            "Ni": 1440,
            "Nj": 721,
            "iDirectionIncrementInDegrees": 0.25,
            "jDirectionIncrementInDegrees": 0.25,
        }
        valid = reference + pd.Timedelta(hours=end)
        common.update(
            {"validityDate": int(valid.strftime("%Y%m%d")), "validityTime": valid.hour * 100}
        )
        if variable == "DSWRF":
            expected = {
                "parameterCategory": 4,
                "stepType": "avg",
                "units": "W m**-2",
                "typeOfFirstFixedSurface": "sfc",
                "level": 0,
            }
            # NOAA uses local 192 in probed operational files; standard 7 is equivalent.
            if get("parameterNumber") not in (7, 192):
                raise WeatherError("unexpected DSWRF parameter number")
        elif variable == "TCDC":
            expected = {
                "parameterCategory": 6,
                "parameterNumber": 1,
                "stepType": "instant",
                "units": "%",
                "typeOfFirstFixedSurface": "10",
                "level": 0,
            }
        elif variable == "TMP":
            expected = {
                "parameterCategory": 0,
                "parameterNumber": 0,
                "stepType": "instant",
                "units": "K",
                "typeOfFirstFixedSurface": "sfc",
                "level": 2,
                "typeOfLevel": "heightAboveGround",
            }
        else:
            raise WeatherError("unsupported GFS variable")
        for key, value in (common | expected).items():
            if get(key) != value:
                raise WeatherError(f"GRIB {variable}.{key}: expected {value!r}, got {get(key)!r}")
        latitude = np.asarray(ec.codes_get_array(handle, "latitudes"), dtype=float)
        longitude = np.asarray(ec.codes_get_array(handle, "longitudes"), dtype=float)
        values = np.asarray(ec.codes_get_array(handle, "values"), dtype=float)
        if get("bitmapPresent"):
            bitmap = np.asarray(ec.codes_get_array(handle, "bitmap"), dtype=bool)
            values[~bitmap] = np.nan
        mask = (
            (latitude >= GRID.south)
            & (latitude <= GRID.north)
            & (longitude >= GRID.west)
            & (longitude <= GRID.east)
        )
        subset = pd.DataFrame(
            {"latitude": latitude[mask], "longitude": longitude[mask], "value": values[mask]}
        ).sort_values(["latitude", "longitude"])
        if tuple(zip(subset.latitude, subset.longitude, strict=True)) != GRID.cells:
            raise WeatherError("decoded GFS Belgium grid does not match 209 expected centres")
        subset = subset.reset_index(drop=True)
        if variable == "DSWRF":
            binary, decimal = int(get("binaryScaleFactor")), int(get("decimalScaleFactor"))
            bits = int(get("bitsPerValue"))
            subset.attrs["packing"] = {
                "packing_type": str(get("packingType")),
                "binary_scale_factor": binary,
                "decimal_scale_factor": decimal,
                "bits_per_value": bits,
                "reference_value": float(get("referenceValue")),
                "quantum_w_m2": 0.0 if bits == 0 else 2.0**binary * 10.0 ** (-decimal),
            }
        return subset
    finally:
        ec.codes_release(handle)
