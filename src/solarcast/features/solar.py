"""Deterministic representative-point geometry and a simple clear-sky reference."""

import math
from dataclasses import dataclass
from typing import cast

import pandas as pd
from pvlib import clearsky, solarposition

from solarcast.features._validation import target_index


@dataclass(frozen=True)
class SolarLocation:
    """Representative point; altitude is a computational assumption, not fleet metadata."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.latitude_deg) or not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude_deg must be finite and within [-90, 90]")
        if not math.isfinite(self.longitude_deg) or not -180 <= self.longitude_deg <= 180:
            raise ValueError("longitude_deg must be finite and within [-180, 180]")
        if not math.isfinite(self.altitude_m):
            raise ValueError("altitude_m must be finite")


# Project-selected rounded point near Brussels; NOT a national PV-weighted centroid.
# Sea-level altitude is an explicit reference assumption, not surveyed elevation.
BELGIUM_REFERENCE_LOCATION = SolarLocation(latitude_deg=50.85, longitude_deg=4.35)


def solar_features(
    times: pd.DatetimeIndex, *, location: SolarLocation = BELGIUM_REFERENCE_LOCATION
) -> pd.DataFrame:
    """Return apparent angles at interval start, daylight, and Haurwitz GHI.

    NREL SPA uses fixed 12 C reference temperature and pressure derived from the
    configured altitude. These are deterministic refraction assumptions, not weather
    observations. Daylight means apparent elevation > 0 degrees at this point.
    """
    utc = target_index(times)
    # Avoid algorithm-specific edge cases on empty input while retaining dtypes.
    if utc.empty:
        position = pd.DataFrame(
            index=utc,
            columns=["apparent_elevation", "apparent_zenith", "azimuth"],
            dtype="float64",
        )
        ghi = pd.Series(index=utc, dtype="float64")
    else:
        position = cast(
            pd.DataFrame,
            solarposition.get_solarposition(
                utc,
                latitude=location.latitude_deg,
                longitude=location.longitude_deg,
                altitude=location.altitude_m,
                method="nrel_numpy",
                temperature=12.0,
            ),
        )
        clear = cast(pd.DataFrame, clearsky.haurwitz(position["apparent_zenith"]))
        ghi = clear["ghi"]
    return pd.DataFrame(
        {
            "target_time": utc,
            "solar_apparent_elevation_deg": position["apparent_elevation"].to_numpy(),
            "solar_apparent_zenith_deg": position["apparent_zenith"].to_numpy(),
            "solar_azimuth_deg": position["azimuth"].to_numpy(),
            "is_daylight": (position["apparent_elevation"] > 0).to_numpy(),
            "clearsky_ghi_w_m2": ghi.to_numpy(),
        }
    )
