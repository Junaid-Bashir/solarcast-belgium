"""Variable-aware temporal alignment, then explicit spatial aggregation."""

import numpy as np
import pandas as pd

from .spatial import SpatialWeights
from .types import FEATURES, ForecastRun, WeatherError, targets_utc


def hourly_radiation(run: ForecastRun, *, radiation_policy: str = "strict") -> pd.DataFrame:
    """Reconstruct hourly means from same-start running averages, within one run.

    Missing predecessors stay NaN. Six-hour window resets use the first one-hour
    average directly. Strict zero negative tolerance: no implicit clipping.
    """
    frame = run.data
    if radiation_policy not in ("strict", "grib_packing_bound_v1"):
        raise WeatherError("unknown radiation precision policy")
    quantum: dict[int, float] = {}
    if radiation_policy != "strict":
        for source in run.metadata.get("sources", []):
            for selected in source["fields"]:
                if selected["variable"] == "DSWRF":
                    packing = selected.get("packing")
                    if packing is None or packing["packing_type"] not in (
                        "grid_simple",
                        "grid_complex",
                        "grid_complex_spatial_differencing",
                    ):
                        raise WeatherError(
                            "bounded radiation requires verified GRIB packing metadata"
                        )
                    q = float(packing["quantum_w_m2"])
                    if not np.isfinite(q) or q < 0:
                        raise WeatherError("invalid radiation packing quantum")
                    quantum[int(selected["end_hour"])] = q
        if not quantum:
            raise WeatherError("bounded radiation requires per-message packing evidence")
    frame["ghi_hourly_w_m2"] = np.nan
    frame["radiation_rounding_adjusted"] = False
    frame["radiation_rounding_bound_w_m2"] = 0.0
    for _, group in frame.groupby(["latitude", "longitude"], sort=False):
        previous: dict[int, tuple[int, float]] = {}
        for index, row in group.iterrows():
            start, end = int(row.dswrf_start_hour), int(row.dswrf_end_hour)
            value = float(row.dswrf_average_w_m2)
            duration = end - start
            flux = np.nan
            bound = 0.0
            if radiation_policy != "strict" and end not in quantum:
                raise WeatherError("missing per-step radiation packing evidence")
            if duration == 1:
                flux = value
                if radiation_policy != "strict":
                    bound = quantum[end]
            elif end - 1 in previous and previous[end - 1][0] == start:
                flux = duration * value - (duration - 1) * previous[end - 1][1]
                if radiation_policy != "strict":
                    if end - 1 not in quantum:
                        raise WeatherError("missing predecessor packing evidence")
                    bound = duration * quantum[end] + (duration - 1) * quantum[end - 1]
                    bound += (
                        8
                        * np.finfo(float).eps
                        * (abs(duration * value) + abs((duration - 1) * previous[end - 1][1]))
                    )
            if np.isfinite(flux) and flux < 0:
                if radiation_policy == "strict" or flux < -bound:
                    raise WeatherError(
                        "negative reconstructed radiation; inspect packing/source bounds"
                    )
                flux = 0.0
                frame.loc[index, "radiation_rounding_adjusted"] = True
            frame.loc[index, "radiation_rounding_bound_w_m2"] = bound
            frame.loc[index, "ghi_hourly_w_m2"] = flux
            previous[end] = (start, value)
    return frame


def aligned_cells(
    run: ForecastRun,
    targets: pd.DatetimeIndex,
    weights: SpatialWeights,
    *,
    radiation_policy: str = "strict",
) -> pd.DataFrame:
    """Repeat hourly flux; linearly integrate hourly states over each quarter hour."""
    times = targets_utc(targets)
    frame = hourly_radiation(run, radiation_policy=radiation_policy)
    indexed = {(row.valid_time, row.latitude, row.longitude): row for _, row in frame.iterrows()}
    rows: list[dict[str, object]] = []
    for target in times:
        left = target.floor("h")
        right = left + pd.Timedelta(hours=1)
        midpoint_fraction = ((target - left) / pd.Timedelta(hours=1)) + 0.125
        for latitude, longitude in weights.expected_cells:
            try:
                low = indexed[(left, latitude, longitude)]
                high = indexed[(right, latitude, longitude)]
            except KeyError as error:
                raise WeatherError(f"missing native endpoint/grid cell at {target}") from error
            ghi = float(high.ghi_hourly_w_m2)
            cloud = (
                float(low.cloud_percent) * (1 - midpoint_fraction)
                + float(high.cloud_percent) * midpoint_fraction
            ) / 100
            temperature = (
                float(low.temperature_2m_k) * (1 - midpoint_fraction)
                + float(high.temperature_2m_k) * midpoint_fraction
            ) - 273.15
            if not np.isfinite([ghi, cloud, temperature]).all():
                raise WeatherError(f"missing variable/radiation predecessor at {target}")
            rows.append(
                {
                    "target_time": target,
                    "latitude": latitude,
                    "longitude": longitude,
                    "ghi_w_m2": ghi,
                    "cloud_fraction": cloud,
                    "temperature_2m_c": temperature,
                }
            )
    result = pd.DataFrame(rows)
    result.attrs["radiation_adjusted_cell_hours"] = int(frame.radiation_rounding_adjusted.sum())
    result.attrs["radiation_policy"] = radiation_policy
    return result


def weather_features(
    run: ForecastRun,
    targets: pd.DatetimeIndex,
    weights: SpatialWeights,
    *,
    radiation_policy: str = "strict",
) -> pd.DataFrame:
    """Seven approved area-weighted features; missing expected cells fail first."""
    aligned = aligned_cells(run, targets, weights, radiation_policy=radiation_policy)
    merged = aligned.merge(
        weights.table, on=["latitude", "longitude"], how="inner", validate="many_to_one"
    )
    rows: list[dict[str, object]] = []
    for target, group in merged.groupby("target_time", sort=True):
        area = group.area_m2.to_numpy(dtype=float)
        ghi = group.ghi_w_m2.to_numpy(dtype=float)
        cloud = group.cloud_fraction.to_numpy(dtype=float)
        means = []
        for mask in (
            group.longitude < 4,
            group.longitude.between(4, 5, inclusive="left"),
            group.longitude >= 5,
        ):
            means.append(float(np.average(ghi[mask], weights=area[mask])))
        ghi_mean, cloud_mean = np.average(ghi, weights=area), np.average(cloud, weights=area)
        values = [
            *means,
            float(np.sqrt(np.average((ghi - ghi_mean) ** 2, weights=area))),
            float(cloud_mean),
            float(np.sqrt(np.average((cloud - cloud_mean) ** 2, weights=area))),
            float(np.average(group.temperature_2m_c, weights=area)),
        ]
        rows.append({"target_time": target, **dict(zip(FEATURES, values, strict=True))})
    result = pd.DataFrame(rows)
    result.attrs.update(aligned.attrs)
    return result
