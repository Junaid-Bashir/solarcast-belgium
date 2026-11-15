"""Pinned Belgian land intersection areas; no PV capacity assumptions."""

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from importlib.resources import files

import numpy as np
import pandas as pd

from .types import GRID, TRANSFORM, WeatherError

BOUNDARY_SHA256 = "bd49151ec45a9b42c63140ce908294c6e34e215583fbf36dbd61e51ec60b3286"


@dataclass(frozen=True, init=False, eq=False)
class SpatialWeights:
    """Positive Belgian intersection areas on an explicit expected native grid."""

    identity: str
    expected_cells: tuple[tuple[float, float], ...]
    _table: pd.DataFrame = field(repr=False)

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        expected_cells: tuple[tuple[float, float], ...],
        boundary_id: str,
    ) -> None:
        if set(table.columns) != {"latitude", "longitude", "area_m2"}:
            raise WeatherError("weights require latitude, longitude, area_m2")
        frame = table.astype(float).sort_values(["latitude", "longitude"]).reset_index(drop=True)
        if (
            frame.empty
            or frame.duplicated(["latitude", "longitude"]).any()
            or not np.isfinite(frame.to_numpy()).all()
            or frame.area_m2.le(0).any()
        ):
            raise WeatherError("invalid/duplicate spatial weights")
        cells = tuple(sorted(expected_cells))
        if len(cells) != len(set(cells)) or not set(
            zip(frame.latitude, frame.longitude, strict=True)
        ) <= set(cells):
            raise WeatherError("weights differ from expected grid")
        for mask in (
            frame.longitude < 4,
            frame.longitude.between(4, 5, inclusive="left"),
            frame.longitude >= 5,
        ):
            if not mask.any():
                raise WeatherError("each longitude band needs positive Belgian area")
        payload = {
            "boundary": boundary_id,
            "transform": TRANSFORM,
            "cells": cells,
            "weights": frame.to_dict(orient="records"),
            "area_crs": "EPSG:3035",
        }
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "expected_cells", cells)
        object.__setattr__(self, "_table", frame)

    @property
    def table(self) -> pd.DataFrame:
        return self._table.copy(deep=True)


def belgium_weights() -> SpatialWeights:
    """Intersect 0.25° cell footprints with Natural Earth 10m Belgium v5.1.2.

    Project each intersection to equal-area EPSG:3035 before measuring. Boundary
    is public domain, cartographic (not cadastral); hash pinned in package.
    """
    try:
        geometry = importlib.import_module("shapely.geometry")
        shapely = importlib.import_module("shapely")
        pyproj = importlib.import_module("pyproj")
    except ImportError as error:
        raise ImportError("Install solarcast-belgium[weather] for spatial weights") from error
    raw = files(__package__).joinpath("belgium.geojson").read_bytes()
    if hashlib.sha256(raw).hexdigest() != BOUNDARY_SHA256:
        raise WeatherError("pinned Belgium boundary checksum mismatch")
    boundary = geometry.shape(json.loads(raw)["geometry"])
    if not boundary.is_valid:
        raise WeatherError("invalid pinned Belgium polygon")
    project = pyproj.Transformer.from_crs(4326, 3035, always_xy=True).transform
    rows = []
    for latitude, longitude in GRID.cells:
        half = GRID.spacing / 2
        intersection = boundary.intersection(
            geometry.box(longitude - half, latitude - half, longitude + half, latitude + half)
        )
        area = float(shapely.transform(intersection, project, interleaved=False).area)
        if area > 0:
            rows.append({"latitude": latitude, "longitude": longitude, "area_m2": area})
    return SpatialWeights(
        pd.DataFrame(rows), expected_cells=GRID.cells, boundary_id=BOUNDARY_SHA256
    )
