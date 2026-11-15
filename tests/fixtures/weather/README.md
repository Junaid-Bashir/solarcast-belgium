# Offline GFS fixtures

`gfs_20250610_00_f036_f038.json` contains **real** NOAA/NCEP operational GFS
0.25° values at three cells and three native times from the bounded the historical GFS ingestion layer
smoke. Reference 2025-06-10 00Z; retrieved 2026-11-14 from the public NODD
`noaa-gfs-bdp-pds` bucket, `gfs.20250610/00/atmos/`. The source-version hash refers
to the full cropped smoke bundle, not a certified historical as-of snapshot.
Native DSWRF averaging bounds are preserved. Publication availability is unknown.

`constant_{DSWRF,TCDC,TMP}.grib2` are **modified synthetic test data**, not NOAA
forecasts. They preserve the f038 messages' actual GRIB metadata and global grid,
but replace every value with respectively 150 W/m², 40%, and 293.15 K, repacked
with ecCodes `packingType=grid_simple` and `codes_set_values` on all 1440×721 cells.
Constant packing makes each file only 179–203 bytes. They exercise the real
decoder offline, including reference/step/parameter/unit/grid validation.

NOAA source attribution does not imply NOAA endorsement. No credentials or large
raw weather files are included. Other unit tests construct explicit three-cell
normalized fixtures with six-hour running-average windows and hand-checkable values.
