# Third-party resources

SolarCast's [Apache-2.0 software license](LICENSE) does not relicense external
datasets, dependencies or model weights. This inventory identifies sources and
known limitations; it is not legal clearance for a particular deployment.
Official references were reviewed during development and rechecked where accessible
on 15 November 2026. Inspect terms for the exact versions you acquire.

| Resource | Project use | Terms and authoritative reference |
|---|---|---|
| Elia Open Data | National Belgian PV estimates, obtained separately | [Elia licence](https://opendata.elia.be/pages/licence/); live text remains unavailable in this review. Full applicability is unresolved; see [investigation](docs/data/elia.md#15-licensing-and-attribution). No large PV snapshots are distributed. |
| NOAA/NCEP GFS via NODD | Historical operational forecast subsets and small test fixtures | [Dataset terms](https://registry.opendata.aws/noaa-gfs-bdp-pds/): public use, NOAA attribution requested, no implied endorsement; identify modified data as modified. See fixture notices below. |
| Natural Earth v5.1.2 | Packaged Belgium-only geographic extraction for area weighting | [Public-domain terms](https://www.naturalearthdata.com/about/terms-of-use/). Source/version/license are embedded in `src/solarcast/data/weather/belgium.geojson`; extraction is modified geometry, not a fleet-capacity map. |
| LightGBM | Optional native regression dependency | [MIT license](https://github.com/microsoft/LightGBM/blob/master/LICENSE). Installed separately; no native library is bundled. |
| pvlib python | Core deterministic solar geometry and clear-sky reference | [BSD 3-Clause license](https://github.com/pvlib/pvlib-python/blob/main/LICENSE). Installed separately; no pvlib source is bundled. |
| Chronos forecasting library | Optional Chronos adapter runtime | [Apache-2.0 code license](https://github.com/amazon-science/chronos-forecasting/blob/main/LICENSE). |
| Chronos-2 weights | Optional pinned checkpoint; not bundled | [Pinned model card](https://huggingface.co/amazon/chronos-2/blob/29ec3766d36d6f73f0696f85560a422f50e8498c/README.md) declares Apache-2.0; no separate weight LICENSE was found during the documented model investigation. Preserve publisher attribution. |
| tfc-t0 code and t0-gamma weights | Independent optional runtime/checkpoint; not bundled | [Code license](https://github.com/theforecastingcompany/tfc-t0/blob/main/LICENSE), pinned [weight LICENSE](https://huggingface.co/theforecastingcompany/t0-gamma/blob/fef67abad3eed4f227c7b536a6e8ee4e9ab1080d/LICENSE) and [NOTICE](https://huggingface.co/theforecastingcompany/t0-gamma/blob/fef67abad3eed4f227c7b536a6e8ee4e9ab1080d/NOTICE). The earlier pinned-file investigation records Apache-2.0; the weight LICENSE could not be fetched again by this review's browser. |

## Intentional bundled non-code resources

- `tests/fixtures/weather/gfs_20250610_00_f036_f038.json` is a tiny cropped real
  NOAA/NCEP operational GFS subset, retrieved 14 November 2026. Source times and
  provenance are recorded in [fixture documentation](tests/fixtures/weather/README.md).
- The three `constant_*.grib2` fixtures are **modified synthetic values** repacked
  from provider message metadata, 179–203 bytes each. They are not original NOAA
  forecasts. NOAA attribution implies no endorsement.
- `src/solarcast/data/weather/belgium.geojson` is the documented Natural Earth
  Belgium extraction (about 15 KB).
- `docs/benchmarks/` contains SolarCast aggregate metric/reporting metadata only,
  and `docs/assets/` contains an intentionally published derived chart. These are
  not raw Elia observations, prediction caches or checkpoint weights. The Elia
  licensing caveat is retained; do not infer broader redistribution rights.

## Other dependencies

Core dependencies are HTTPX, pandas, NumPy, tzdata and pvlib. Optional preparation
uses ecCodes, PyArrow, Shapely and pyproj; model runtimes can add PyTorch, Hugging
Face Hub and other transitive dependencies. They are installed by the user, not
vendored or relicensed. Their installed distribution metadata and upstream license
files remain authoritative; see [declared dependency ranges](pyproject.toml).
The wheel contains SolarCast source, its small geography resource and its own
license/notices, not those dependency distributions. Reassess notices if binaries,
containers, datasets or weights are redistributed later.
