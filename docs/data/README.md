# Data sources and local storage

Elia Open Data is the initial source for Belgian aggregated PV generation.
See the [verified Elia investigation](elia.md) for dataset versions, field mappings,
national filtering, interval-start semantics, DST samples and remaining uncertainties.
Historical fetching/normalization is implemented; see the usage section in that
document. It preserves source provenance and does not infer availability times.

Before implementing an adapter, verify official documentation for target meaning,
coverage, interval labels, timezone, units, missing values, publication latency,
revisions, access limits and licensing/redistribution terms. Record the source and
verification date. Do not infer historical availability from download time.

Use root `data/` for local datasets, `model_weights/` for downloaded weights and
`artifacts/` for generated results. These are ignored by Git; create them only as
needed. Weather storage uses cropped-native Parquet with JSON source provenance.

The [weather investigation and implemented contract](weather.md) select archived
operational GFS 0.25° forecasts for the first weather track, with ECMWF alternatives
and access/provenance limits documented. Bounded ingestion, local cache and optional
forecast-context integration are implemented, along with the
[controlled weather-LightGBM comparison](../experiments/weather-lightgbm-v1.md). Preserve
forecast initialization, interval bounds and observed versus assumed availability.
Future observed weather, reanalysis and later hindcasts are not interchangeable
with operational historical forecasts. Software licensing does not grant rights
to redistribute external datasets or model weights.
