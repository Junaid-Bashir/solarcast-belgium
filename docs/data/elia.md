# Elia PV data contract investigation

Research verified on **2026-11-13**; historical ingestion is now implemented as
described in section 20. No storage or forecasting is implemented.
Sources are official Elia APIs/pages and
the platform provider's API reference. Findings below distinguish live observations,
documented semantics, recommendations, and unresolved questions.

## 1. Dataset overview

Use the national measured-and-upscaled PV estimate as the prospective target,
not an Elia forecast or capacity value. The history spans two schemas. Current
dataset metadata and small record queries were retrieved successfully even though
several portal pages displayed a temporary service-error banner. Prefer these live
metadata endpoints over cached search results, some of which still describe the
new datasets as pre-launch simulations. [S1–S4]

## 2. Current dataset

| Property | Verified finding |
|---|---|
| Title | Photovoltaic power production estimation and forecast on Belgian grid (Historical) |
| Identifier | `ods209` |
| Description | Belgian-grid PV generation estimated from measurements and upscaling; generation and forecast values are MW |
| Resolution / refresh | 15-minute intervals; historical metadata says daily refresh |
| Observation field | `measured_upscaled` |
| Dimensions | `datetime`, `region`, `gridconnectiontype` |
| National selection | `region='Belgium' AND gridconnectiontype='Total'` |
| Near-real-time counterpart | `ods210`, with observation field `measured` |

Sources: [historical metadata][S1], [near-real-time metadata][S2]. In `ods210`, the
description says quarter-hour updates while DCAT metadata says five-minute accrual.
These are inconsistent; neither is a publication-latency guarantee. The actual
generation interval remains 15 minutes.

Live national query results on the verification date:

| Dataset and selection | Earliest target (UTC) | Latest target (UTC) |
|---|---|---|
| `ods032`, Belgium, all rows | 2020-09-30 22:00 | 2026-10-25 22:45 |
| `ods209`, Belgium/Total, all rows | 2026-10-25 23:00 | 2026-11-22 22:45 |
| `ods209`, Belgium/Total, non-null target | 2026-10-25 23:00 | 2026-11-12 22:45 |
| `ods210`, Belgium, all grid types, all rows | 2026-11-12 23:00 | 2026-11-23 22:45 |

These are query-time bounds, not promised retention periods or proof of completeness.
Future timestamps in a historical dataset may carry only forecasts. Bounds were
queried using `select=min(datetime) as first,max(datetime) as last`, with the
selection above and `measured_upscaled is not null` for observed coverage. [S5]

## 3. Historical dataset/version history

Elia's current transition notice identifies **26 October 2026** as go-live:
`ods032` → `ods209` for history, `ods087` → `ods210` for near-real-time. [S4]
The API bounds above place the historical seam at Belgian local midnight on
26 October, with adjacent UTC quarter-hours and no boundary overlap in these
queries. Older cached wording mentions a 10:00 launch; do not use that announcement
time as the data partition boundary.

| Change | Before | After |
|---|---|---|
| Historical observation | `measured` | `measured_upscaled` |
| Grid connection | No field | `gridconnectiontype`: TSO, DSO, Total |
| Geography | National, regions and provinces | National and province/Brussels-level labels; Flanders/Wallonia removed |
| Longer forecast | `weekaheadforecast`, `weekaheadconfidence10/90` | `dayplustenforecast`, `dayplustenforecastconfidence10/90` |
| Other historical fields | `resolutioncode`, `loadfactor` | Neither present in live `ods209` field list |

The live `ods032` metadata and records support a start of **1 October 2020 local**;
this investigation did not establish a source for earlier history. Both historical
datasets are needed for a multi-year series continuing after the transition.
Future ingestion should map them separately, preserve provenance, and validate
the seam and potential methodology/capacity discontinuities before concatenation.
Do not infer scientific continuity from timestamp continuity alone. [S1, S3, S5]

## 4. Forecast target

Recommend **`ods209.measured_upscaled` in MW**, selecting Belgium/Total, and
**`ods032.measured` in MW** for earlier history. `ods210.measured` is the corresponding
near-real-time candidate. Dataset descriptions identify measured-and-upscaled
generation; the older and near-real-time field descriptions identify an interval
average. This is a national production estimate, not a direct meter reading of
every Belgian PV installation. [S1–S3]

The current historical field has a label but no field-level description or unit
annotation; its MW interpretation is supported by the dataset-level description.
Its exact upscaling methodology and whether the transition changed that methodology
remain unresolved. Do not claim independently verified national physical ground truth.
Capacity measures fleet size, and forecasts estimate future production; neither
should replace this observed-estimate target.

## 5. Geographic structure

A live `group_by=region,gridconnectiontype` query returned 36 combinations: 12
geographies × three grid types. Geographies were `Belgium`, `Antwerpen`,
`BrabantWallon`, `Brussel`, `Hainaut`, `Liege`, `Limburg`, `Luxembourg`, `Namur`,
`OostVlaanderen`, `VlaamsBrabant`, `WestVlaanderen`. Values are exact API strings,
not translated display names. [S6]

Select national **Belgium/Total directly**. Never sum national records with their
subdivisions, or Total with TSO and DSO. The grid field describes connection to
Elia's grid versus a distribution grid. National selection avoids needing to
reconstruct the total from subdivisions. Old `ods032` requires only Belgium;
do not fabricate a missing grid field. [S1, S3, S6]

## 6. Temporal resolution

Metadata specifies 15 minutes and explicitly defines `datetime` as **interval start**.
Observed `ods209` rows at 12:00 and 12:15 UTC confirm quarter-hour spacing in the
sample. `ods032` and `ods210` also expose `resolutioncode` (sample `PT15M`);
`ods209` does not. Treat intervals as `[start, start + 15 minutes)`. [S1–S3]

Power is MW, not MWh. Any later energy calculation must account for interval duration.

## 7. Timezone/DST behavior

Historical metadata says local display, UTC downloads/API, with portal timezone
`Europe/Brussels`. JSON samples contain explicit `+00:00` offsets. A one-record CSV
export for 2026-11-12 12:00 UTC also returned `2026-11-12T12:00:00+00:00`. [S1, S3, S7]
This verifies those API/CSV paths; do not assume every future export option preserves
the same representation without checking.

Small server-side counts in `ods032`, filtered to Belgium, gave:

| Belgian date | UTC half-open window | Rows / distinct timestamps |
|---|---|---|
| 2026-03-29, spring | `[2026-03-28T23:00Z, 2026-03-29T22:00Z)` | 92 / 92 |
| 2026-10-25, autumn | `[2026-10-24T22:00Z, 2026-10-25T23:00Z)` | 100 / 100 |

Queries used `count(*)` and `count(distinct datetime)`; no full days were downloaded.
Three-record samples around each change remained consecutive in UTC. At spring,
00:45 UTC corresponds to 01:45+01:00 and 01:00 UTC to 03:00+02:00. At autumn,
00:45 UTC corresponds to 02:45+02:00 and 01:00 UTC to 02:00+01:00. Repeated local
clock labels are unambiguous when offsets are retained. [S8, S9]

**Recommendation:** normalize to timezone-aware UTC; convert to Brussels for local
dates and future calendar/solar features. Construct local midnight boundaries
independently, convert to UTC, then generate 15-minute grids. Never strip offsets
or require 96 rows per day. The new dataset starts after the 2026 autumn change;
its DST behavior has not been independently sampled across a transition.

## 8. Capacity fields

`monitoredcapacity` is MW, labelled monitored capacity, described as total available
production capacity (`ods210`: total available capacity). It accompanies geography
and, in the new datasets, grid connection type. [S1–S3]

It could support capacity normalization, capacity-adjusted persistence and growth
analysis. Those are future uses. The inspected metadata does not specify a separate
capacity revision cadence, publication delay, fleet coverage, AC/DC basis, or a
guarantee that this equals comprehensive installed national nameplate capacity.
Dataset refresh frequency is not proof of capacity update frequency. Historical
capacity values must not be assumed known at earlier issue times.

## 9. Elia forecast fields

The live schemas expose these MW fields. [S1–S3]

| Family | Point forecast | Lower / upper fields | Documented timing |
|---|---|---|---|
| Latest | `mostrecentforecast` | `mostrecentconfidence10`, `mostrecentconfidence90` | Most recent value; no record issue time |
| Day-ahead 11AM | `dayahead11hforecast` | `dayahead11hconfidence10`, `dayahead11hconfidence90` | Description says published at 11AM |
| Day-ahead 6PM | `dayaheadforecast` | `dayaheadconfidence10`, `dayaheadconfidence90` | 6PM comes from field labels |
| D+10, new | `dayplustenforecast` | `dayplustenforecastconfidence10`, `dayplustenforecastconfidence90` | Description says ten days ahead |
| Week-ahead, old | `weekaheadforecast` | `weekaheadconfidence10`, `weekaheadconfidence90` | Old schema only |

P10 is the lower production bound, P90 the upper; descriptions frame them as lower
and upper limits of an 80% interval, or approximately 10% tail probabilities.
Do not reverse them using an exceedance-probability convention. No explicit P50
field is present; do not assume a point forecast is a median.

The transition notice says forecasts extend up to ten days. The `ods210` description
still says week-ahead, despite D+10 fields: a documentation inconsistency. Exact
clock timezone, publication latency, latest-forecast refresh behavior, D+10 issue
schedule, and historical constancy of schedules are not established. Use these as
potential external benchmarks only; exclude them from SolarCast inputs unless a
forecast-correction experiment is explicitly requested.

## 10. Historical forecast-vintage availability

These schemas contain target time and named forecast products, but **no explicit
forecast issue timestamp, availability timestamp, or revision identifier**. No
as-of/history endpoint was established in the inspected documentation. [S1–S3]

Named day-ahead columns preserve different products, not a complete issue-time
archive. `mostrecentforecast` cannot reconstruct arbitrary historical vintages.
Day-ahead columns may be useful benchmarks, but immutable original publication,
exact availability and revision behavior need confirmation before operational
comparisons. Dataset-level processing timestamps are not per-record availability.
Do not manufacture an issue timestamp by subtracting one day from target time.

## 11. API access

Official portal console: [Elia API console][S10]. Base:
`https://opendata.elia.be/api/explore/v2.1/catalog/datasets/`.

Verified metadata and records paths are `{dataset_id}` and `{dataset_id}/records`.
Anonymous GET requests succeeded without credentials. JSON records use
`total_count` plus `results`. Provider documentation describes ODSQL selection,
filtering, grouping, ordering and limit/offset pagination. Ordinary record queries
allow at most 100 per page, with `offset + limit < 10000`; use bounded date windows
instead of paginating across all history. [S1–S3, S11]

Example **verified query parameters** on `ods209/records`:

```text
select=datetime,region,gridconnectiontype,measured_upscaled,monitoredcapacity
where=region='Belgium' and gridconnectiontype='Total' and datetime>=date'2026-11-12T12:00:00Z' and datetime<date'2026-11-12T12:30:00Z'
order_by=datetime
limit=2
offset=0
```

URL-encode parameters. A separate `limit=2,offset=1` query returned the second and
third quarter-hours as expected. Prefer half-open UTC bounds, explicit field
selection and stable ascending ordering. Filter dimensions before pagination.

`{dataset_id}/exports/csv` was verified with a single exact-timestamp filter; it
returned semicolon-separated UTF-8 CSV. The provider documents JSON, CSV, XLSX and
Parquet export alternatives; the portal also exposes export pages. Export avoids
the ordinary records pagination ceiling, but this does not establish unlimited
Elia quotas. Parquet/XLSX downloads were not tested. [S7, S11]

No authoritative numerical Elia request-rate or daily quota was established.
Do not invent one. Later client design should use modest request frequency,
bounded retries for transient failures/429, and honor server retry instructions.

## 12. Relevant schema

| Source field | Meaning | Unit | SolarCast use |
|---|---|---|---|
| `datetime` | Interval start | Offset-aware datetime | Required target timestamp |
| `region` | Geographic selection | Text | Required national filter |
| `gridconnectiontype` | TSO/DSO/Total selection, new schemas | Text | Required national-total filter |
| `measured_upscaled` | Current historical PV estimate | MW, dataset-level declaration | Primary current target |
| `measured` | Old historical / current near-real-time estimate | MW | Adapter-specific target mapping |
| `resolutioncode` | Interval duration, absent in `ods209` | ISO duration text | Validate when present |
| `monitoredcapacity` | Available monitored capacity | MW | Optional future covariate |
| Forecast families above | Elia forecast products | MW | Separate optional benchmark |
| `loadfactor` | Measured generation / monitored capacity, old schema | Percent | Not required initially |

Live metadata reports numeric fields as `double`; sampled records contain JSON
numbers or null. Some cached portal JSON-schema renderings had a string type for
`measured_upscaled`; verify actual response types and fail clearly on incompatible
changes rather than trusting cached schema rendering. [S1–S3]

## 13. Example record

Actual `ods209` record, relevant fields only, retrieved on 2026-11-13 via national
Total selection and `measured_upscaled>0`, ordered by descending datetime: [S12]

```json
{
  "datetime": "2026-11-12T15:45:00+00:00",
  "region": "Belgium",
  "gridconnectiontype": "Total",
  "measured_upscaled": 8.29,
  "monitoredcapacity": 12611.241,
  "dayahead11hforecast": 19.78
}
```

Attribution: Elia Transmission Belgium SA. Fields omitted; values otherwise unchanged.
This is a schema illustration, not a committed historical dataset or frozen benchmark.

## 14. Data-quality considerations

Nulls occur, including future rows with forecasts but no measured estimate. A
national `ods032` aggregate counted 212,740 rows versus 212,102 non-null measured
values: 638 null observations at query time. This is not a full gap/duplicate audit.
No explicit quality flag, final/provisional indicator, per-record revision counter,
or publication timestamp exists in the inspected field lists. Absence of these
fields does not prove values are unrevised. [S1–S3, S5]

Future preprocessing should retain missingness; distinguish null from physical zero;
validate key uniqueness, quarter-hour alignment and finite values; report gaps;
and preserve retrieval provenance. Do not silently clip negatives or fill gaps.
Upscaling changes, capacity revisions and the schema seam merit separate diagnostics.

## 15. Licensing and attribution

Dataset metadata names **Elia Open Data Licence**, links to the [official license
page][S13], and attributes **Elia Transmission Belgium SA**. [S1–S3]

The official page's search-indexed text identifies CC BY 4.0 and adds Belgian-law /
Brussels-court wording. **The live license page returned only the portal service
error during this investigation, so that text could not be reverified directly.**
Treat applicability/full terms as pending live confirmation, not as a completed
legal clearance based on a search snippet.

The [official CC BY 4.0 deed][S14] permits sharing/adaptation subject to attribution,
a license link, marking modifications, and no added restrictions; it disclaims
warranties. These are conditional background terms until Elia's current linked
text is available. No broader legal conclusion is made here.

Recommended strategy: users obtain data from Elia; do not commit raw datasets.
Preserve attribution, dataset/source links, retrieval date, license reference and
transformation notes in cache metadata and derived outputs. Recheck the official
license before redistribution. SolarCast's Apache-2.0 code license remains separate.

## 16. Recommended SolarCast field subset

- **Required:** `datetime`, observation field for the dataset version, `region`,
  and new-schema `gridconnectiontype`. Validate `resolutioncode` where provided;
  otherwise retain the documented 15-minute duration as dataset metadata.
- **Useful later:** `monitoredcapacity`; explicitly selected day-ahead forecasts
  in a separate benchmark table, subject to availability verification.
- **Not initially needed:** subnational/TSO/DSO series, latest and D+10/week-ahead
  forecasts, quantile fields and old `loadfactor`.

Ingestion uses historical datasets. Add near-real-time retrieval only when
needed; do not introduce an automatic fallback that silently mixes publication states.

## 17. Proposed normalized internal schema

After national filtering, one series needs only:

| Internal field | Type / rule |
|---|---|
| `target_time` | Timezone-aware UTC interval start; unique within a frozen series snapshot |
| `pv_generation_mw` | Nullable numeric national measured-and-upscaled estimate |
| `monitored_capacity_mw` | Optional nullable numeric; preserve the source's limited semantics |

Keep dataset ID, source field mapping, national selection, 15-minute duration,
interval-start convention, retrieval timestamp, source URL, license status and
content fingerprint in snapshot metadata. Retain per-partition provenance when
joining dataset versions. A snapshot download is not an as-of archive.

`available_at` remains **unknown**, not equal to target or retrieval time. Future
contracts may carry nullable availability plus a documented availability policy;
strict historical operational evaluation is blocked until defensible availability
information or a clearly labelled approximation exists. Preserve issue time
separately when actual forecast products are introduced.

For future caching, recommend local **Parquet plus a small JSON metadata sidecar**, with timezone-aware
UTC timestamps and nullable values. It suits this small quarter-hour series and
repeated reloads; no database is warranted. Storage and dependencies are deferred.

## 18. Known uncertainties

Before claiming operationally realistic evaluation, resolve observation publication
lags, revision/finalization policy, original forecast availability and capacity
vintages. Also confirm upscaling methodology and continuity across the transition,
capacity coverage and AC/DC basis, exact forecast clock/timezone and refresh
guarantees, numeric access quotas, and live licensing terms. DST of the new schema
and export formats beyond the sampled CSV remain untested. Multi-year missingness
and duplicate audits belong to later ingestion/validation, not this small probe.

## 19. Official source links and probe recipes

All API recipes below use the verified `/records` endpoint. Encode query values;
the observations above are dated snapshots and may change on re-execution.

- **S1:** [Current historical metadata][S1] (field list, units, cadence, license).
- **S2:** [Current near-real-time metadata][S2].
- **S3:** [Old historical metadata][S3].
- **S4:** [Current transition notice][S4].
- **S5:** [Historical records][S5]: `select=min(datetime) as first,max(datetime) as last,count(*) as n`; `where=region='Belgium'`; add `gridconnectiontype='Total'` for new history and `measured_upscaled is not null` / `measured is not null` for observed bounds.
- **S6:** [New historical records][S6]: `select=region,gridconnectiontype,count(*) as n`; `group_by=region,gridconnectiontype`; `limit=50`.
- **S7:** [CSV export][S7]: select relevant fields, filter Belgium/Total and `datetime=date'2026-11-12T12:00:00Z'`; one record retrieved.
- **S8:** [Spring DST count][S8]. Same query with `count(distinct datetime)` returned 92 unique times; sampled 00:45–01:15 UTC, `limit=3`.
- **S9:** [Autumn DST count][S9]. Same distinct-time query returned 100; sampled 00:45–01:15 UTC, `limit=3`.
- **S10:** [Elia API console][S10].
- **S11:** [Official Explore API 2.1 reference][S11] (provider documentation linked by Elia).
- **S12:** [Example record query][S12].
- **S13:** [Elia license page][S13] (live content unavailable; limitation above).
- **S14:** [Creative Commons official CC BY 4.0 deed][S14].

[S1]: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods209
[S2]: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods210
[S3]: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032
[S4]: https://opendata.elia.be/explore/dataset/ods209/information/
[S5]: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/records
[S6]: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods209/records
[S7]: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods209/exports/csv
[S8]: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/records?select=count%28%2A%29%20as%20n&where=region%3D%27Belgium%27%20and%20datetime%3E%3Ddate%272026-03-28T23%3A00%3A00Z%27%20and%20datetime%3Cdate%272026-03-29T22%3A00%3A00Z%27&limit=1
[S9]: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/records?select=count%28%2A%29%20as%20n&where=region%3D%27Belgium%27%20and%20datetime%3E%3Ddate%272026-10-24T22%3A00%3A00Z%27%20and%20datetime%3Cdate%272026-10-25T23%3A00%3A00Z%27&limit=1
[S10]: https://opendata.elia.be/api-console/explore/v2.1/
[S11]: https://help.opendatasoft.com/apis/ods-explore-v2/
[S12]: https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods209/records?where=region%3D%27Belgium%27%20and%20gridconnectiontype%3D%27Total%27%20and%20measured_upscaled%3E0&order_by=datetime%20desc&limit=1
[S13]: https://opendata.elia.be/pages/licence/
[S14]: https://creativecommons.org/licenses/by/4.0/

## 20. Implemented historical ingestion

```python
from solarcast.data import EliaPVClient

with EliaPVClient(timeout=30.0, max_retries=2) as client:
    pv = client.fetch_history(
        start="2026-10-25T22:45:00Z",
        end="2026-10-25T23:15:00Z",
    )
```

This two-interval example crosses the schema seam. Inputs accept timezone-aware
Python `datetime` / pandas `Timestamp` values or offset-aware timestamp strings.
Date-only/naive inputs, empty/reversed ranges and non-quarter-hour bounds raise
`ValueError` before HTTP. UTC conversion occurs before routing. Requests use
`[start, end)`; the boundary is defined once in `data/_schema.py` as
`2026-10-25T23:00:00Z` (26 October midnight Brussels).

Output is a sorted pandas DataFrame with a fresh RangeIndex and these dtypes:

| Column | dtype | Semantics |
|---|---|---|
| `target_time` | `datetime64[ns, UTC]` | Interval start |
| `pv_generation_mw` | nullable `Float64` | National measured/upscaled estimate, MW |
| `monitored_capacity_mw` | nullable `Float64` | Source monitored capacity, MW |
| `source_dataset` | pandas `string` | `ods032` or `ods209` |
| `source_target_field` | pandas `string` | `measured` or `measured_upscaled` |
| `retrieved_at` | `datetime64[ns, UTC]` | Time each successful response was normalized |

Provenance columns are deliberate: mixed-source tables retain their origin even
after ordinary pandas operations. `retrieved_at` is not `available_at`. Observation
publication timing remains unknown; monitored capacity is not established as a
leakage-safe feature. No Elia forecast fields are selected. No `ods210` fallback,
cache, database, forecast API or feature generation is included.

### Requests and failures

The client requests **one elapsed UTC day at a time**, split first at the transition.
At national quarter-hour resolution that is at most 96 distinct records, fitting
one `limit=100, offset=0` response. This is not an assumption about Belgian local
day length: 25-hour requests span two windows. An oversized/incomplete response
fails explicitly; no multi-year offset-pagination loop is used. Every request
selects only required fields and national filters, with ascending timestamp order.

HTTPX provides connection pooling, explicit per-operation timeouts (default 30s)
and mockable transport. Requests are sequential, spaced at least 0.25s apart by
default; `request_interval` is configurable. This is a courtesy setting, not a claim
about Elia's quota. Retries are bounded (two retries after the initial attempt),
cover transport failures and HTTP 429/500/502/503/504, with exponential backoff.
Other HTTP statuses fail immediately. `Retry-After` seconds or HTTP dates are
honored; waits over 60s cause an error rather than retrying earlier than instructed.
Use the context manager or call `close()` to release resources.

`EliaAPIError` reports transport/status/envelope failures; `EliaSchemaError`
reports missing fields, invalid source types/resolution or violated national filters;
`PVDataValidationError` reports invalid timestamps, duplicates, range violations,
non-finite values or seam problems. Errors include dataset/range context. Configure
standard Python logging in the application; library code does not print.

Null PV/capacity remains `pd.NA`; numeric zero stays zero. Negative values are
preserved with warning logs. NaN/infinity are rejected before nullable conversion
so they cannot silently become ordinary missing observations. Missing interval rows
are reported without insertion, interpolation or removal. Returned rows are sorted;
unexpected source ordering is logged. Duplicates raise without deduplication.

Crossing requests must contain the last old-schema quarter-hour and the first
new-schema quarter-hour. Overlap, wrong-side source timestamps or a seam gap raises;
neither source is silently preferred. Continuity of timestamps still does not prove
methodological continuity of the estimates. Requests outside actual source coverage
can return typed empty/partial tables with missing-row warnings; there is no silent
fallback or availability inference.

### Development and limitations

Offline tests use synthetic rows with verified field names and HTTPX MockTransport;
normal tests never access the live service. Run the small explicit live example with
`python examples/fetch_elia_history.py`. Caching/Parquet and broad historical quality
audits remain deferred. The snapshot provenance is intentionally minimal; a future
cache must additionally preserve source URL, license/attribution, selections and
fingerprints. Existing licensing and vintage uncertainties above still apply.

Manual implementation smoke checks on 2026-11-13 retrieved four `ods209` rows for
2026-11-12 12:00–13:00 UTC and two rows straddling the transition. Both passed UTC,
ordering, schema and source checks. The four daytime observations were finite,
nonnegative and non-null. At the seam, PV was zero on both rows while monitored
capacity changed from 12128.284 to 12107.687 MW. This small observation is retained
without correction and is not an explanation of the capacity/methodology change.
