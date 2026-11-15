# Historical weather/NWP sources and forecast-vintage contract

**Current status:** historical GFS ingestion and context integration are implemented;
see [section 22](#22-historical-gfs-ingestion). Sections 1–21 document the dated
source investigation, provider evidence and design rationale. Later sections state
the implemented contract and precision policy.

**Source investigation date: 14 November 2026.** During that investigation, no NWP fields,
multi-year weather data, ingestion implementation or weather-aware model were
downloaded/built. Read-only probes retrieved small catalogues, file indexes and
HTTP HEAD metadata. No paid requests, account registration or user credentials
were used. Recommendations below are distinct from documented provider facts and
from the limited probe records verified.

## 1. Recommendation

Use **NOAA/NCEP operational deterministic GFS, regular 0.25° `pgrb2` atmospheric
forecast output**, from the official NOAA Open Data Dissemination (NODD) public
bucket `noaa-gfs-bdp-pds`, as the first reproducible weather source. Start with
**DSWRF at surface, instantaneous TCDC for the entire atmosphere, and TMP at 2 m**.
Use hourly short-lead output, one eligible run per SolarCast issue, and keep only
a Belgian spatial subset after decoding. A six-hour conservative run-availability
assumption selects the same-day **00Z** run at the current 11:00 Brussels issue.
This is an explicitly approximate retrospective policy, not proof of historical
publication times. [G1–G6]

Why this choice: operational run identities, long freely accessible history,
verified 2024/2025 objects, all three variables, hourly short leads, no API key,
and a simple official HTTPS/index/range access path. It is **not** a claim that
GFS has the lowest Belgian radiation error. Its coarse spatial grid, global-message
transfer volume and unknown historical first-publication times are real costs.

ECMWF remains a serious later comparison, preferably its **actual operational
IFS archive** with verified product/field entitlement and run provenance. Do not
substitute ERA5 or an IFS hindcast archive merely to obtain convenient historical
coverage. An important qualification: the Azure ECMWF mirror does retain some
old operational files, which were verified here; it is wrong to infer that every
cloud mirror has the short retention of the main ECMWF portal. [E1–E6, O1]

## 2. SolarCast requirements inspected

The fixed history-only zero-shot TSFM benchmark has t0-gamma first (866.33 MW RMSE),
Chronos-2 second (938.13), LightGBM third (1026.88), and 48h persistence effectively
tied with LightGBM (1027.03). LightGBM wins the most individual days; both TSFMs
share strongly correlated errors. These findings motivate testing new information,
not assuming which architecture should win with weather. See the
[complete benchmark](../experiments/tsfm-zero-shot-v1.md).

The existing issue is **11:00 Europe/Brussels = 09:00Z in summer / 10:00Z in
winter**. PV history uses `floor_15min(issue - 30min)` and only completed intervals.
That PV safety lag must not be reused as an NWP publication-latency rule.
Targets are interval starts for the **next Belgian calendar day**, with 92/96/100
quarter-hours; UTC intervals remain regular through DST.

The backtester builds contexts before attaching actuals. The current
`ForecastContext.features` validates target alignment but cannot certify how
external features were obtained. Moreover, current LightGBM reconstructs its
exact 13-feature schema and both TSFM adapters ignore `context.features`.
**Adding columns to that table alone would neither establish vintage safety nor
enable a weather model.** The approved NWP path is described in section 22 and the
[weather-informed model guide](../models/lightgbm.md#optional-history--nwp-configuration).

## 3. What a historical weather forecast means

Separate four times:

1. SolarCast forecast issue **T**, when its PV forecast is made.
2. NWP **reference/initialization R**, the meteorological run identity.
3. NWP **publication/availability A**, when the required output became accessible.
4. NWP **valid time V**, or interval `[V_start,V_end)`, being predicted.

Eligibility is **A <= T**, not merely R <= T. Tomorrow's NWP valid times may be
after T because these are forecasts known at T. A newer run, later correction or
future observation cannot enter by matching the same valid timestamp.
`retrieved_at` in 2026 does not reconstruct availability in 2024/2025.

Operational forecast archives contain the original changing model systems.
Reforecasts/hindcasts produced later with a different system answer a different
question, even when their reference-time coordinate is historical. ERA5,
ERA5-Land, CAMS reanalysis, DWD ICON-DREAM, and final analyses must not be passed
off as the weather forecast available to an operational PV forecaster.

## 4. ECMWF products and access findings

| Product/access | Verified properties | Decision |
|---|---|---|
| IFS Open Data, main portal | Recent rolling runs; 0.25° and new 0.1° IFS from 20 Oct 2026; four cycles; GRIB2; free CC BY 4.0 | Useful prospective feed, insufficient by itself for 2024–2025 history |
| Operational IFS archive / MARS | Actual archived forecasts, reference times/steps, server-side area/grid/parameter selection; archive/API entitlement needed | Scientifically suitable; access terms/cost not universally free |
| ECMWF cloud mirrors | Independent retention; Azure historical operational index personally verified | Real alternative, but exact history/variable completeness must be audited |
| AIFS single/ensemble | Four cycles, 0.25° open output, 6-hourly steps to 360h; operational family introduced in 2025 | Shorter history and coarse temporal output; defer |
| CAMS global composition forecasts | Actual IFS-based forecasts from 2015, ADS account/API, 0.4° output, hourly surface fields, 00/12Z, five-day horizon | Viable free forecast archive, but coarse and late for current issue |
| ERA5/CDS reanalysis | Retrospective reconstruction, not the original operational weather forecast | Exclude from primary operational-input track |

Sources: [E1–E9, C1–C2]. “Free data licence” and “free archive service” are separate.
The published ECMWF commercial MARS Web API service charge is **EUR 3,000/year**;
research/societal-benefit waivers are discretionary. No waiver or entitlement was
obtained, and no paid service was accessed. [E3]

IFS native high-resolution output is about 9 km; that does not mean every access
route supplies native 9 km hourly fields. The current operational archive advertises
hourly high-frequency output through step 144. Open Data normally distributes
3-hourly short-lead fields. Official pages disagree over some 06/18Z maximum
steps (90 versus 144) and main-portal retention (12 runs/2–3 days versus about
four days); neither disagreement affects the needed first 48 hours, but a future
client must inspect actual product messages rather than generalize between routes.
Do not apply the new October 2026 0.1° product retrospectively to 2025. [E1,E2,E5,E6]

IFS radiation `ssrd` (169), **surface downward shortwave radiation**, is accumulated
energy in J m^-2 in the standard forecast diagnostics. Difference matching
accumulation bounds within one run and divide by duration to obtain mean W m^-2;
do not read it as instantaneous GHI or substitute net shortwave `ssr` (176).
`2t` is Kelvin, and `tcc` is fractional cloud cover in standard IFS output.
Historical open subsets need their own variable audit. [E1,E9]

### Publication schedules

The current direct IFS dissemination schedule starts transferring 00Z short-lead
steps around **05:45–06:12Z**, and 06Z steps around **11:45–12:12Z**. These are
progressive transfer windows, not timestamps for every public mirror. Open Data
release and mirror ingestion add route-specific delays; 0.1° adds about two hours
relative to 0.25°. The 06Z IFS run is not justified at SolarCast's 09/10Z issue
from these schedules. A 00Z choice is plausible for an audited older route;
current 0.1° timing needs particular care at the 09Z summer deadline. [E1,E4]

CAMS documentation gives guaranteed availability by **10:00Z for 00Z**, and
**22:00Z for 12Z**, possibly earlier without guarantee. Thus same-day CAMS 00Z
cannot be assumed at the 09Z summer issue; the preceding 12Z is the conservative
choice, with a full horizon/availability check. ADS completion/queue time is not
historical dissemination time. CAMS is genuine forecast data, not categorically
rejected as reanalysis; its resolution and timing make it a secondary choice here.
[C1,C2]

### Azure mirror: personally verified, with limits

A signed **public** read request obtained the index for the 10 June 2025 00Z
operational IFS 0.25° step 24. It contains `ssrd` and `2t`, but no `tcc` entry in
that particular index. Anonymous Azure access failed; the public Planetary
Computer signing service provided a short-lived token without a user account.
No token is stored in SolarCast. Its STAC item preserves reference and valid times.
Candidate January 2024/December 2023 legacy-key probes returned 404; this is not
proof that every older IFS object is unavailable. The collection's retention
description and temporal extent are incomplete/stale relative to the verified
2025 object. Do not promise a complete 2024–2026 radiation/cloud/temperature
archive on that evidence. [E6, probe records]

## 5. Other serious candidates

**DWD ICON-EU:** attractive European resolution (roughly 6–7 km native; older
regular 0.0625° delivery), runs every three hours, hourly short-lead output,
direct/diffuse solar fluxes, clouds and 2 m temperature. The live DWD tree is a
recent rolling feed, not a verified 2024–2025 run archive. Public DWD access is
free/no registration, subject to DWD terms. The current legal-notice page returned
403 during research; do not infer a particular data licence from ICON's separate
BSD code licence. DWD also announced a feed/native-grid migration after
30 November 2026; the announcement itself was blocked here, while the live new
`v1` tree and the access provider's migration warning were visible. These changes
and archive uncertainty argue against making it v1's required dependency.
[D1–D5]

**Open-Meteo:** distinguish three APIs. Historical Forecast stitches runs and is
not the desired issue-time vintage store. Previous Runs supplies fixed lead-day
offsets, not a complete original run at a chosen SolarCast issue. Single Runs
preserves explicit `run=...`, but its IFS history from **14 March 2024 is labelled
Cycle 49R1 hindcasts**; ECMWF records 49R1's operational introduction on
12 November 2024. That early history cannot be asserted to be the operational
forecast available in March 2024. Most other Single Runs models begin
2 April 2026. Require actual-production provenance before using any such run in
the primary operational track. [O1–O4,E8]

Open-Meteo's data licence and hosted API terms differ: CC BY 4.0 data do not make
its free hosted tier commercially unrestricted. The free service is for
non-commercial use, with 10,000 calls/day and other limits; historical/Single Runs
commercial access requires the relevant paid tier. No paid API was queried.
[O5]

## 6. Comparison against the practical requirements

| Criterion (importance) | NOAA GFS / NODD | Direct IFS archive | CAMS / ADS | ICON-EU live | Open-Meteo Single Runs |
|---|---|---|---|---|---|
| Historical run archive (critical) | Verified sampled old objects; GDEX deeper | Yes, entitlement | Yes, since 2015 | No suitable deep official feed verified | Yes, but distinguish hindcasts |
| Original forecast vintage (critical) | Operational ref/lead identity | Operational stream, not reforecast | Operational forecasts | Present live runs | Early IFS explicitly hindcasts |
| Radiation (critical) | Surface DSWRF | SSRD | SSRD/direct available | Direct + diffuse | Model-dependent derived fields |
| Clouds (high) | Instant TCDC | TCC | TCC | CLCT/layers | Model-dependent |
| Belgium spatial detail (high) | 0.25° output | Native ~9 km or selected grid | 0.4° output | ~6–7 km native | Model-dependent |
| Time resolution (high) | Hourly short leads | Hourly archive / route-dependent | Hourly surface | Hourly short leads | Often hourly after processing |
| Historical depth (high) | S3 prefixes from 2021; GDEX from 2015 | Decades; exact fields entitlement/catalogue | 2015 onward | Rolling, not years | IFS 2024 hindcasts; others Apr 2026 |
| Availability at issue (high) | 00Z under disclosed +6h assumption | Audit 00Z dissemination | Previous 12Z conservatively | Run/route evidence needed | Adds provider ingestion lag |
| Licence (high) | NOAA NODD terms; derived changes identified | CC BY 4.0 + ECMWF terms | Copernicus/ADS terms | DWD terms; recheck exact version | CC BY data + separate API terms |
| Programmatic access (high) | HTTPS index + byte ranges | MARS/Web API, area subset | `cdsapi`, area subset | HTTPS GRIB, changing grids | Convenient JSON run API |
| Cost (high) | Public unsigned access | Service charges/possible waiver | Free account, queues/limits | Free | Free non-commercial / paid commercial |
| OSS reproducibility (high) | Strong, no account; audit missing files | Depends on users' entitlement | Account/terms acceptance | Historical retention problem | Terms + provenance constraints |
| Storage/transfer (medium) | Small cropped store; global field transfer | Server-side crop possible | Server-side crop possible | Regional fields then crop | Small point JSON responses |
| Complexity (medium) | GRIB/index/radiation decoding | Credentials + MARS + GRIB | Credentials + queue + GRIB | Grid migration + archive | Easy API, harder vintage assurance |

The chosen route wins on verifiable original runs and self-service access, not
an assumed accuracy ranking. The free CAMS archive and entitlement-based IFS
remain useful alternatives if GFS radiation quality proves inadequate on a
separate validation set. No five-provider implementation is proposed.

## 7. Selected GFS product, historical evidence and overlap

NCEP's inventory identifies `gfs.tCCz.pgrb2.0p25.fFFF`, cycles **00/06/12/18Z**,
and forecast hours through 384. The official model/product documentation specifies
hourly output through 120h, then 3-hourly through 384h. Short-lead hourly files
were personally verified. Native model resolution (~13 km in the documented
system) is distinct from the selected 0.25° output grid. [G9]
Do not confuse GFS forecast output with GDAS/FNL analyses or warm-start states.
The AWS registry has stale explanatory model-version text; preserve the actual
GRIB generating-process/table metadata rather than assuming a single physics
version over all years. [G1,G2]

The observed bucket's first `gfs.` date prefixes are **20210101–20210104**.
Selected indexes/HEADs were checked at:

| Reference / step | Index result | Core-field evidence |
|---|---|---|
| 1 Jan 2021 00Z / 24h | 200, legacy path without `atmos` | Historical key exists; early-path support separate |
| 18 Dec 2023 00Z / 24h | 200 | Start of current PV snapshot era |
| 1 Jan 2024 00Z / 24h | 200; GRIB HEAD 200 | TMP 2m, instantaneous and averaged TCDC, DSWRF |
| 10 Jun 2025 00Z / 24,25,27h | All 200; GRIB HEADs 200 | Original reference, leads, different averaging bounds |
| 1 Oct 2026 00Z / 24h | 200; GRIB HEAD 200 | Later operational-object evidence |

These are small metadata observations, **not a complete coverage or decoding
audit**. The current S3 prefixes imply prospective overlap from 1 January 2021
with Elia's documented October 2020-onward old schema. The run-date envelope
through 25 October 2026 spans **2124 calendar dates**. Requiring a preceding-day
issue for next-day targets gives **2123 potentially usable target dates** from
2 January 2021 through 25 October 2026, before the Elia schema seam; exact UTC
boundaries, missing messages and PV quality must still be checked.
The entire existing PV snapshot, 18 December 2023–31 October 2025 (**684 local
days**), falls inside the archive's date span. Both the original 2024 training
year and 2025 benchmark period are potentially covered, with full per-run
completeness still unverified. [G1,G3, Elia contract]

For older years, **NCAR GDEX d084001** (formerly ds084.1, DOI 10.5065/D65D8PWK)
provides operational GFS 0.25° forecasts from 15 January 2015, parameter/area
subsetting and a CC BY 4.0 declaration. Its catalogue describes **3-hourly**
forecasts through 240h, not the hourly NODD product. It also announces stopping
updates in early 2026 while displaying later/future end-date metadata; therefore
do not use its headline maximum as proof of 2026 completeness. Subsetting requires
login, whereas public NODD does not. It is an optional same-model access route,
not a silently interchangeable hourly cache. [G3,G4]

## 8. Variables and exact radiation semantics

| Native field | Required level/time statistic | Units | Normalized use |
|---|---|---|---|
| `DSWRF` — Downward Short-Wave Radiation Flux | **surface**, average over the encoded interval | W m^-2 | Reconstruct non-overlapping hourly energy/mean GHI |
| `TCDC` — Total Cloud Cover | **entire atmosphere**, **instantaneous** | % | Fraction after dividing by 100; do not select averaged/convective-layer variants |
| `TMP` — Temperature | **2 m above ground**, instantaneous | K | `temperature_2m_c = TMP - 273.15` |

Use all-sky downward surface shortwave, not net solar, top-of-atmosphere radiation,
upward flux, UV or direct-normal flux. NCEP has standard and local-table encodings;
decode/store the actual identifiers and GRIB tables, not only a guessed ecCodes
alias. Native field names/levels/units are established by NCEP's product inventory
and parameter tables. Direct/diffuse components, layer clouds, wind, humidity,
pressure and precipitation are deferred; an available variable is not automatically
a justified feature. [G2,G5,G6,G7]

**The important finding:** hourly filenames do not imply an hourly DSWRF average.
The June 2025 index reports step 24 as **18–24 hour average**, step 25 as
**24–25 average**, and step 27 as **24–27 average**. TCDC has both instantaneous
and average entries at the same level. Inspect `stepType`, `startStep`, `endStep`,
statistical-processing type, units and level in decoded GRIB before accepting a
message; the index is evidence for the future test fixture, not a substitute for
the binary-header check.

Let A(s,e) be a native W m^-2 average from lead s to e hours. Its energy is
`I(s,e) = A(s,e) * (e-s) * 3600` J m^-2. For two outputs with the **same start s**:

```text
E(e1,e2) = 3600 * [(e2-s)*A(s,e2) - (e1-s)*A(s,e1)]
GHI_mean(e1,e2) = E(e1,e2) / [3600*(e2-e1)]
```

For a one-hour increment ending at 24h, this gives `6*A(18,24)-5*A(18,23)`.
At the reset to `(24,25)`, use A(24,25) directly; **do not difference it against
the previous block**. A preceding endpoint is required when cumulative windows
overlap. Never difference across runs, missing messages, ensemble members,
different cells or incompatible grids. Negative increments may indicate packing
rounding or a decoding error: flag them and fail outside an explicitly documented
precision tolerance, rather than silently clipping all negatives to zero.
Perform interval arithmetic in float64 even if the cropped values are stored as
float32; this avoids adding avoidable cancellation error to native GRIB packing.

This same interval discipline applies to alternative radiation products: standard
IFS SSRD is energy in J m^-2, while ICON averaged direct/diffuse fields would need
their own verified running-average reconstruction. Do not apply ERA5's accumulation
recipe to every IFS product. [E9,D2]

## 9. Run selection and publication-latency policy

NCEP's current production status shows 00Z short-lead products completing around
03:40–04:15Z and the full run around 05:15Z; times are **30-day running averages**,
not historical guarantees. The 06Z production begins around 09Z and progresses
afterwards. Sample S3 00Z object last-modified times were around 03:38–03:58Z,
with some later forecast steps appearing before intermediate hourly files.
Selecting one “last step” is therefore insufficient to establish bundle readiness.
[G8, probe records]

**Proposed v1 policy**, fixed before any weather-model scores:

- Retrospective approximation `gfs_init_plus_6h_v1`: effective availability is
  R+6h for the complete required variable/step bundle. At 09Z or 10Z this selects
  the same-day 00Z run; 06Z remains ineligible until 12Z under this policy.
- This lag is a **SolarCast assumption**, supported as conservative by sampled
  timings, not a fabricated observed timestamp or service-level guarantee.
  A 06Z run might actually be usable at some winter issues; this v1 approximation
  does not claim to reconstruct the exact latest historical run.
- In an observed-availability mode, select the maximum R among complete bundles
  whose recorded availability is <= T. A newer partial bundle is ineligible.
  Record a deterministic older-run fallback, age and reason, or fail explicitly.
- Use one run for the whole target grid, not a per-valid-time mix of newer runs.
  Every endpoint needed for radiation differencing or interpolation must belong
  to that eligible bundle and be available by T.
- Treat corrections as distinct bundle versions. For equal reference times,
  only a correction known by T may be chosen; its effective availability is no
  earlier than its correction-publication time. Never assign an old run's R+6h
  to a newly revised copy. Unknown revision timing remains a retrospective
  uncertainty and is rejected in strict observed-availability mode.
- Historical HEAD `Last-Modified` and current retrieval times are evidence fields,
  not guaranteed original public-first-seen timestamps. Never backdate a forecast
  retrieved today merely because its initialization is old.

For the primary 00Z path, **hourly steps 18–48 inclusive** form a small sufficient
starting envelope for next-day UTC endpoints and radiation predecessors at the
current issue. Summer targets span roughly R+22 to R+45:45, winter R+23 to
R+46:45; the final quarter needs the following hourly endpoint. DST uses exact UTC
target boundaries, never a fixed 96 rows. Other issue schedules or eligible runs
must derive their required endpoints rather than hard-code this envelope.
The 18–48 envelope covers next-day LightGBM targets, not a future TSFM's bridge
beginning at today's history cutoff; that later experiment needs earlier steps too.

**What can be guaranteed?** The selector can guarantee compliance with a recorded
availability registry or with the declared lag rule. Without historical publication
logs, it cannot prove actual historical availability. Strict operational-as-of
mode must reject unknown availability; assumption-based retrospective results
must retain that label. Start recording first-seen/completeness times prospectively
so future benchmarks can tighten this guarantee.

## 10. Quarter-hour alignment

Normalize radiation to non-overlapping hourly intervals first. Proposed simplest
v1 disaggregation: repeat the **hourly mean flux** over each of its four UTC
quarter-hours. Each quarter then has energy `GHI_hour * 900`, preserving the
hourly total. This is a piecewise-constant **hourly-resolution forecast feature**,
not newly resolved 15-minute cloud physics. Never linearly interpolate cumulative
energy or native overlapping averages.

Interpolate the selected run's instantaneous temperature and TCDC between its
hourly valid endpoints, then use the mean over each PV quarter-hour (equal to
the midpoint for a linear segment). Reject missing endpoints, incompatible units,
out-of-range cloud fractions or extrapolation. Future **forecast** endpoints are
allowed when their run was known by T; future observed endpoints are not.

A later predefined validation alternative may redistribute hourly solar energy
using deterministic per-cell clear-sky geometry, with nonnegative weights summing
to one and an explicit all-dark fallback. It must preserve hourly energy and
must not be selected on the final test set. Do not introduce that extra choice
before the minimal weather ingestion contract is verified.

## 11. Belgium extraction, cell counts and spatial features

Proposed bounding box with a small margin: **N=51.75, W=2.25, S=49.25, E=6.75**,
with east-positive longitudes (2.25–6.75°E, 49.25–51.75°N). This is a design
extent, not a newly hard-coded setting. It covers
the Belgian mainland; exact border/mask support must be checked against a pinned,
appropriately licensed official boundary for spatial aggregation. At the GFS 0.25° centres
there are **11 latitudes ×19 longitudes =209 points** in that box.

At 50.5°N a 0.25° cell is about 27.8×17.7 km, roughly 492 km². Belgium's official
30,689 km² area therefore corresponds to approximately **62 cell-area equivalents**.
Expect of order 60–80 centre-inside cells, more cells touching borders; these are
estimates, not a counted polygon intersection. By comparison native ~9 km IFS
would resolve of order 380 area equivalents, and ~7 km ICON-EU about 630. A
delivered regular grid and native model mesh must not be conflated. [B1]

**Aggregation recommendation:** keep the cropped cell table, and use fixed
Belgium-polygon intersection-area weights. Preserve simple west/central/east
variation with longitude bands `<4°E`, `[4,5)°E`, `>=5°E` intersected with the
country. This is an explicit geographic partition, not a capacity distribution.
Pin boundary, grid and weight hashes. Equal cell weights are only a documented
fallback approximation; box averages including foreign/sea cells are not national
Belgium averages. West cloudy/east sunny and uniformly intermediate conditions
can have identical national means, so three radiation means plus dispersion are
more useful than a permanent single-Brussels-point proxy.

Proposed **seven** new LightGBM features, appended to an independently versioned
weather schema (the existing 13-feature model stays the control):

```text
nwp_ghi_west_mean_w_m2
nwp_ghi_central_mean_w_m2
nwp_ghi_east_mean_w_m2
nwp_ghi_belgium_spatial_std_w_m2
nwp_cloud_belgium_mean_fraction
nwp_cloud_belgium_spatial_std_fraction
nwp_temperature_2m_belgium_mean_c
```

Compute weighted standard deviations as geographic dispersion, not probabilistic
ensemble uncertainty. Temperature is included as a modest PV-efficiency/weather
signal, without pretending it equals module temperature. No panel tilt/orientation,
snow, capacity normalization, observed irradiance or dozens of extra variables
are assumed. National GHI mean may be retained for QC without becoming another
redundant v1 model feature.

## 12. Capacity weighting investigation

Elia's `ods032` metadata has a regional dimension and describes
`monitoredcapacity` as total available production capacity; the current local
benchmark retained national rows only. A national scalar does not provide spatial
weights. Historical regional rows might be useful later, but require reconciliation
of monitored versus installed capacity, region coverage, MW/MWp/AC definitions,
upscaling populations and publication vintages. [B2]

Trustworthy regional sources exist: VEKA publishes PV counts and **inverter AC
capacity** by Flemish province/municipality; BRUGEL publishes cumulative capacity
by technology/municipality for Brussels; CWaPE supplies Walloon network/PV
statistics. The inspected CWaPE network dashboard includes <=10 kVA penetration,
which is not by itself a complete all-size capacity map. BRUGEL's 2025 report
published in July 2026 illustrates why reference year is not availability date.
No harmonized, national, historical-as-of grid-weight dataset was verified here.
Do not invent equal provincial capacity, use latest revised weights retrospectively,
or fit weights to test PV. Geographic aggregation is the defensible v1 choice.
[B3–B6]

## 13. Proposed minimal data contract

Keep one run manifest plus a wide native/cropped table and an aligned feature table;
no weather warehouse is needed.

**Run manifest:** provider/model/product, deterministic/member identity,
`forecast_reference_time` (UTC), model/process/table version, grid id,
source object URLs/keys, requested messages/steps, index/message checksums,
ETag/version id if supplied, retrieved time, source last-modified time,
variable units/levels/statistics, box/boundary/weight hashes, transformation
version, missing-step/field diagnostics, bundle completeness, and licence citation.

**Availability fields:** nullable `available_at_observed`, nullable
`available_at_assumed`, `availability_basis`, `availability_policy_id`, and
publication/first-seen evidence. Never populate the observed field with R+6h.

**Native cropped rows:** run id, R, grid-cell id/latitude/longitude, native valid
time, DSWRF native mean plus its start/end steps and processing type, instantaneous
TCDC and 2m TMP, units, and quality flags. Preserve both duplicated-cloud-message
selection criteria and radiation bounds; a wide row does not make their different
time statistics identical.

**Normalized rows:** run id, R, `[valid_start,valid_end)` in UTC, cell id,
`ghi_mean_w_m2`, cloud fraction, 2m temperature °C, QC/transformation id.

**Model feature rows:** SolarCast issue T, target interval start/end, selected run
id/R, effective availability plus its basis, lead, the seven spatial features,
and completeness/QC flags. All targets in an origin refer to one selected bundle.
Do not overwrite the same valid-time row when a newer reference time appears.

## 14. Backtester and model integration proposal

```text
Official GFS objects/indexes
  -> selected messages -> decode -> Belgium-only vintage cache
  -> issue-time run selector -> interval-aware transformations/spatial features
  -> trusted context builder with selected-run provenance
  -> future weather-aware LightGBM -> existing Forecast/evaluator
```

The backtester/context builder, not the model, owns vintage selection. A small
future-known weather payload should carry target-aligned features **and provenance**;
it must validate R, availability basis, policy id and exact targets before inference.
Training must reconstruct the eligible NWP vintage separately for every historical
origin, then join PV labels at fit time. No global “best available weather per
valid time” join is permitted. Plain arbitrary `context.features` cannot be the
trust boundary without those checks. The selector must be validated
and feature-ready tables first; model/context wiring belongs to a subsequent
an explicit design rather than an implicit change to the Forecaster protocol.

First controlled model experiment: an explicitly versioned weather-aware LightGBM
with the seven additional features, compared with a newly fitted history-only
control on **the same new training/validation/test populations**. Keep raw forecasts,
availability accounting and common-support scoring. Do not compare a weather
model trained through 2025 with an old control trained only through 2024 and call
the difference a weather effect.

### TSFM compatibility

| Family | Native future covariates? | Current SolarCast adapter |
|---|---|---|
| Chronos-2 | **Yes**: native past/future-known covariates | History-only; deliberately ignores feature table |
| t0-gamma | **Yes**: `future_covariates` over context plus horizon | History-only; no covariate argument exposed |

Official inference implementations support these pathways [T1,T2]. A later
weather-covariate experiment must cover the **full cutoff-to-final-target bridge**,
not just tomorrow's 92/96/100 rows, and explicitly handle required historical
covariate spans. A single 00Z run does not cover two weeks of past weather.
Chronos requires future-covariate keys to be present in its past-covariate mapping;
t0's array contract spans context plus horizon. Neither obligation is met by
simply adding tomorrow's weather to the current history-only adapter.
Do not silently fill missing future covariates with zeros or insert future
observations; upstream t0 treats future NaNs as zero, making validation important.
No existing adapter changes, new deep architecture or ensemble is proposed now.

## 15. Storage, bandwidth and access cost

Proposed local layout: cropped native **Parquet** partitioned by reference month
or run date, small JSON manifests and derived quarter-hour Parquet. Avoid millions
of one-cell files. Retain only a bounded temporary cache of selected GRIB messages
unless a user explicitly budgets for a larger raw archive; record original message
hashes/ranges before discarding temporary global fields. Zarr is unnecessary for
this small, tabular v1; GRIB remains the source encoding, not a Python model format.

Planning example, **731 days** (2024–2025), one 00Z run/day, **31 hourly endpoints**
(18–48), **209 box cells**, **3 variables**, float32:

```text
731 * 31 * 209 * 3 * 4 bytes = 56.8 MB of numeric values
```

With timestamps, masks, identities and manifests, budget roughly **0.1–0.5 GB**
for the cropped native/processed store, subject to a one-run measurement in
the historical GFS ingestion layer. Seven quarter-hour national features are about 2 MB of numeric values
over those two years. Four runs/day multiply the numeric budget by four; a full
15-day horizon multiplies it further and is not required.

**Transfer is much larger than local storage.** NODD S3 supports byte-range
selection of GRIB messages, not an arbitrary Belgium box inside a compressed
global message. The three selected June-2025 step-24 messages total **2,316,553
bytes**, versus about **549 MB** for its whole multi-variable GRIB file. Using
that one-step size as an explicit extrapolation gives about **52.5 GB** for
731×31 selected field sets, plus indexes/requests; actual packing varies by
lead/date/variable. Do not advertise the 56.8 MB local value budget as the download
volume or download the full 549 MB files. If a range request returns full-file
200 instead of the expected validated 206 range, stop rather than accepting it.

NCEP's recent NOMADS filter and GDEX's historical subset service can perform
spatial selection; neither is the same API as an old S3 object. A GDEX login/3h
subset route is a useful optional bandwidth-saving route, with its own frozen
resolution/semantics, if the hourly transfer budget is unacceptable. Do not
silently substitute it. No multi-year download is authorized by this research.

NODD HTTPS access needs no AWS account or key and has no published dataset access
charge. Local compute/network costs and any cloud compute/storage are separate.
Do not enable billable S3 compute or paid API plans as part of a data client.
[G1,G3,G4]

## 16. Licence and attribution

NOAA's NODD dataset terms permit public use, request NOAA attribution, prohibit
implied NOAA endorsement, and require modified data not to be represented as
unaltered original NOAA output. Cite NOAA/NCEP, access date and exact source;
identify cropping, interval conversion and spatial aggregation. These terms
support reuse and redistribution under the stated conditions, including derived
features, but do **not** turn NOAA data into SolarCast Apache-licensed code.
Preserve manifests and provider notices when sharing processed subsets. [G1]

GDEX separately labels its dataset CC BY 4.0; ECMWF Open Data uses CC BY 4.0 plus
ECMWF terms; Copernicus/ADS and Open-Meteo have their own dataset/access terms.
Check the actual route's terms rather than assuming a provider's model-code
licence covers data, API access or every third-party transformation. No legal
clearance for an individual commercial deployment is asserted. [G3,E1,O5]

## 17. Proposed periods, selected before weather scores

The old 2025 holdout is known. It can be reused only as a clearly labelled
development comparison, not a fresh confirmation for decisions informed by it.
A coverage-driven proposal is **2024–2025 training, Jan–Feb 2026 validation,
and March–September 2026 confirmation** (214 target days), before the October
2026 Elia schema seam. This spans 1004 calendar target dates across all splits
and several solar seasons, and uses about 78 MB of the above native numeric values
or roughly 72 GB of selected-message transfer under the same rough size assumption.
Warm-up PV dates and preceding forecast run/step support add a small margin.

For a concrete maturity-safe starting proposal, use training origins
`[2024-01-01,2025-12-30)` (729 target days, 2 Jan 2024–30 Dec 2025), fit at
1 Jan 2026 08:00Z, validation origins `[2026-01-01,2026-02-27)` (57 target days,
2 Jan–27 Feb), and test origins `[2026-02-28,2026-09-30)` (214 target days,
1 Mar–30 Sep). The deliberate boundary gaps keep validation target labels from
overlapping the first test issue. The 1004-date envelope above is a storage
planning envelope; the actual evaluated splits exclude those boundary gaps.

This proposal is **not a frozen benchmark yet**. The missing Nov–Dec 2025/2026
PV snapshots and every selected NWP vintage must first be audited through the
existing Elia contract and new NWP manifest. Preserve fit/label maturity gaps
and omit a boundary origin if its next-day targets overlap validation or test;
do not mechanically reuse every last training calendar date. If coverage prevents
this split, decide/freeze a new seasonally justified split before evaluating
weather-model errors, documenting the reason. Do not pick a period because NWP
looks better. No PV/NWP payload for this new period was downloaded here.

## 18. Ingestion invariants and offline validation

The ingestion contract comprises **one bounded GFS/NODD client**, optional decoding
dependencies, a small configurable Belgium subset, the three exact native
fields/levels/statistics, immutable source/run manifests and bounded local caching,
radiation-bound reconstruction, valid-time normalization, and a vintage selector
with distinct observed/assumed availability. Start with one or two tiny runs to
validate Windows decoding, units and actual message sizes; no bulk historical
job or weather model belongs in the first implementation acceptance step.

Mandatory offline fixtures/tests:

1. **Weather leakage canary:** an older complete 00Z run is eligible; a newer run
   has reference time before T but publication after T and contains `999999`.
   The selector/context table receives only the older run. At a later issue,
   the newer run may become eligible. Include a late revised copy of an old run.
2. Missing/intermediate message or partially published bundle cannot be selected
   merely because its final step exists; fallback/failure is recorded explicitly.
3. Exact `<=` availability boundary and separate PV safety-lag behavior; unknown
   historical availability fails strict mode and remains labelled in lag mode.
4. Duplicate TCDC instant/average/layer messages cannot be selected by short name
   alone; wrong level/unit/statistic/reference/grid is rejected.
5. DSWRF overlapping 18–24/24–25 windows, shared-start differencing, resets,
   missing predecessors, tiny packing errors and large negative energy cases.
6. Constant synthetic flux reproduces hourly energy after quarter-hour mapping;
   no cross-run interpolation or extrapolation, including endpoints beyond the
   PV valid instant whose forecast run was legitimately known.
7. 92/96/100 target grids, interval-end versus interval-start labels, UTC/DST
   and changed issue times; complete required endpoint coverage.
8. Spatial mask excludes foreign/sea box cells; weights sum to one within each
   national/band population and hashes change when geometry/grid changes.
9. Cache identity includes run, variable statistic, grid/box, version,
   availability policy, units and transformation; altered indexes/bytes fail.
10. HTTP range boundaries/status/length, rate limiting, missing objects and retry
    handling; no huge full-file fallback, credentials or paid service activation.

Source-preparation changes must not silently modify forecasters or evaluation.
Weather-informed comparisons use the same controlled forecasting framework; source
ingestion does not require an ensemble or a generic multi-provider hierarchy.

## 19. Remaining uncertainties

- Historical source/route first-publication times and outages are not fully
  reconstructed; six hours is an assumption, not a guaranteed upper bound.
- Metadata probes do not establish every date/step/variable's completeness,
  GRIB binary-header semantics, decoder compatibility or future retention.
- S3 metadata/objects can be copied or corrected; first-seen evidence and hashes
  must be retained, and new corrections cannot silently replace an old vintage.
- GFS/IFS cycles, grids and precision change; provider chronology must be recorded.
- Azure old-date/variable coverage and source-documentation discrepancies remain
  unresolved. Hindcast init dates must not masquerade as production dates.
- Geographic weights are not PV-capacity weights; exact polygon cell counts,
  boundary licensing and capacity-vintage reconciliation remain work for later.
- Hourly interpolation adds no cloud-motion information; radiation downscaling,
  PV orientation/temperature and national upscaling introduce representativeness
  error. GFS has not yet been evaluated for Belgian radiation or PV accuracy.
- New 2026 Elia coverage and unbiased confirmatory splits are proposals requiring
  a data audit, not already validated experimental results.

## 20. Official sources and probe record

All sources checked on 14 November 2026. Live status pages and cloud holdings
can change. No source implies a complete historical availability guarantee.

- **G1:** [NOAA GFS/NODD official AWS registry, terms and unsigned access](https://registry.opendata.aws/noaa-gfs-bdp-pds/).
- **G2:** [NCEP GFS products inventory and file naming](https://www.nco.ncep.noaa.gov/pmb/products/gfs/).
- **G3:** [NCAR GDEX operational GFS archive d084001, DOI, date span and licence](https://gdex.ucar.edu/datasets/d084001/).
- **G4:** [GDEX historical subsetting and access routes](https://gdex.ucar.edu/datasets/d084001/dataaccess/).
- **G5:** [NCEP example inventory: levels and instant/average fields](https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f003.shtml).
- **G6:** [NCEP shortwave parameters and units](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-4.shtml).
- **G7:** [NCEP cloud parameters](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-6.shtml), [temperature parameters](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-0.shtml).
- **G8:** [NCEP current production schedule/status; averages, not historical logs](https://www.nco.ncep.noaa.gov/pmb/nwprod/prodstat/).
- **G9:** [NOAA GFS native resolution and forecast-output frequency](https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php), [hourly 0.25° NOMADS product](https://www.nco.ncep.noaa.gov/pmb/products/gfs/nomads/).
- **E1:** [ECMWF Open Data products, terms, retention and October 2026 changes](https://www.ecmwf.int/en/forecasts/datasets/open-data).
- **E2:** [ECMWF operational archive](https://www.ecmwf.int/en/forecasts/datasets/operational-archive), [archive access](https://www.ecmwf.int/en/forecasts/access-forecasts/access-archive-datasets).
- **E3:** [ECMWF service pricing and discretionary waivers](https://www.ecmwf.int/en/forecasts/accessing-forecasts/payment-rules-and-options/tariffs).
- **E4:** [IFS/AIFS/CAMS dissemination schedule](https://confluence.ecmwf.int/spaces/DAC/pages/272310483/Dissemination+schedule).
- **E5:** [ECMWF technical Open Data documentation](https://confluence.ecmwf.int/spaces/DAC/pages/272310539/ECMWF+open+data+real-time+forecasts+from+IFS+and+AIFS), [official client](https://github.com/ecmwf/ecmwf-opendata).
- **E6:** [Microsoft Planetary Computer collection metadata, original-run fields and attribution](https://planetarycomputer.microsoft.com/api/stac/v1/collections/ecmwf-forecast), [ECMWF cloud mirror registry](https://registry.opendata.aws/ecmwf-forecasts/).
- **E7:** [ECMWF IFS/AIFS system chronology](https://www.ecmwf.int/en/forecasts/documentation-and-support/changes-ecmwf-model).
- **E8:** [IFS Cycle 49R1 operational implementation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=462890236).
- **E9:** [IFS Cy49R1 physical processes: accumulated surface flux diagnostics](https://www.ecmwf.int/sites/default/files/elibrary/112024/81626-ifs-documentation-cy49r1-part-iv-physical-processes.pdf).
- **C1:** [Copernicus ADS actual CAMS composition forecasts and catalogue](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview).
- **C2:** [ECMWF CAMS forecast documentation, variables, schedules and ADS area requests](https://confluence.ecmwf.int/spaces/CKB/pages/212454117/CAMS+Global+atmospheric+composition+forecast+data+documentation).
- **D1:** [DWD Open Data README/access terms](https://opendata.dwd.de/README.txt), [current ICON-EU tree](https://opendata.dwd.de/weather/nwp/icon-eu/).
- **D2:** [Official DWD ICON tutorial 2025](https://www.dwd.de/EN/ourservices/nwp_icon_tutorial/pdf_volume/icon_tutorial2025_en.pdf?__blob=publicationFile&v=3).
- **D3:** [DWD new feed tree](https://opendata.dwd.de/weather/nwp/v1/).
- **D4:** [Open-Meteo's own DWD integration/grid-migration notice](https://open-meteo.com/en/docs/dwd-api).
- **D5:** [DWD legal notice, inaccessible with 403 in this investigation](https://www.dwd.de/EN/service/legal_notice/legal_notice_node.html).
- **O1:** [Open-Meteo Single Runs: explicit run identity, hindcasts and dates](https://open-meteo.com/en/docs/single-runs-api).
- **O2:** [Open-Meteo Previous Runs: fixed offsets versus full runs](https://open-meteo.com/en/docs/previous-runs-api).
- **O3:** [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api).
- **O4:** [Open-Meteo availability metadata and server propagation](https://open-meteo.com/en/docs/model-updates).
- **O5:** [Open-Meteo hosted access limits, commercial tiers and data terms](https://open-meteo.com/en/pricing).
- **B1:** [Official Belgian land area](https://www.belgium.be/nl/over_belgie/land/belgie_in_een_notendop/fiche_belgie).
- **B2:** [Elia ods032 field metadata](https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032), [existing SolarCast Elia investigation](elia.md).
- **B3:** [VEKA regional/municipal PV capacity, AC definition and downloads](https://www.vlaanderen.be/veka/energie-en-klimaatbeleid-in-cijfers/toelichting-energiekaart-veka).
- **B4:** [BRUGEL capacity by technology/municipality and raw exports](https://brugel.brussels/actualites/mise-a-jour-des-donnees-energies-renouvelables-618).
- **B5:** [CWaPE Walloon statistics and scope](https://www.cwape.be/statistiques).
- **B6:** [BRUGEL 2025 report publication in July 2026](https://brugel.brussels/actualites/publication-du-cahier-thematique-2025-consacre-aux-energies-renouvelables-745).
- **T1:** [Official Chronos-2 native inference/covariates](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py).
- **T2:** [Official t0 native covariate contract](https://github.com/theforecastingcompany/tfc-t0/blob/main/t0/forecaster.py).

Personally retrieved NOAA indexes include
[June 2025 step 24](https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20250610/00/atmos/gfs.t00z.pgrb2.0p25.f024.idx),
[step 25](https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20250610/00/atmos/gfs.t00z.pgrb2.0p25.f025.idx)
and [step 27](https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20250610/00/atmos/gfs.t00z.pgrb2.0p25.f027.idx).
Small local research records are under ignored `artifacts/research/weather/`;
they contain indexes/catalogue/HEAD evidence, not weather grids or user secrets.

## 21. Investigation validation record

The source investigation preserved existing forecasting behavior, benchmark
configurations and stored prediction artifacts. The following dated verification
record describes the software state used for the investigation, not current test totals.

After documentation updates:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 351 passed, 2 explicitly opt-in native tests skipped (14.46 seconds)
.\.venv\Scripts\python.exe -m ruff check .
# Passed
.\.venv\Scripts\python.exe -m ruff format --check .
# 73 files already formatted
.\.venv\Scripts\python.exe -m mypy
# Passed: 41 source/test files
```

Checksum audits confirmed the original LightGBM and zero-shot TSFM source/artifact
identities. These records do not assert historical provider publication availability.


## 22. Historical GFS ingestion

Implemented 14 November 2026, without changing the selected provider/product or
starting weather-informed modeling. Sections 1–21 above are the preserved research
record. This section describes the current implementation and supersedes their
statements that ingestion/context integration are future work.

### Provider contract and access

`solarcast.data.weather.GFSClient` implements **NOAA/NCEP operational deterministic
GFS atmospheric `pgrb2.0p25`**, public NODD bucket `noaa-gfs-bdp-pds`. No account,
AWS SDK, token or environment credential is required. Official sources rechecked
before implementation:

- [NOAA NODD GFS registry, access and terms](https://registry.opendata.aws/noaa-gfs-bdp-pds/).
- [NCEP operational product inventory](https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f003.shtml).
- [NCEP GFS system description](https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php).

Modern HTTPS paths are
`gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF`, plus `.idx` inventories.
Implementation deliberately supports 2024+ 00/06/12/18Z cycles and consecutive
hourly steps 1–120, **at most 49 files per call**. The older path layout, step zero,
three-hour long leads and other products are rejected, not silently reinterpreted.

Only these exact messages are selected: surface **DSWRF average** in W/m²,
entire-atmosphere **instantaneous TCDC** in percent, and instantaneous **TMP at
2 m above ground** in K. Average TCDC and convective-layer TCDC are excluded.
The live probe found DSWRF encoded as NCEP local parameter `(0,4,192)`, short name
`sdswrf`; the decoder accepts that verified local encoding or standard `(0,4,7)`.
It checks centre, operational/forecast status, reference/valid times, steps and
statistical type, units, parameter, level and regular global 1440×721 quarter-degree
grid. TMP uses `typeOfLevel=heightAboveGround`, level 2; ecCodes' first-fixed-surface
string alone is insufficient to identify it. This refines the planned decoder;
it does not change the selected variables or their physical meaning.

Indices are capped at 512,000 bytes. The client reads exactly three bounded global
message ranges per step, never a complete global GRIB file. Each range defaults
to at most 8 MB; total streamed bytes per run are capped at 128 MB. HTTP 206,
Content-Range, lengths, complete GRIB message framing and a stable per-object ETag
are required. HTTP 200 for a range is rejected **before its body is read**.
Defaults: explicit 60-second HTTP timeout, two retries for transport errors and
429/500/502/503/504, bounded 1/2-second backoff, 0.15-second request spacing.
No redirects or full-object fallback. These client bounds are SolarCast guards,
not a claimed NOAA rate-limit agreement. Indices and message SHA-256 hashes,
ETags, Last-Modified and byte ranges are retained as provenance.

### Installation and optional dependencies

```powershell
python -m pip install -e ".[dev,weather]"
python examples/fetch_weather_vintage.py --live
```

The `weather` extra adds ecCodes `>=2.50,<2.51` (GRIB), PyArrow `>=26,<27`
(Parquet), Shapely `>=2.2,<2.3` (polygon intersections), and pyproj `>=3.7,<4`
(equal-area projection; supported version resolves according to Python version).
Heavy imports occur only when those operations are requested. Core dependencies
are unchanged; history-only forecasting needs none of these packages. Native
Windows Python 3.12 wheels were exercised successfully. Optional Linux/Windows CI
jobs run offline fixtures, including tiny genuine-format GRIB messages; provider
HTTP is never part of CI.

### Native schema versus model features

`ForecastRun` owns a defensive copy of this cropped-native table:

| Column | Meaning |
| --- | --- |
| `reference_time` | One initialization, aware UTC |
| `valid_time` | Native hourly endpoint, aware UTC |
| `forecast_lead` | Proper duration, exactly valid minus reference |
| `latitude`, `longitude` | Individual 0.25° grid centres, east-positive longitude |
| `dswrf_start_hour`, `dswrf_end_hour` | Native radiation averaging bounds relative to initialization |
| `dswrf_average_w_m2` | Original provider running interval average, W/m² |
| `cloud_percent` | Native instantaneous entire-atmosphere TCDC, percent |
| `temperature_2m_k` | Native instantaneous TMP at 2 m, Kelvin |

Run attributes retain provider/product metadata, source-version fingerprint,
nullable `observed_available_at`, revision flag and nullable correction-version
availability. The downloader leaves observed availability **null**. Naive times,
duplicate keys, inconsistent leads/bounds, invalid cloud percentages and nonpositive
Kelvin values fail. Native missing values remain missing; no observation, reanalysis,
zero or different-vintage substitution occurs. Coverage is checked separately for
the requested target period, so a bounded subset is not mislabeled a full run.

`hourly_radiation`, `aligned_cells` and `weather_features` are separate functions;
the downloader does not aggregate away spatial information. `WeatherCovariates`
owns only target times plus the seven numerical features specified in section 11,
and carries issue/reference/availability, basis, policy, source version, spatial
weights hash and incomplete-candidate exclusion notes. Features use W/m², cloud
fraction and °C. No raw GRIB or future PV labels enter a forecast context.

### Radiation and quarter-hour alignment

The hourly flux reconstruction for same-start running averages is
`(e-s)*A(s,e) - (e-1-s)*A(s,e-1)` in W/m²; multiplying by 3600 gives that hour's
energy in J/m². A one-hour average after a reset is used directly. Each call
handles one run, each grid cell separately. It cannot difference across runs or
across six-hour reset windows. Missing required predecessors remain NaN and fail
requested coverage. Negative reconstructed values fail with an explicit error:
v1 has zero negative tolerance and **does not clip packing artifacts**.

For a PV interval `[t,t+15min)`, take the enclosing reconstructed hourly flux
unchanged. Four quarters conserve hourly energy; they do not imply resolved
15-minute radiation physics. Cloud and temperature instead use the mean of linear
interpolation between enclosing hourly instantaneous states, equal to the
interpolated midpoint. Cloud is divided by 100 and temperature has 273.15 subtracted.
Both hourly state endpoints must exist; there is no extrapolation. Sparse requested
targets are supported, and Brussels next-day schedules retain 92/96/100 slots.
Fetch raw hourly endpoints from `floor(first_target)` through
`ceil(last_target + 15min)` relative to the chosen initialization. The first native
radiation hour may be unreconstructable but is not needed when it ends at the
first target's containing-hour start. This is not a full-run missingness claim.

### Belgian spatial definition

The authoritative `GRID` is N51.75/W2.25/S49.25/E6.75 with 0.25° centres:
**209 points**. All are preserved and required at requested hourly endpoints,
including the surrounding margin. Spatial features use positive intersections
between their 0.25° cell footprints and a pinned Belgian country polygon, projected
to **EPSG:3035** before measuring area. The seven features comprise west/central/east
GHI means, Belgian GHI spatial population standard deviation, Belgian cloud
fraction mean and population standard deviation, and Belgian temperature mean.
Longitude bands remain `<4`, `[4,5)`, `>=5` degrees east; each mean normalizes its
own positive areas. No capacity weighting is invented.

the weather-source investigation required a pinned licensed boundary but did not select its source.
Implementation uses [Natural Earth v5.1.2 10m countries, Belgium geometry](https://github.com/nvkelso/natural-earth-vector/blob/v5.1.2/geojson/ne_10m_admin_0_countries.geojson),
whose [public-domain terms](https://www.naturalearthdata.com/about/terms-of-use/)
allow redistribution and commercial use. The packaged 15 KB GeoJSON is a derived
Belgium-only extraction, with source/version/license embedded. This is a
cartographic boundary, **not an official Belgian cadastral boundary**; its
generalization is an explicit approximation. GISCO was not vendored with ambiguous
commercial-use restrictions. Boundary SHA-256:
`bd49151ec45a9b42c63140ce908294c6e34e215583fbf36dbd61e51ec60b3286`.
There are **89 positive-area cells**, totaling approximately 30,669.78 km².
The weights fingerprint includes actual areas, grid, projection, boundary hash and
transform version; the verified Windows result is
`15c6dbc8cef0e928ceac6fc2afd8c55003bdbf1aeca2978e4ec3cc98b65401f1`.
Numerical library changes can change this fingerprint and must be disclosed.

### Availability and whole-vintage selection

Default `NWPAvailability()` is **`gfs_init_plus_6h_v1`**, with basis `assumed`.
Eligibility is inclusively `reference + 6h <= issue`. At 09Z/10Z, same-day 00Z
qualifies, 06Z does not qualify until 12Z. This approximates availability of the
**entire requested bundle**, not just its final file. It does not establish that
all historical messages were actually published on time. Custom nonnegative
delays have distinct policy names. `NWPAvailability("observed")` instead requires
caller-supplied evidence for complete-bundle availability; unknown evidence rejects
the run. Neither mode uses retrieval time or S3 Last-Modified as publication time.

`select_nwp_vintage` filters eligibility before feature construction, sorts eligible
initializations newest first, and selects the newest **complete whole vintage** for
all targets, variables and cells. An incomplete candidate can cause a documented
whole-run fallback to an older complete eligible run; exclusion reasons accompany
the selected payload. It never stitches different runs. No eligible complete run
raises `WeatherError`. Multiple eligible source versions of one cycle are ambiguous
and rejected rather than sorted accidentally. Known revisions additionally require
explicit version-availability evidence and cannot be backdated to initialization;
unknown correction timing rejects a revision in either mode.

Important limitation: the public archive alone does not prove that a retrieved
object is the original historical copy. Lag mode does **not** certify operational
as-of reproducibility. Declared revision guards cannot discover undisclosed archive
replacements. Strict observed mode is intentionally unusable without external,
defensible complete-bundle publication evidence. Cached availability assertions
are not fabricated or persisted as provider facts.

### Cache correctness

`WeatherCache(root)` stores one immutable directory per canonical request-identity
SHA-256, containing `native.parquet` and `manifest.json`. Identity covers provider,
product, UTC initialization, exact native steps, three variables/levels/statistics/
units, bounding grid, transformation and decoder-contract versions. Source-version
identity additionally hashes source index/message digests and response provenance.
The manifest records retrieval time separately from null observed availability.
Reload checks request identity, Parquet checksum, source-version fingerprint and
native schema/time invariants. Changed variables, steps, grid or run cannot share
a key. Partial/corrupted entries fail explicitly; there is no automatic destructive
refresh. Writes stage in a sibling directory then rename atomically; conflicting
source versions are rejected. Only cropped values are persisted, not global GRIB.
Root `data/` and generated `artifacts/` are already Git-ignored. Tiny test fixtures
and the public-domain boundary are deliberately versionable package resources.

### ForecastContext/backtester integration

```python
from solarcast.data.weather import WeatherStore, belgium_weights

store = WeatherStore(tuple(local_runs), belgium_weights())
result = backtest(model=model, data=pv, spec=spec, weather_provider=store)
# Engine: store.for_issue(request.issue_time, request.target_times)
# Model: context.weather.features, or context.weather is None
```

`backtest` accepts an optional structural weather provider and invokes it before
`model.predict`; the context validates exact issue/target agreement and the payload
enforces eligible availability. Weather-selection failures use existing strict or
continue failure handling. Weather-enabled origin tables retain run/reference/
availability/basis/policy/weights and exclusion notes. Without a weather provider,
weather is `None` and existing output schemas are unchanged. PV availability remains
the existing, separately selected policy. Models do not fetch weather archives.
Persistence, history-only LightGBM, Chronos-2 and t0-gamma algorithms were not changed
and do not consume weather. Arbitrary custom providers remain trusted caller code,
just as PV availability policies do; validation is not a security sandbox.

Adding optional fields changes frozen benchmark **source** fingerprints in
`backtesting.py` and `models/types.py`. Existing stored prediction artifacts and
configurations are preserved. Historical results remain tied to their recorded
source version; frozen evaluation guards must not be bypassed or manifests rewritten
to pretend the current source is identical. No benchmark or model fitting was run.

### Bounded live smoke actually performed

`examples/fetch_weather_vintage.py` defaults to the historical issue
2025-06-10 11:00 Brussels (**09Z**), chooses 00Z by the declared policy, and requests
two afternoon hours of tomorrow. It performs no network access without `--live`.
`--full-day` explicitly expands to the full next local day; it does not trigger a
multi-day download. `--issue`, `--cache`, and `--output` are configurable.

On Windows Python 3.12 the opt-in smoke fetched steps **36/37/38**, exactly nine
selected messages, totaling **6,740,262 transferred bytes including indices**.
It verified GRIB metadata/units, cropped **627 rows = 3×209 cells**, persisted and
reloaded Parquet, then selected the run for eight targets
**2025-06-11 12:00–13:45 UTC**. The selected availability is explicitly **assumed
2025-06-10 06Z**, not observed. Both the reset at step37 and running-average
reconstruction at step38 occur in this sample. Band GHI ranges approximately
830.41–893.42 W/m², national cloud mean .0640–.1091, and national temperature
20.79–21.60°C. These are smoke diagnostics, not forecast skill results.
Local outputs: `data/weather_gfs_v1/<identity>/`, and
`artifacts/research/weather/prompt17/{smoke.json,features.csv}`. This is the
verbatim historical output directory; the current example defaults to
`artifacts/research/weather/gfs_vintage/`. A separate initial
three-message contract probe is also local/ignored. No years of weather were fetched.

### Verification, acceptance and limits

Offline tests include real cropped normalized values; three 179–203-byte modified
constant GRIB2 fixtures retain actual source metadata and exercise ecCodes when
installed. Synthetic tests hand-check running averages, reset windows and energy,
midpoint state integration, area-weighted population statistics, missing variables/
times/cells/predecessors, duplicate keys, UTC and forecast leads, inclusive availability
in both modes, revision/ambiguity rejection, immutable cache corruption/identity,
bounded HTTP retries/ranges/body limits and no-body full-file rejection.

The mandatory **`test_mandatory_weather_leakage_canary_through_backtester`** puts
999999 W/m² in every cell of the newer 06Z run. At 09Z its assumed availability
is still future; the selector and model context receive only older 00Z values.
At 12Z the newer run qualifies and its sentinel is accepted, proving exclusion
depends on availability rather than hard-coded sentinel removal.

All 15 scientific acceptance requirements are covered: reference preserved;
separate valid time; explicit policy; no fabricated observed availability; future
runs excluded; sentinel canary; verified units/statistics; radiation/reset handling;
explicit Belgium subset; variable-aware alignment; no silent missing replacement;
run-specific cache; optional context weather; existing history-only models;
and provider-network-free CI. Verification commands/results are recorded
in the verification record below.

Limitations remain material: approximate historical availability and unknown archive
revisions, coarse NWP grid/cartographic boundary, strict failures for negative
packing artifacts, no subhourly NWP physics, no spatial capacity weights, no long
leads/step zero/old archive layouts, no model-side missingness policy and no bulk
experiment scheduler. The live check covered one initialization and two daytime
hours, not a full seasonal archive or actual publication latency. A source contract
change fails validation instead of silently adapting fields. The [weather-informed
experiment](../experiments/weather-lightgbm-v1.md) separately evaluates predictive value.


## 23. Historical GFS verification record

Commands ran from the repository root. Base environment intentionally has no
weather binary libraries; the separate Windows `weather-smoke` environment has
`.[dev,weather,lightgbm]`. All pytest commands below use offline fixtures.

| Exact command | Result |
| --- | --- |
| `.\.venv\Scripts\python.exe -m pytest -q` | 413 passed, 8 skipped (two native TSFM opt-ins, six optional weather tests) |
| `.\.venv\weather-smoke\Scripts\python.exe -m pytest -q` | 419 passed, 2 skipped (native TSFM opt-ins only) |
| `.\.venv\weather-smoke\Scripts\python.exe -m pytest tests/test_weather.py tests/test_weather_gfs.py -q` | 68 passed |
| `.\.venv\Scripts\python.exe -m pytest tests/test_persistence.py tests/test_lightgbm.py tests/test_chronos.py tests/test_t0.py -q` | 108 passed; actual installed LightGBM and fake-runtime TSFM regressions, no new weight downloads |
| `.\.venv\Scripts\python.exe -m pytest tests/test_weather.py::test_mandatory_weather_leakage_canary_through_backtester -q` | 1 passed |
| `.\.venv\Scripts\python.exe -m ruff check .` | Passed |
| `.\.venv\Scripts\python.exe -m ruff format --check .` | 85 Python files formatted |
| `.\.venv\Scripts\python.exe -m mypy` | Passed, 51 source/test files |
| `.\.venv\weather-smoke\Scripts\python.exe -m mypy` | Passed, 51 source/test files |
| `.\.venv\Scripts\python.exe -m build` | Wheel and sdist built; packaged GeoJSON verified |
| `.\.venv\weather-smoke\Scripts\python.exe examples/fetch_weather_vintage.py --live` | Bounded live smoke described above, then cache-hit repeat |
| `.\.venv\weather-smoke\Scripts\python.exe examples/fetch_weather_vintage.py` | Exits without network access |

Artifact checksum audit: **47/47** the zero-shot TSFM experiment result-manifest entries and **34/34**
the history-only LightGBM experiment recorded artifacts still match. Expected recorded source drift is
`backtesting.py` and `models/types.py` for the parent benchmark, plus `pyproject.toml`
for the TSFM freeze. Those old identities and outputs were not rewritten.


## 24. Radiation precision metadata and explicit bounded policy

Full-day winter QC before real-data model training exposed negative reconstructed
radiation values of approximately −0.14 W/m² that the default strict radiation policy
correctly rejected. The decoder now preserves DSWRF packing type, binary/decimal
scale factors, bits per value, reference value and lattice quantum. Native cache
identity uses `eccodes-gfs-v2-packing`; previous immutable entries are retained.
This adds provenance, without changing the selected source, variables or units.

The **default remains `radiation_policy="strict"`**. the weather-informed LightGBM experiment explicitly opts into
`"grib_packing_bound_v1"`: propagate one full per-message lattice quantum through
the running-average difference, plus float64 roundoff. Only negative radiation
within that bound is set to zero, with each adjusted native cell-hour counted.
Unknown precision evidence and negatives beyond the bound still fail. Native
averages remain unchanged. This is an explicit input-precision policy, not missing
weather filling or PV prediction clipping. No future observations, reference-time
changes, cross-vintage mixing, spatial changes or extra weather variables are added.

`hourly_radiation`, `aligned_cells`, `weather_features`, `select_nwp_vintage` and
`WeatherStore` accept the named policy. `WeatherCovariates` and weather-enabled
origin/training audits retain policy and adjustment counts. Prepared experiment
features have a separate checksummed identity that includes the frozen policy.
The [weather-informed LightGBM experiment report](../experiments/weather-lightgbm-v1.md#pretraining-data-quality-correction-and-operational-amendments)
records the measured packing evidence, propagated formula, pretraining plan
amendments and limits. All original strict-policy and vintage-canary tests remain.
