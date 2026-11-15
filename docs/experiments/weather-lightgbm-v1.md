# Weather-informed LightGBM: controlled development benchmark v1

## Questions and evidence status

This experiment asks whether historical operational NWP adds more forecasting
value than switching from history-only LightGBM to the best zero-shot TSFM, and
how weather-LightGBM compares with seasonal persistence and history-only models.
No winner is assumed. This is **a development comparison**: the March–October
2025 scores from the history-only LightGBM and zero-shot TSFM experiments are already known. Reusing their targets permits
an exact cached TSFM comparison; it does not create an untouched confirmatory
holdout or establish provider-pretraining independence.

The methodology below was fixed before real-data weather-model fitting or scoring.
Acquisition is bounded by this experiment, not an unrestricted archive download.
The final results and interpretation are recorded after the run in the results
section below. No ensemble or new deep model is part of this experiment.

## Frozen data and schedules

Configuration: `configs/experiments/weather_lightgbm_v1.toml`, inheriting the
hash-pinned `lightgbm_v1.toml` and authenticating the original zero-shot TSFM experiment configuration,
freeze and results manifest. The expanded plan is saved before acquisition;
source, runtime and prepared-weather hashes are frozen before fitting; validation
selection and fitted model hashes are frozen before test inference.
The final run also saves the exact TOML, an authenticated source-code ZIP and
installed-package listing, so the untracked/no-HEAD Git state does not leave hashes
as the only record of the implementation.

| Setting | Fixed value |
| --- | --- |
| Elia data | Original national `ods032.measured`, interval-start average MW, 18 December 2023–1 November 2025 Brussels, end exclusive |
| Training origins | `[2024-01-01, 2024-12-30)`, daily: 364 origins |
| Fit boundary | 2025-01-01 08:00 UTC, unchanged |
| Validation origins | `[2025-01-01, 2025-02-27)`, daily: 57 origins |
| Test origins | `[2025-02-28, 2025-10-31)`, daily: 245 origins |
| Test targets | March–October 2025, next Brussels calendar day |
| Issue clock | 11:00 Europe/Brussels: 09Z summer, 10Z winter |
| PV availability | Completed intervals by floor(issue minus 30 minutes) |
| Context | Issue-anchored 14 elapsed days; no change to historical features or TSFM rollouts |
| Primary reference/rank | Exact elapsed 48h persistence; lower common-support all-interval RMSE |
| Secondary population | Existing Brussels reference-point apparent elevation >0 at interval start |
| Clipping | Raw model predictions; no nighttime removal or prediction clipping |

The existing snapshot is hash-checked before use, never silently refreshed or
replaced with synthetic data. No Elia forecast/capacity columns or observed future
weather enter model features. Historical PV publication latency/revisions remain
uncertain; completion plus 30 minutes is an approximation.

## NWP inputs, vintage policy and alignment

Use the [implemented GFS contract](../data/weather.md#22-historical-gfs-ingestion):
NOAA/NCEP operational deterministic `pgrb2.0p25` atmospheric output, public NODD
bucket `noaa-gfs-bdp-pds`. Retrieve only surface DSWRF interval averages in W/m²,
instantaneous entire-atmosphere TCDC in percent, and instantaneous 2m TMP in Kelvin.

For every training, validation and test issue, candidate initialization is the
latest six-hour UTC cycle satisfying `reference + 6h <= issue`. The unchanged
the historical GFS ingestion layer selector validates complete coverage and inclusive eligibility. The
declared policy is `gfs_init_plus_6h_v1`, basis **assumed**: all current issues
select same-day 00Z. Availability is not inferred from retrieval or Last-Modified.
The archive does not certify original historical as-of copies or publication time.

Native reference/valid times, lead durations, radiation averaging windows and
209 Belgian bounding-grid centres remain in immutable native Parquet caches with
source-message hashes. Prepared features are cached separately with exact issue,
targets, availability/basis, run version, grid-weight identity and precision audit.
Neither training nor prediction performs network access. Missing runs, endpoints,
variables or cells fail the **whole origin**, with explicit accounting; there is
no future-run, observation, reanalysis, zero or partial-run substitution.

DSWRF running averages are differenced only within the same start window/run/cell;
one-hour windows after resets are used directly. Reconstructed hourly means repeat
across four quarters, conserving energy. Cloud and temperature use the interval
mean of linear interpolation between hourly instantaneous states, then convert
percent to fraction and Kelvin to Celsius. The model does no further interpolation.

Spatial aggregation is unchanged: pinned Natural Earth v5.1.2 Belgium intersections
with cell footprints, measured in EPSG:3035, with 89 positive-area cells. West is
longitude <4°, central [4°,5°), east >=5°. Means use normalized intersection areas;
standard deviations are weighted population values. No capacity weights are used.
Verified weights identity:
`15c6dbc8cef0e928ceac6fc2afd8c55003bdbf1aeca2978e4ec3cc98b65401f1`.

## Pretraining data-quality correction and operational amendments

The first full-day winter acquisition exposed negative reconstructed radiation
down to approximately −0.14 W/m², causing the historical GFS ingestion layer's strict policy to reject the
first winter origins. This was discovered from weather-input QC, **before any
real-data weather-model fitting, validation scoring or test scoring**. The strict
policy was not silently relaxed and no PV outcome was used to choose a tolerance.

Probed 2024-01-01 DSWRF messages at steps 40–42 use
`grid_complex_spatial_differencing`, binary scale factor 1 and decimal scale
factor 2. Their decoded lattice quantum is `q=2^1 * 10^-2 = 0.02 W/m²`.
ecCodes' `packingError` key is absent for these messages; the implementation
does not pretend it was observed. [WMO GRIB packing regulations](https://codes.ecmwf.int/grib/format/grib2/regulations/)
define the scale representation, and [ECMWF's packing/differencing explanation](https://confluence.ecmwf.int/plugins/viewsource/viewpagesrc.action?pageId=208501579)
describes the origin of small negative increments in separately packed fields.

The explicit opt-in **`grib_packing_bound_v1`** policy preserves packing metadata
per message. It permits one full lattice quantum of uncertainty per native average
(a conservative policy, rather than an assumed nearest-rounding half-quantum).
For duration `d`, propagated bound is `d*q_end + (d-1)*q_previous`, plus a small
float64 arithmetic allowance. Only reconstructed negative values within that
bound are set to zero; every cell-hour adjustment is counted in provenance.
Larger negatives, absent packing evidence or unknown packing types still fail.
Native averages are untouched. This is radiation-input precision handling,
**not PV prediction clipping**, nighttime masking or missing-value filling.
The default radiation policy remains strict. Tests cover both behaviors.

The decoder cache contract was advanced to `eccodes-gfs-v2-packing`; previous
native cache versions and the original strict-plan failure records were preserved.
`plan_initial_strict.json` and `pretraining_amendment.json` document this correction.
A two-worker pilot projected several hours, so acquisition alone was subsequently
bounded at eight independent workers on the eight-logical-CPU host, each with
at least 0.3 seconds between requests (aggregate launch bound about 26.7/s).
`plan_two_workers.json` and `operational_amendment.json` preserve that change.
Periods, feature definitions, availability, model parameters and scoring were
unchanged by the operational amendment. No model result motivated either amendment.

## Model features and controlled validation ablation

One `LightGBMForecaster` supports explicit feature configurations; its existing
default remains history-only, with model ID `lightgbm`. `context.features` remains
ignored. Weather-informed configurations require validated `context.weather`.

| Feature category | Inputs |
| --- | --- |
| Future-known deterministic | Local time-of-day and day-of-year sine/cosine; reference-point solar elevation/azimuth and Haurwitz clear-sky GHI; elapsed forecast horizon |
| Historical-observed | Exact 24h/48h/7D PV lags; cutoff-anchored trailing-24h PV mean/max |
| Future forecast covariates | Three regional GHI means and Belgian GHI spatial standard deviation; Belgian cloud fraction mean/std; Belgian 2m temperature mean in °C |

The four predeclared validation configurations are:

1. `history`: the original 13 features.
2. `history+radiation`: original features plus four GHI features.
3. `history+radiation+cloud`: plus the two cloud features.
4. `history+nwp`: all seven approved weather features.

No NWP age/lead metadata features or additional cloud layers are added. NWP lead
is preserved in native data/provenance; the existing target-minus-issue horizon
remains a deterministic feature. Temperature is selected only if the predefined
validation comparison favors the complete feature set.

All configurations use exactly the original CPU LightGBM parameters: squared-error
objective, 100 trees, learning rate .05, 15 leaves, minimum child 20, seed 0,
one thread, deterministic column-wise training, native missing-history handling,
no early stopping, feature search, regularization search or final-test retuning.

Training weather is selected at **each historical training issue**, not at fit
time or target time. Labels join separately and must be available at the fit
boundary. If training weather is missing, its origin is excluded with an audit.
For a controlled information ablation, **all four models use the same remaining
finite-label training rows**, including the history-only comparator. This avoids
confounding added information with different training support. The original
full-training history-only predictions are retained and compared separately; when
no weather training origins fail, predictions must reproduce them within 1e-9 MW.

Choose the weather variant with lowest validation **common-support RMSE**, breaking
an exact tie in favor of fewer features. History is reported as an ablation
reference even if every weather variant is worse; it is not used to hide an
unfavorable weather result. The selected weather variant and all fitted model
hashes are frozen before the first weather-model test inference. No final-period
refit or validation-label refit is performed.

## Comparators, support and cache integrity

Primary comparison: exact 48h persistence, paired-training history-LightGBM,
**t0-gamma** (best aggregate zero-shot model in the zero-shot TSFM experiment), and the selected
weather-LightGBM. Chronos-2 remains in an extended table at negligible cache cost.

TSFM checkpoints, revisions, float32 CPU/native-P50 settings, disabled t0
adjustments, 1344-slot cap and 14D issue-anchored history are unchanged. Reuse
requires the original config/freeze/results hashes, model-cache hashes, exact
23,520 issue-target keys, exact 245-origin population and checkpoint/context
metadata. The old frozen source is authenticated as the provenance of those
cached predictions; it is not misrepresented as identical to today's source.
No old artifact or source identity is rewritten to bypass its guards.

All primary tables use exactly the same finite forecast/actual pairs across all
four models. Independent availability is reported over all scheduled origins.
Failed weather origins retain their target rows with failed status and missing
predictions. The daylight table is additional; primary scores retain night and
all DST targets. Bias is prediction minus actual. RMSE skill is relative to 48h
persistence on the same population. Absolute/relative MAE/RMSE changes and RMSE
skill changes isolate weather information and compare it with switching models.

## Predeclared diagnostics and plots

- Reuse elapsed six-hour horizon bins, exact horizons, Belgian local target hours,
  calendar months and seasons. Preserve counts and common support in every table.
- Define forecast-cloud regimes before scoring: low <.25, medium [.25,.75),
  high >=.75 national cloud fraction. Separately label forecast spatial cloud
  variability as standard deviation >=.15 versus <.15. Report all-interval and
  daylight regime scores. These are forecast-only proxies, not observed weather,
  temporal-variability measurements or meteorological event classifications.
- Score each common target day and assign fractional equal-day win credit within
  1e-9 MW daily RMSE. Also report sole wins.
- Compute Pearson signed-error correlations on common quarters and daylight, plus
  daily-RMSE correlations. Complementarity does not establish ensemble improvement.
- Plot fixed representative days 15 March, 15 June and 15 September. An unavailable
  common-support date is shown as unavailable, not replaced by a favorable day.
- Select the challenging-day plot by **largest daylight mean forecast cloud spatial
  standard deviation**, breaking ties chronologically. This uses NWP forecasts,
  not model errors or future observed weather. The proxy does not prove actual
  conditions were challenging or that GFS forecast them correctly.

## Runtime, artifacts and reproduction

The weather environment uses the existing optional `weather` and `experiments`
extras; no new dependency is needed. Hardware, Python and package versions are
recorded in `pre_fit_freeze.json`. Native archive transfer/decoding, spatial/temporal
preparation, training-matrix preparation, model fit and predict are measured
separately. `Forecaster.predict` includes feature construction/translation, but
excludes engine context selection and prepared-weather cache loading. Cached
TSFM times are explicitly the original zero-shot TSFM experiment CPU run, not contemporaneous
reruns or equal-thread microbenchmarks. No reliable peak-RAM measurement is claimed.

```powershell
python -m pip install -e ".[dev,weather,experiments]"
python -m pip install "lightgbm==4.7.0" "numpy==2.5.4" "pandas==3.1.1" "pvlib==0.16.1"
python scripts/benchmark_weather_lightgbm.py --config configs/experiments/weather_lightgbm_v1.toml
```

Stages can be separated with `--prepare-only` and then `--prepared`; the latter
uses only local prepared weather. `--evaluate-only` authenticates and re-evaluates
saved predictions without fitting, downloading weather or loading TSFM weights.
Completed weather success/failure manifests are resumable; corruption and partial
entries are explicit errors. Existing final inference cannot be overwritten.

The v1 configuration requires the original local LightGBM/TSFM experiment snapshot and frozen
artifact bundles, deliberately stored outside Git. On a fresh clone, obtain those
immutable inputs, or create a separately named replication configuration pinned
to an independent parent run. Regenerated timestamps/hardware metadata do not
produce the same original artifact hashes; do not rewrite v1 pins to claim they do.

Artifacts live under ignored `artifacts/experiments/weather_lightgbm_v1/`: plans,
amendments, freezes, source/runtime metadata, weather/training availability,
validation ablation, training audits/model files, prediction caches, primary/
daylight/extended leaderboards, information-value comparisons, horizon/local-hour/
month/season and regime tables, daily winners, correlations, compute/data overhead
and representative/challenging-day plots. Native caches are under
`data/weather_gfs_v1/`; prepared numerical covariates under `data/weather_features_v1/`.

## Limitations

This development test cannot serve as independent confirmation after prior results
were known. Availability and archive-vintage fidelity are approximate; pretraining
overlap is unknown; the PV target is an estimated/upscaled aggregate and fleet
capacity changes are not explicitly modeled. Weather-LightGBM learns from imperfect
GFS forecasts, so PV errors can combine weather and PV-mapping errors. No observed
weather was added to attribute those components or verify meteorological skill.

The coarse 0.25° grid, generalized country boundary and area rather than capacity
weights are approximations. Quarter-hour alignment preserves semantics, not new
subhourly atmospheric information. Packing-bounded radiation adjustments are
explicit uncertainty handling, not recovered unquantized truth. Missing-origin
exclusions can select weather conditions and must be read with availability counts.
Validation feature selection can overfit one short winter period; no statistical
significance or causal improvement is claimed from diagnostic subgroup scores.
History-LightGBM receives deterministic calendar/solar features whereas the TSFMs
receive PV history alone, and squared-error training differs from native-P50 TSFM
point forecasts. Their comparison is between fixed forecasting approaches, not
an architecture-only causal experiment. The paired LightGBM ablation more directly
isolates the added forecast covariates within one fixed learning procedure.
Global-message range retrieval adds substantial transient bandwidth even though
stored Belgian subsets are small. No ensemble, probability forecast or production
deployment conclusion follows from this experiment alone.

## Results

The controlled run completed with **666/666 usable weather origins**: all 364
training, 57 validation and 245 test issues. No weather origin, model forecast or
test actual was missing. The primary common population is all **23,520** scheduled
quarter-hours, including the 92/100-slot DST target days; daylight contains
**13,580** quarters. UTC scored bounds are 2025-02-28 23:00 through
2025-10-31 22:45, representing March–October Belgian target days.

Training retained **34,657 finite-label rows** from 34,944 candidates, excluding
the same 287 null labels as the original experiment. All four variants used those
identical rows. The current history-only model reproduced the original cached
test predictions **exactly: maximum absolute difference 0.0 MW**.

### Validation selection

All variants scored the same 5,472 validation quarters; all were available.

| Feature set | Feature count | MAE MW | RMSE MW | Bias MW |
| --- | ---: | ---: | ---: | ---: |
| History | 13 | 291.8954 | 719.6849 | −97.2011 |
| History + radiation | 17 | 233.9942 | 564.9074 | −51.5849 |
| **History + radiation + cloud** | **19** | **231.1953** | **558.9339** | **−43.3400** |
| History + radiation + cloud + temperature | 20 | 231.5123 | 560.9672 | −41.4298 |

The frozen winner is **`history+radiation+cloud`**, model ID
`lightgbm_history_radiation_cloud`. Radiation accounts for most of the validation
gain; cloud adds a smaller additional improvement. Temperature did not improve
this validation score and is excluded from the final model. Its 2.03 MW RMSE
difference is not evidence that temperature is generally unhelpful. All seven
weather features remain in the prepared cache; final inference uses only the six
radiation/cloud columns. Test performance of unselected weather variants was not
used to revisit selection.

Final inference identity:
`6dceac4a33d87ae56c32c1276bbc8dcaa349ebe9e20937704b739e48fb462468`.
Pre-fit source/data/environment identity:
`1b82188ac646feea52a889f29beb902ed676dc70a6fcf84a738237263ddcada7`.

### Primary common-support leaderboard

All models: **23,520 scored quarters, 245/245 origins, 100% forecast availability**.
Skill is relative to exact elapsed 48h persistence on this same population.

| Rank | Model | MAE MW | RMSE MW | Bias MW | RMSE skill |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | **Weather-LightGBM: history + radiation + cloud** | **337.0600** | **624.1890** | **4.0548** | **0.392241** |
| 2 | t0-gamma, unchanged zero-shot | 439.0268 | 866.3308 | 84.3835 | 0.156473 |
| 3 | History-LightGBM | 549.2297 | 1026.8808 | −187.6945 | 0.000149 |
| 4 | 48h seasonal persistence | 511.4227 | 1027.0334 | −0.7162 | 0 |

In the extended five-model table, Chronos-2 ranks third with MAE **466.9755 MW**,
RMSE **938.1344 MW**, bias **41.0236 MW**, and skill **0.086559**. Both TSFM
forecasts are authenticated originals; neither was rerun or changed.

### Daylight leaderboard

All models score the same **13,580** daylight quarters. The RMSE ranking is unchanged.

| Rank | Model | MAE MW | RMSE MW | Bias MW | RMSE skill |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | **Weather-LightGBM** | **573.5884** | **821.3610** | **−3.1623** | **0.392312** |
| 2 | t0-gamma | 756.1760 | 1140.1039 | 146.3947 | 0.156488 |
| 3 | History-LightGBM | 939.2100 | 1351.3355 | −337.0517 | 0.000207 |
| 4 | 48h seasonal persistence | 885.7430 | 1351.6160 | −1.2511 | 0 |

### Information value versus changing the forecasting approach

Changes below are destination minus source: a negative error change is better.

| Comparison | MAE change MW | Relative MAE change | RMSE change MW | Relative RMSE change | Skill change |
| --- | ---: | ---: | ---: | ---: | ---: |
| History-LightGBM → weather-LightGBM | **−212.1698** | **−38.6304%** | **−402.6918** | **−39.2150%** | **+0.392092** |
| History-LightGBM → t0-gamma | −110.2030 | −20.0650% | −160.5500 | −15.6347% | +0.156324 |
| t0-gamma → weather-LightGBM | −101.9668 | −23.2256% | −242.1418 | −27.9503% | +0.235768 |

Within this development experiment, adding forecast radiation/cloud to the fixed
LightGBM procedure improved errors more than switching from history-LightGBM to
the best history-only TSFM. This does not establish a universal model ranking,
operational as-of fidelity or an architecture-only causal result.

### Horizon and local-time diagnostics

| Elapsed issue-to-target bin, hours | History-LightGBM RMSE MW | Weather-LightGBM RMSE MW | t0 RMSE MW | 48h RMSE MW |
| --- | ---: | ---: | ---: | ---: |
| [12,18) | 15.9692 | 13.2365 | 7.9136 | 0 |
| [18,24) | 734.4023 | 540.6583 | 684.7255 | 859.6449 |
| **[24,30)** | **1861.3788** | **1058.1868** | **1534.1731** | **1794.1083** |
| [30,36) | 462.1729 | 382.3633 | 423.6969 | 511.2497 |
| [36,42) | 15.9692 | 13.2365 | 7.4048 | 0 |

The largest absolute gain over history-LightGBM is **803.19 MW** in [24,30),
approximately **43.15%**, corresponding broadly to next-day midday/afternoon at
the fixed issue clock. Lead and local time are confounded here; this does not
isolate a causal lead-time effect. Night-dominated zero-reference-error bins have
undefined RMSE skill, correctly retained as such rather than fabricated percentages.

The largest hourly absolute reduction is at **13:00 Belgian time**:
2112.95 → 1124.48 MW. Gains are not uniform: at 21:00 weather-LightGBM is worse
than history-LightGBM (43.57 versus 27.91 MW), and persistence/t0 remain more
accurate in several low-production hours. Nighttime was never removed.

### Forecast-weather regimes

All-interval diagnostic scores; the artifact also contains daylight versions.
Cloud-group counts are 6,217 low, 7,594 medium and 9,709 high forecast-cloud quarters.

| Forecast regime | History RMSE MW | Weather RMSE MW | t0 RMSE MW | Weather reduction vs history | Weather reduction vs t0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low cloud (<.25) | 1307.55 | 636.67 | 716.23 | **51.31%** | 11.11% |
| Medium cloud | 975.07 | 660.48 | 785.33 | 32.26% | 15.90% |
| High cloud (>=.75) | 846.21 | 585.73 | 1003.63 | 30.78% | **41.64%** |
| Spatially uniform cloud (<.15 std) | 1146.64 | 598.24 | 970.18 | 47.83% | 38.34% |
| Spatially variable cloud (>=.15 std) | 930.68 | 642.31 | 782.66 | 30.99% | 17.93% |

The answer depends on the comparator: the largest gain over history-LightGBM is
under **low forecast cloud**, whereas the largest gain over t0 is under **high
forecast cloud**. t0 is closer to the weather model in the low-cloud group.
These forecast-only groups do not verify actual sky conditions, temporal weather
stability or meteorological causality. Spatial uniformity includes both uniformly
clear and uniformly overcast forecasts and must not be equated with clear weather.

### Daily wins and complementarity

| Model | Sole days won | Share of 245 days |
| --- | ---: | ---: |
| **Weather-LightGBM** | **94** | **38.37%** |
| t0-gamma | 62 | 25.31% |
| 48h seasonal persistence | 50 | 20.41% |
| History-LightGBM | 39 | 15.92% |

There were no tied daily winners. Weather-LightGBM wins the most days but **not a
majority**. Aggregate RMSE leadership does not mean dominance at every origin.

Weather-LightGBM/t0 Pearson signed-error correlation is **0.5428** on all quarters,
**0.5459** during daylight; daily-RMSE correlation is **0.3526**. This indicates
some error complementarity, not demonstrated ensemble skill. No combination,
blend-weight fitting or ensemble test was performed.

### Representative and challenging-day plots

Saved `representative_periods.png` and `difficult_weather_day.png` were visually
reviewed for labels, local times, all four forecast curves and actual Elia PV.
The fixed dates include counterexamples: on **15 June**, persistence wins
(362.29 MW daily RMSE versus weather 735.14); on **15 September**, history-LightGBM
wins (410.48 versus weather 464.70). The weather model wins the fixed 15 March
case (530.24 MW). No date was replaced to make the weather model look better.

The predeclared forecast-cloud heterogeneity proxy selects **1 March 2025**, with
daylight mean spatial cloud-fraction standard deviation **0.37734**. History-LightGBM
also wins that day (224.90 MW versus weather 244.30, persistence 283.99 and t0
454.88). The selected day is therefore not a weather-model victory showcase,
nor proof that actual cloud conditions were correctly forecast.

### Compute and data overhead actually measured

Host: Windows 11, **Intel Core i7-8665U 1.90 GHz, 4 physical/8 logical CPUs**, about
16.98 GB installed RAM, Python 3.12.4. Current packages include LightGBM 4.7.0,
NumPy 2.5.4, pandas 3.1.1, pvlib .16.1, ecCodes 2.50.1, PyArrow 26.0.1,
Shapely 2.2.0, pyproj 3.8.0, Matplotlib 3.11.2 and SciPy 1.18.1.

| Stage/model | Observed time |
| --- | ---: |
| Shared origin-aware training-matrix construction | 25.55 s, once |
| History-LightGBM fit | .424 s |
| Selected 19-feature weather-LightGBM fit | .509 s |
| History-LightGBM predict, 245 origins | 10.105 s total; 41.25 ms/origin |
| Weather-LightGBM predict, 245 origins | 11.808 s total; 48.20 ms/origin |
| History / weather full backtest wall | 15.54 / 26.08 s, including engine/context work |
| Cached original t0 predict | 75.208 s total; 306.97 ms/origin; load 2.576 s |
| Cached original Chronos predict | 63.046 s total; 257.33 ms/origin; load .756 s |

LightGBM uses one CPU thread; original TSFM timings use four Torch threads and
are explicitly reused observations, not reruns under identical instantaneous
machine load. LightGBM's load time is zero here because models remain resident
after fitting; that does not imply zero cold-start or memory cost.

The bounded eight-worker preparation took **3,647.47 seconds (60.79 minutes)**
wall time. Accounted completed bundles contain **39,220,571,109 bytes (39.22 GB)**
of selected-message/index transfers, while planned native caches occupy only
**76.32 MB** and prepared features/status records **8.44 MB**. Initial strict-QC
probes and interrupted partial transfers are additional overhead, so 39.22 GB is
not a claimed exact total of all bytes on the wire during development.

Summed per-origin retrieval/GRIB decode/native-cache work is **24,969.54 worker
seconds**, averaging **37.49 s/run**. Separate spatial/temporal preprocessing is
**4,086.93 worker seconds**, averaging **6.14 s/run**. Summed worker times are not
wall time. Dataset preparation and forecast inference are separate from both.
There are **63,936** prepared target rows across all splits. Precision handling
flagged **144,243 of 3,479,850 native cell-hours (4.145%)**; all source averages
remain stored unchanged. Final PV predictions retain negatives: history 73,
weather 2, t0 5,397 negative quarters; none was clipped.

### Reporting-only repair after frozen inference

All test forecasts completed before a diagnostic pass failed: pandas' target
policy index had second resolution while prepared weather used nanoseconds.
`ForecastRequest` had already normalized the actual inference requests, so model
inputs and forecasts were unaffected. The reporting path now applies that same
nanosecond normalization. Diagnostics were regenerated with `--evaluate-only`.

The original inference freeze/source ZIP/model files remain unchanged. A separate
`analysis_amendment.json`, exact diff and `analysis_source_snapshot.zip` authenticate
the reporting-script revision and **21 protected forecast/model/provenance files**.
The guard permits no changed model, feature, data or configuration source. No refit,
new test inference, feature reselection or forecast alteration occurred. This is
not a rewritten freeze pretending the original and repaired scripts were identical.

### Interpretation and next priority

The evidence supports the narrower result that forecast radiation/cloud adds
substantial predictive value to this fixed LightGBM procedure on the shared 2025
development population, beyond the gain from the tested history-only TSFM switch.
The weather model also retains meaningful failures, loses most individual days,
and remains limited by both NWP and PV-mapping errors. Known-holdout reuse,
unverified publication/as-of fidelity, winter-only validation, fleet nonstationarity
and unknown TSFM pretraining overlap prevent a strong independent operational or
universal-best-model claim.

The next priority is **another scientifically justified experiment: a fresh,
predeclared confirmatory period with the selected feature recipe frozen and
availability/vintage evidence audited**. After confirmation, probabilistic forecasts
are a sensible next modeling step. The moderate t0/weather error correlation makes
an ensemble worth a later controlled study, but correlation alone does not justify
claiming its benefit. No ensemble result is claimed.

### Scientific acceptance checklist

| Requirement | Status/evidence |
| --- | --- |
| Real Elia PV | Pass: original hash-verified national snapshot |
| Historical NWP forecasts | Pass: operational GFS GRIB, explicit reference/steps |
| No future observed weather | Pass: no observations or reanalysis fetched/used |
| Reference and valid times preserved | Pass: native schema and source manifests |
| NWP available by issue | Pass **under declared init+6h assumption**; observed publication/as-of fidelity remains unverified |
| Training weather vintage-correct | Pass under the same policy, per historical issue, audited |
| Weather canaries pass | Pass: selector/backtester, training matrix and full prediction path |
| Test period frozen before final evaluation | Pass; **development**, not untouched confirmation; pretraining amendments preserved |
| Same common forecast origins | Pass: all 245, 23,520 pairs; independent availability also reported |
| History-LightGBM not retuned | Pass: original parameters/rows; predictions exactly reproduced |
| TSFM configuration unchanged | Pass: pinned checkpoints/context/schedule and authenticated cached predictions |
| Weather selection uses validation | Pass: four predefined variants; selected 19-feature model before test inference |
| Missing NWP not silently substituted | Pass: whole-origin failure policy; zero actual missing origins |
| Negative predictions not clipped | Pass: negative forecasts retained in raw/scored outputs |
| Nighttime retained | Pass: all 23,520 primary quarters; separate daylight table |
| DST handled | Pass: unit/native model tests and actual 92/100-target origins |
| Compute environment recorded | Pass: pre-fit metadata, package listing and source archives |

## Software verification before the real-data benchmark

the historical GFS ingestion layer's full 68-test suite passed before extending the model. Subsequent
checks include the new precision-policy, training-vintage and full-prediction
canaries, historical/deterministic feature identity, missing-run/time/variable
accounting, 92/96/100-target predictions, preserved history-only predictions,
authenticated cached schedules/checkpoints, frozen-plan drift rejection and an
offline common-support diagnostic workflow. No normal test fetches provider data
or model weights.

| Command from repository root | Result before benchmark |
| --- | --- |
| `.\.venv\Scripts\python.exe -m pytest -q` | 430 passed, 9 skipped (native TSFM opt-ins and absent optional weather libraries) |
| `.\.venv\weather-smoke\Scripts\python.exe -m pytest -q` | 437 passed, 2 native TSFM opt-ins skipped |
| `.\.venv\weather-smoke\Scripts\python.exe -m pytest tests/test_weather.py tests/test_weather_gfs.py tests/test_weather_precision.py tests/test_lightgbm_weather.py tests/test_weather_experiment.py tests/test_weather_benchmark.py -q` | 86 passed |
| `.\.venv\weather-smoke\Scripts\python.exe -m pytest tests/test_backtesting.py tests/test_lightgbm_data.py tests/test_lightgbm_weather.py tests/test_weather.py -k 'canary or leakage' -q` | 5 passed, 76 deselected |
| `.\.venv\Scripts\python.exe -m pytest tests/test_persistence.py tests/test_lightgbm.py tests/test_chronos.py tests/test_t0.py -q` | 108 passed |
| `.\.venv\Scripts\python.exe -m ruff check .` | Passed |
| `.\.venv\Scripts\python.exe -m ruff format --check .` | 93 files formatted, including documented Python code blocks |
| `.\.venv\Scripts\python.exe -m mypy` | Passed, 57 source/test files |
| `.\.venv\Scripts\python.exe -m mypy scripts/benchmark_weather_lightgbm.py` | Passed, experiment runner checked explicitly |
| `.\.venv\Scripts\python.exe -m build` | Wheel and source distribution built |

The original 47 recorded TSFM artifacts and 34 original LightGBM artifacts are
preserved. Their recorded source identities intentionally differ from the current
extended implementation; old manifests are not rewritten.

After the reporting-only repair, the same full commands passed again: **431 passed,
9 skipped** without weather binary libraries; **438 passed, 2 skipped** in the
weather/LightGBM environment. The added test authenticates reporting amendments
and rejects changed model/configuration sources; the diagnostic regression also
asserts nanosecond target normalization. Ruff, formatting and both mypy commands
pass. Both exported figures were visually reviewed. The successful final diagnostic
command was:

```powershell
.\.venv\weather-smoke\Scripts\python.exe scripts/benchmark_weather_lightgbm.py --config configs/experiments/weather_lightgbm_v1.toml --evaluate-only
```

## Reproducibility resources

The executable experiment is `scripts/benchmark_weather_lightgbm.py`, configured
by `configs/experiments/weather_lightgbm_v1.toml`. Prepared covariates, fitted models,
source archives, diagnostics and figures remain outside Git. Original baseline/TSFM
configurations, PV snapshots and prediction artifacts were preserved; old freezes
are not rewritten to match the current public source tree. The original run has
no committed source revision, so authenticated source archives retain its exact
implementation identity. See the [reproducibility guide](../reproducibility.md).
