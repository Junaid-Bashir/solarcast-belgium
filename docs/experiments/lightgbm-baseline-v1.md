# First real Elia PV benchmark: LightGBM baseline v1

## Research question and scope

Does the initial pooled LightGBM improve next-day Belgian national PV forecasts
over historical persistence using only PV history and deterministic time/solar
information? This is a retrospective experiment under an approximate availability
policy. It is not a comparison with Elia's forecasts or an operational deployment.

The configuration was fixed before inspecting any real test performance. There
was one model fit, no hyperparameter search, no feature selection from test scores,
no early stopping, and no retraining. The separately reserved validation period
was unused. Synthetic example results elsewhere in this repository are software
checks and are unrelated to these real-data scores.

## Data, target and chronological periods

Attribution: **Elia Transmission Belgium SA**. National `ods032.measured`, mapped
by the existing client to `pv_generation_mw`: an estimated/upscaled interval-average
PV power in MW, indexed by UTC interval **start**. The national Belgium selection
is used directly, without summing provinces. No Elia forecast fields are fetched
or used as features. Capacity is retained as provenance, never a model input.

| Setting | Fixed value |
|---|---|
| Data | 18 December 2023 00:00 Brussels inclusive to 1 November 2025 00:00 Brussels exclusive |
| UTC data bounds | `[2023-12-17T23:00Z, 2025-10-31T23:00Z)` |
| Training origin dates | 1 January–29 December 2024 inclusive |
| Training target dates | 2 January–30 December 2024 inclusive |
| Fit boundary | 1 January 2025 08:00 UTC; labels eligible through cutoff 07:30 UTC |
| Reserved validation origins | 1 January–26 February 2025 inclusive; unused |
| Reserved validation targets | 2 January–27 February 2025 inclusive |
| Test origin dates | 28 February–30 October 2025 inclusive: 245 origins |
| Test target dates | 1 March–31 October 2025 inclusive |
| Issue time | Daily **11:00 Europe/Brussels**, canonicalized to UTC |
| Target policy | Existing next local calendar day; 92/96/100 quarters |
| Context | Eligible observation starts within 14 elapsed days before issue |
| Observation safety lag | **30 elapsed minutes**, with quarter-hour flooring |
| Primary reference | Exact elapsed **48-hour seasonal persistence** |
| Primary ranking | Lower RMSE, all intervals including night |

A nearly full calendar year provides training seasonality; eight test months cover
spring, summer and autumn and both 2025 DST changes. This is a calendar split,
not a percentage split. Gaps between boundaries avoid immature training labels
and validation targets overlapping test issues. The experiment stays entirely in
`ods032`: it does not cross or establish continuity at the October 2026 schema seam.
It does not test winter performance or the new `ods209` schema.

## Availability and leakage audit

For issue T, cutoff is `floor_15min(T - 30min)`. Only observations whose complete
interval ends by cutoff enter context. At 11:00 local, the latest possible interval
start is 10:15 local. Thirty minutes is a disclosed experimental assumption, not
a verified Elia publication latency. This run uses the values retrieved in 2026;
historical revisions and issue-time vintages cannot be reconstructed. A daily
historical product also does not prove these same values were available intraday.

Training rebuilds each origin's historical features through the same context
selection as backtesting and joins fit-time-eligible labels separately. Training
target grids must finish by the fit cutoff. Actuals enter backtest results only
after inference. Later completed test observations may enter subsequent inference
contexts; fixed fitted weights never see test labels. Deterministic future time
and solar geometry are permitted. Lags use only eligible exact historical matches;
rolling statistics stop at the cutoff, require all 96 elapsed quarters, and never
extend into a forecast day. Existing future-value canaries and DST tests pass.

## Features and model configuration

The unchanged initial recipe contains exactly 13 float64 features:

```text
time_of_day_sin, time_of_day_cos, day_of_year_sin, day_of_year_cos,
solar_apparent_elevation_deg, solar_azimuth_deg, clearsky_ghi_w_m2,
horizon_hours, pv_lag_24h_mw, pv_lag_48h_mw, pv_lag_7d_mw,
pv_trailing_24h_mean_mw, pv_trailing_24h_max_mw
```

Solar geometry uses the documented Brussels reference (50.85 N, 4.35 E, altitude
0 m), fixed refraction assumptions, and Haurwitz clear-sky GHI. These are not
weather measurements or forecasts. No capacity normalization is performed.

The fixed LightGBM defaults are 100 trees, learning rate 0.05, 15 leaves,
min_child_samples 20, seed 0, squared-error regression, one CPU thread,
deterministic mode and no sampling. Missing features stay NaN and are handled
natively (`use_missing=True`, `zero_as_missing=False`). Predictions are raw:
no clipping, nighttime masking or bias correction.

## Data-quality result

The 23 monthly/edge shards were retrieved on **14 November 2026** through the
existing Elia client and then reused for the benchmark. The complete selected
grid has **65,664 rows and expected intervals**, **0 absent timestamp rows**,
**302 explicit null PV observations**, **0 duplicate timestamps**, **0 off-grid
timestamps**, and **0 negative PV observations**. All rows are `ods032.measured`.
The first observation is 17 December 2023 23:00 UTC; the last is 31 October 2025
22:45 UTC. Bounds are half-open as specified above.

Nulls occur in early history through October 2024, including a recurring null
in the final local quarter-hour on many days. No physical cause is asserted.
There are no null PV observations in the selected 2025 test period. The original
snapshot retains all nulls. Missing training labels are excluded and counted;
historical lag/rolling features remain missing where required. Baselines retain
their whole-origin failure policy; missing test actuals would be excluded from
common scoring rather than filled. No data repairs were applied.

| Training dataset diagnostic | Count |
|---|---:|
| Candidate issue/target rows | 34,944 |
| Excluded missing target labels | 287 |
| Final fitted rows | 34,657 |
| Features | 13 |
| Fitted rows with at least one missing feature | 31,507 |
| Missing feature cells | 74,298 |
| Missing 24h-lag cells | 19,370 |
| Missing 48h-lag cells | 5 |
| Missing 7d-lag cells | 15 |
| Missing trailing-mean cells | 27,454 |
| Missing trailing-max cells | 27,454 |

The 24h lag is often unavailable by design at this issue time. Early-history nulls
also make the strict complete-window rolling features unavailable for many training
rows. All future-known features are complete. These missingness patterns differ
from the complete 2025 test observations and limit interpretation of this initial
recipe. They were disclosed without changing features or filling values.

## Baselines, scoring and diagnostics

All models receive exactly the same scheduled origins and availability/context
policy. Exact UTC seasonal persistence is run at 24h, 48h and 7d; last-value
persistence is included as a sanity check. Any missing seasonal match fails the
whole origin under the existing finite Forecast contract; there is no fallback.

**24h is ineligible for complete next-day forecasts at 11:00**: today's afternoon
has not happened. Its failed origins and zero forecast availability are saved and
reported separately, without changing the baseline or supplying future PV. The
predeclared main comparison contains 48h, 7d, last-value and LightGBM. Adding the
all-failed 24h model would make the intersection empty for every model. The main
comparison uses identical common finite issue/target pairs, including any support
lost to missing data in an otherwise eligible baseline. Failures remain visible
in the all-model availability table.

MAE, RMSE and signed bias (prediction minus actual), in MW, reuse the existing
evaluation framework. RMSE skill is `1 - model_RMSE / reference_RMSE` on the same
support. All quarters receive equal weight, so DST days have unequal day weights.
All-interval results remain primary. Daylight is separately selected by apparent
elevation > 0 degrees at interval start at the representative point; this is an
approximation to national daylight, not a rule for modifying observations.

Horizon groups are **elapsed six-hour bins**, lower bound inclusive, upper bound
exclusive: e.g. 12 means [12h,18h). Exact-quarter-hour horizon tables are also
saved. Belgian local-hour, calendar-month and meteorological-season tables use
the same globally aligned support. SON contains only September–October here.
Daily MAE/RMSE tables identify best/worst days diagnostically. Representative
plots were preselected as 15 March, 15 June and 15 September, before scores.
The separate worst-day plot uses the three largest LightGBM daily RMSEs.

## Reproduction and local artifacts

Run from the repository root with Python 3.11–3.13:

```powershell
python -m pip install -e ".[dev,experiments]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m mypy scripts/benchmark_lightgbm.py
python -m pytest tests/test_backtesting.py tests/test_features.py tests/test_lightgbm_data.py tests/test_lightgbm.py tests/test_persistence.py -k "canary or future or cutoff or rolling" -q
python scripts/benchmark_lightgbm.py --config configs/experiments/lightgbm_v1.toml --validate-only
python scripts/benchmark_lightgbm.py --config configs/experiments/lightgbm_v1.toml
```

`--download-only` can obtain/audit the same snapshot before fitting. The existing
Elia client performs bounded one-day requests. Completed monthly normalized CSV
shards and metadata are reused under ignored `data/elia_benchmark_v1/`; checksums,
UTC offsets, numeric values, source mapping and structural quality are validated
on reload. Gaps/nulls are reported without repair. A failed download raises and
preserves completed shards. Corrupt/incomplete shard pairs fail rather than being
silently overwritten. There is no fallback to synthetic data.

Ignored `artifacts/experiments/lightgbm_baseline_v1/` contains copied TOML and hash,
snapshot manifest, quality and training summaries, training audit timestamps,
per-model predictions and origin failures, all-model availability, primary and
daylight leaderboards, complete metrics, aligned predictions, grouped/daily
diagnostics, selected best/worst cases, figures, environment freeze and metadata.
The metadata records package versions, source hashes, config/snapshot hashes,
retrieval times, seed, feature list, Git status and commit if one exists. No model
serialization is introduced. Output directories are protected from overwrite.
To repeat fitting, preserve/rename an existing output directory first; unchanged
data snapshots will still be reused. Package/platform changes or upstream data
revisions can change scores. Snapshot hashes identify this particular run.

Rebuild plots from saved aligned predictions without downloading/refitting:

```powershell
python scripts/benchmark_lightgbm.py --config configs/experiments/lightgbm_v1.toml --plots-only
```

The code license does not license the data. The linked [Elia licence page](https://opendata.elia.be/pages/licence/)
still returned a service-error banner on 14 November 2026; full terms remain
pending live confirmation as documented in the data contract. Raw snapshots and
large generated tables are local and ignored, not redistributed automatically.

## Overall results: no meaningful improvement established

Run completed **14 November 2026, 15:55:26 UTC**, using **LightGBM 4.7.0**.
Fitting, forecasting and diagnostics from the cached snapshot took 156.672 seconds.
All four eligible models returned every forecast: **245/245 origins**, **23,520
common scored observations**, **100% forecast availability**, no missing test
actuals and no alignment exclusions. Target days include 243 normal days, one
92-interval spring day, and one 100-interval autumn day. Exact 24h persistence
failed all 245 origins for unavailable matches; its availability is 0%, scores
undefined, and it is explicitly unranked.

### Primary leaderboard: all intervals

| Rank | Model | MAE MW | RMSE MW | Bias MW | RMSE skill vs 48h | Scored |
|---:|---|---:|---:|---:|---:|---:|
| 1 | LightGBM | 549.229740 | 1026.880784 | -187.694482 | 0.000148598 | 23,520 |
| 2 | Seasonal persistence 48h | 511.422685 | 1027.033399 | -0.716234 | 0 | 23,520 |
| 3 | Seasonal persistence 7d | 631.100365 | 1220.885459 | 3.905361 | -0.188749518 | 23,520 |
| 4 | Last-value persistence | 2407.383329 | 2816.612745 | 1659.875721 | -1.742474342 | 23,520 |

LightGBM ranks first under the predeclared RMSE rule, but improves RMSE by just
**0.152615 MW**, or **0.014860%**. Its MAE is **7.3925% worse** than 48h persistence,
and it underpredicts by 187.694 MW on average. These results support **similar
RMSE performance**, not a claim of material ML superiority. No significance or
uncertainty analysis was performed. Ranking by MAE would favor 48h persistence;
the primary ranking was not changed after inspecting this outcome.

### Daylight-only leaderboard

| Rank | Model | MAE MW | RMSE MW | Bias MW | RMSE skill vs 48h | Scored |
|---:|---|---:|---:|---:|---:|---:|
| 1 | LightGBM | 939.209998 | 1351.335527 | -337.051708 | 0.000207494 | 13,580 |
| 2 | Seasonal persistence 48h | 885.742982 | 1351.615979 | -1.251126 | 0 | 13,580 |
| 3 | Seasonal persistence 7d | 1092.863123 | 1606.730027 | 6.595669 | -0.188747434 | 13,580 |
| 4 | Last-value persistence | 1913.695607 | 2318.315339 | 619.043255 | -0.715217469 | 13,580 |

The RMSE ranking is unchanged and the advantage remains negligible (0.020749%).
Daylight errors are much larger than all-interval errors. Nighttime remains in
the primary table; it was not removed to improve the presentation.

## Horizon and Belgian local-time diagnostics

| Elapsed horizon bin | Common n per model | 48h RMSE MW | LightGBM RMSE MW | LightGBM skill |
|---|---:|---:|---:|---:|
| [12h,18h) | 4,900 | 0.000 | 15.969 | Undefined: zero reference RMSE |
| [18h,24h) | 5,880 | 859.645 | 734.402 | 0.145691 |
| [24h,30h) | 5,880 | 1794.108 | 1861.379 | -0.037495 |
| [30h,36h) | 5,880 | 511.250 | 462.173 | 0.095994 |
| [36h,42h) | 980 | 0.000 | 15.969 | Undefined: zero reference RMSE |

The largest absolute RMSE gain is 125.243 MW in [18h,24h); the largest loss is
67.270 MW in [24h,30h), covering much of midday. The overnight bins expose small
raw positive LightGBM predictions where the reference exactly matches zero PV.
Only the occupied parts of edge bins contribute observations. Horizon and time
of day are coupled at a single daily issue clock: this is not evidence of a
separate causal horizon effect.

The complete `by_local_hour.csv` shows 980 common quarters per clock-hour group:

| Brussels target hour | 48h RMSE MW | LightGBM RMSE MW | LightGBM skill |
|---:|---:|---:|---:|
| 08 | 685.494 | 578.619 | 0.155909 |
| 09 | 1164.245 | 922.903 | 0.207295 |
| 12 | 1922.902 | 2065.624 | -0.074222 |
| 13 | 1993.992 | 2112.953 | -0.059660 |
| 17 | 1028.594 | 902.574 | 0.122516 |
| 19 | 328.070 | 399.788 | -0.218606 |

Morning and late-afternoon gains coexist with noon underprediction and weaker
evening performance. Hours 00–04 and 23 have reference RMSE zero and LightGBM
RMSE 15.969 MW; ratio skill is undefined there. Small reference errors can also
make skill ratios unstable, so MW errors should be read alongside them.

## Monthly and seasonal diagnostics

| Target month | Common n | 48h RMSE MW | LightGBM RMSE MW | LightGBM skill |
|---|---:|---:|---:|---:|
| March 2025 | 2,972 | 954.156 | 1182.310 | -0.239117 |
| April 2025 | 2,880 | 1229.376 | 1457.436 | -0.185509 |
| May 2025 | 2,976 | 1133.787 | 1001.473 | 0.116701 |
| June 2025 | 2,880 | 984.022 | 1030.788 | -0.047525 |
| July 2025 | 2,976 | 984.646 | 920.423 | 0.065224 |
| August 2025 | 2,976 | 968.997 | 748.290 | 0.227769 |
| September 2025 | 2,880 | 1169.124 | 978.573 | 0.162986 |
| October 2025 | 2,980 | 714.168 | 718.697 | -0.006341 |

| Meteorological season in test | Common n | 48h RMSE MW | LightGBM RMSE MW | LightGBM skill |
|---|---:|---:|---:|---:|
| MAM | 8,828 | 1110.361 | 1225.423 | -0.103626 |
| JJA | 8,832 | 979.197 | 905.876 | 0.074878 |
| SON: September–October only | 5,860 | 964.952 | 856.330 | 0.112567 |

Performance varies materially: losses in March/April outweigh much of the gain
in August/September. This partial-year result cannot establish consistency over
all seasons or years. These are descriptive test diagnostics, not tuning inputs.

## Daily extremes and failure-case inspection

Daily extremes below use all intervals on the same common support. They are
diagnostic examples, not the main benchmark result.

| Selection | Target local date | n | LightGBM MAE MW | LightGBM RMSE MW | 48h RMSE MW |
|---|---|---:|---:|---:|---:|
| Best | 31 October 2025 | 96 | 68.125 | 126.524 | 309.510 |
| Best | 26 October 2025 | 100 | 87.355 | 148.272 | 541.267 |
| Best | 20 October 2025 | 96 | 77.767 | 149.285 | 277.057 |
| Worst | 2 March 2025 | 96 | 1235.808 | 2123.303 | 1623.423 |
| Worst | 1 April 2025 | 96 | 1218.362 | 2007.137 | 1598.018 |
| Worst | 2 April 2025 | 96 | 1191.229 | 1996.528 | 649.367 |

The three worst days have high, relatively smooth actual daytime profiles and
large LightGBM underprediction concentrated around midday. Mean absolute error
between 10:00 and 16:00 local is 4012.385, 3684.789 and 3677.227 MW respectively;
the corresponding bias is negative throughout those midday windows. Peak absolute
errors occur at 12:30, 13:30 and 13:00 local. Neither target PV nor required
48h/7d profiles have missing data on these cases.

On 2 March and 1 April, 48h persistence also substantially underpredicts. On
2 April it is much closer to actual than LightGBM, while 7d persistence is worse
(2336.231 MW daily RMSE). A blanket explanation that the errors were impossible
to anticipate from PV history is not supported. The model's damped peaks deserve
later controlled investigation. Capacity growth, training missingness, statistical
shrinkage or changing atmospheric conditions are potential hypotheses, not causal
findings. No weather data was consulted, so no cloud event is asserted.

## Figures

The first three days were fixed in the configuration before scoring. They include
both strong persistence behavior and model underprediction, rather than selecting
only LightGBM successes. All plots use saved common-aligned predictions and local
Belgian time; forecasts are raw MW values.

Local-only research figure: **Preselected March, June and September profiles**, `artifacts/experiments/lightgbm_baseline_v1/actual_vs_forecast.png`.
This original generated artifact is not distributed with the public source.

Local-only research figure: **Three largest LightGBM daily RMSE cases**, `artifacts/experiments/lightgbm_baseline_v1/worst_days.png`.
This original generated artifact is not distributed with the public source.

Local-only research figure: **Absolute-error empirical cumulative distributions**, `artifacts/experiments/lightgbm_baseline_v1/absolute_error_ecdf.png`.
This original generated artifact is not distributed with the public source.

These images are local ignored artifacts; they render after running the experiment
or retaining its output directory. They were visually inspected for readable
axes, legends and profile alignment. The ECDF includes nighttime zeros; it should
be read alongside the separately reported daylight metrics.

## Verification and scientific acceptance

The commands in the reproduction section were executed using
`.\.venv\Scripts\python.exe`. Before the real benchmark:

- Full pytest: **274 passed**, including 15 new offline experiment-utility tests.
- Ruff check and format check: **passed**.
- Mypy: **32 source/test files passed**; explicit script check: **1 file passed**.
- Leakage selection shown above: **16 passed, 105 deselected**.
- `--validate-only`: **passed**, no download or scoring.
- `python -m pip check`: **no broken requirements**.
- `python -m build`: **sdist and wheel built**; experiment script/config included
  in the sdist. The live benchmark is not part of CI.
- `--download-only`: **succeeded**, real data audit reported before fitting.
- Main benchmark: **exit 0**, one fit and complete recorded outputs.
- `--plots-only`: **exit 0**; all three figures regenerated without fitting or
  downloading. All recorded artifact hashes still matched. Recorded source/config
  hashes also matched the workspace after reporting: no model/experiment-code
  changes were made after test inspection. Final Ruff checks passed. The run
  produced **35 artifact files** and used **23 cached CSV/sidecar pairs**, all ignored.

| Scientific acceptance item | Result |
|---|---|
| Real Elia data used | Confirmed; client retrieval and hashed snapshot |
| Dataset IDs recorded | Confirmed: ods032, measured, national Belgium |
| Chronological split | Confirmed: 2024 train, reserved early-2025 validation, later-2025 test |
| No test target labels in training | Confirmed: label cutoff precedes every test origin |
| Identical origins/common-support alignment | Confirmed: 245 origins; 23,520 quarters per ranked model; 24h failure disclosed |
| Leakage tests pass | Confirmed: 16 selected checks and full suite |
| No weather observations used | Confirmed |
| No Elia forecast values used as features | Confirmed; not fetched |
| No test-set hyperparameter tuning | Confirmed; fixed defaults and feature recipe |
| Missing values not silently zero-filled | Confirmed: label exclusions and NaN feature counts recorded |
| DST handling valid | Confirmed: 92/96/100 target counts and existing DST tests |
| Nighttime/daylight distinguishable | Confirmed: separately labelled leaderboards |
| Reproducible configuration | Confirmed: TOML, hashes, snapshot manifest, environment and script |

SolarCast version is 0.1.0. The run used Python 3.12.4, pandas 3.1.1, NumPy 2.5.4,
pvlib 0.16.1, tzdata 2026.6, Matplotlib 3.11.2 and HTTPX 0.28.1. The repository
had no commit/HEAD; all project files were untracked, so source hashes and Git
status were recorded instead. No commit or push was performed.

Configuration SHA-256:
`b35880938d8bee5945e48e4e5edfb1e6f7cec1795bc44d7094b668d99d68aafb`.
Snapshot manifest SHA-256:
`98e0cd1f701a60a51f405d1bc6ff415952c9513c28a27fc38b75dca6cc3156e7`.

## Assumptions and limitations

The explicit assumptions are retrospective national MW estimates as the target,
30-minute event-time safety lag, fixed Brussels representative geometry, a single
annual training period, and fixed weights throughout test. They do not establish
original publication timing, immutable data vintages, national spatial irradiance,
or causal explanations of errors. Many training rolling features are missing;
test missingness differs. Fleet changes are not capacity-normalized. Exact UTC
lags can shift civil-clock matching around DST. Raw nighttime offsets remain.

This experiment cannot establish statistical superiority, operational as-of
performance, winter reliability, multi-year generalization, `ods209` continuity,
performance with weather, or superiority over Elia. It does not test how much
further a better-designed or tuned classical model could improve. No test-driven
changes were made to features, preprocessing, training windows or hyperparameters.
The holdout now has published diagnostics: future modeling decisions informed
by it require a fresh holdout or an explicit development-set designation.

**Conclusion:** initial LightGBM and 48h persistence have essentially tied RMSE;
48h persistence has better MAE and much less bias. A trustworthy baseline is now
available without requiring ML to be the winner.
