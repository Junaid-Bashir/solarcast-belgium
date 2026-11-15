# First controlled history-only zero-shot TSFM benchmark

## Executive summary

t0-gamma has the lowest common-support test RMSE, **866.33 MW**, against
938.13 for Chronos-2, 1026.88 for LightGBM and 1027.03 for 48h persistence.
Both zero-shot TSFMs improve on this fixed LightGBM setup: 15.63% and 8.64%
lower RMSE respectively. The ranking is unchanged on daylight-only intervals.
Every model successfully forecasts all 245 origins and all 23,520 targets.
LightGBM nevertheless wins the most individual days, and different models win
different months. t0-gamma's mean predict call takes 0.307 seconds on the tested
CPU, compared with 0.035 for LightGBM, excluding loading and local training.
These observations show useful pretrained-model performance in this fixed
retrospective setup; they do not establish universal superiority, operational
as-of accuracy or economic value. The reused holdout and unknown pretraining
overlap require particular caution.

## Question, hypothesis and scope

Can pretrained zero-shot TSFMs improve day-ahead Belgian national PV forecasts
over the frozen locally trained LightGBM and primary 48h seasonal baseline,
without weather information? The hypothesis is that pretrained temporal patterns
may help; no family is assumed to win. Accuracy and incremental compute cost are
separate questions. This is a retrospective four-model comparison, not an
operational forecast service or a comparison against Elia's own forecasts.

**Holdout disclosure:** the history-only LightGBM experiment LightGBM/baseline scores and diagnostic
plots, and the Chronos/t0 integration June technical smoke outputs, were already known.
This is the mandated reuse of that original period, not a new untouched holdout.
The TSFM checkpoints/context/point rules were selected in earlier milestones;
none of the four model implementations or parameters was changed for this run.
No new configuration was selected using these final test errors. Subsequent
decisions informed by this report require a fresh confirmatory period or an
explicitly labelled development experiment.

## Frozen configuration and data

`configs/experiments/tsfm_zero_shot_v1.toml` pins and inherits the original
`lightgbm_v1.toml` by SHA-256 and expands it in the run's `freeze.json` before
inference. The configuration was saved before timing/final evaluation. Its hash is
`91ea9258adb09cf799c38028548e749454e12d019d94804e0a1f51212cdf40b4`.
The original configuration hash remains
`b35880938d8bee5945e48e4e5edfb1e6f7cec1795bc44d7094b668d99d68aafb`;
the original snapshot manifest hash remains
`98e0cd1f701a60a51f405d1bc6ff415952c9513c28a27fc38b75dca6cc3156e7`.

Attribution: **Elia Transmission Belgium SA**. The 23 previously retrieved
normalized shards contain national `ods032.measured` interval-average/upscaled PV
estimates, in MW at UTC interval starts. They are not a sum of provincial rows.
CSV and sidecar hashes are verified against the original manifest before loading;
missing/corrupt shards fail rather than being refreshed. No new Elia retrieval,
synthetic fallback, filling or source remapping occurs in this experiment.

| Setting | Frozen value |
|---|---|
| Data bounds | `[2023-12-17T23:00Z, 2025-10-31T23:00Z)` |
| Training origin dates | 1 January–29 December 2024 inclusive |
| Training target dates | 2 January–30 December 2024 inclusive |
| Fit time / eligible label cutoff | 1 January 2025 08:00Z / 07:30Z |
| Reserved validation origins | 1 January–26 February 2025 inclusive |
| Timing-only sample | 15 and 16 January 2025; no accuracy/context selection |
| Test origin dates | 28 February–30 October 2025 inclusive: 245 origins |
| Test target dates | 1 March–31 October 2025 inclusive |
| Issue | 11:00 Europe/Brussels, converted to UTC |
| Availability approximation | Completed intervals by floor(issue minus 30 minutes) |
| Targets | Next Brussels calendar day, 92/96/100 quarter-hours |
| Context | Issue-anchored 14 elapsed days, selected identically for all models |
| Primary population / rank | All intervals including night; lower common-support RMSE |
| Reference | Exact elapsed 48h seasonal persistence |
| Additional population | Existing representative-point daylight rule |

The snapshot has 65,664 quarter-hour rows, zero absent/duplicate/misaligned rows,
302 explicit null PV values and zero negative observations. Nulls are in early
history; test targets are complete. No continuity across the October 2026 schema
change is established. Fleet-capacity evolution and historical value revisions
remain unresolved.

## Models and information sets

Exactly four models are compared:

| Model ID | Frozen implementation / information |
|---|---|
| `seasonal_persistence_48h` | Exact eligible PV at target minus 48 elapsed hours; no fallback |
| `lightgbm` | Original 100 trees, 0.05 learning rate, 15 leaves, min-child 20, seed 0, one CPU thread; one 2024 fit |
| `chronos` | `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, library 2.3.2 |
| `t0` | `theforecastingcompany/t0-gamma`, revision `fef67abad3eed4f227c7b536a6e8ee4e9ab1080d`, library 0.6.1 |

Both TSFMs use the already approved 1344-slot maximum (14 days). The common
issue-anchored window with safety lag supplies 1342 quarter-hours per test origin.
The cap counts elapsed slots, preserving NaNs instead of compressing time. There
is no context search, model-native context alternative, weather, monitored
capacity, Elia forecast, local fine-tuning or checkpoint selection in this run.
Both use native scaling, float32 CPU inference and native P50, with gamma's optional
adjustments explicitly off. Raw negative predictions remain unmodified.

**Zero-shot** means provider-pretrained weights with no local Belgian PV training,
fine-tuning, gradient update, test-period weight adaptation or checkpoint choice
from final test errors. It does **not** certify that provider pretraining excluded
Elia or related data; pretraining overlap remains unknown.

LightGBM retains all 13 original features: deterministic calendar cycles,
Brussels representative solar geometry/Haurwitz clear-sky GHI, horizon, exact
24h/48h/7D PV lags, and cutoff-anchored trailing-24h mean/max. Solar geometry is
not observed weather. TSFMs receive historical PV alone; engineered deterministic
future inputs are not transferred to them. This compares fixed forecasting
approaches, with disclosed feature representations, rather than identical
internal inputs or optimized mean-versus-median prediction objectives.

LightGBM uses 34,657 finite-label training rows from 34,944 candidates; 287 null
labels are excluded. Its original missing feature patterns are preserved, including
74,298 missing feature cells. Training labels stop before the fit-time cutoff;
weights remain fixed across validation and test. Later completed test PV may
enter later rolling inference histories, exactly as for the other models.

## Evaluation and diagnostic rules

The unchanged backtester supplies only eligible `ForecastContext` history. Actuals
are joined after inference. At issue 11:00/cutoff 10:30 local, both TSFMs forecast
from cutoff through the final next-day target, then select exact requested UTC
timestamps: 146/150/154 complete rollout steps for 92/96/100 targets. DST days are
retained. No padding, clipped outputs or nighttime masking is introduced.

The primary table uses the intersection of finite forecast/actual issue-target
pairs across **all four models**. Independent availability and failed origins are
reported separately; missing forecasts are never replaced. All intervals have
equal weight, so 23/25-hour days receive 92/100 rather than 96 weights.
MAE/RMSE/bias reuse existing MW definitions; bias is prediction minus actual.
RMSE skill is `1 - model RMSE / 48h-reference RMSE` on the same population.
The daylight table separately selects apparent elevation >0 at interval start
at the existing Brussels reference point; it does not replace the primary table.

Horizon bins retain the existing elapsed six-hour convention: [12,18), [18,24),
[24,30), [30,36), [36,42) hours where populated. Exact horizons and local Belgian
hours are saved too. Month and meteorological-season groups retain common
support/sample counts; SON contains September–October only, and winter is absent.

Daily MAE/RMSE is calculated on each target day's common pairs. Days receive equal
weight in win counts. Models within absolute **1e-9 MW RMSE** of the daily minimum
share equal fractional win credit; sole-win counts are also saved. This is a
tie convention, not a statistical equivalence threshold.

Complementarity uses Pearson correlations of signed quarter-hour errors on all
common pairs and separately daylight, plus correlations of daily RMSE. These
diagnostics do not fit an ensemble or demonstrate gains from one. Predeclared
normal plot dates are 15 March, 15 June and 15 September. The difficult-day plot
maximizes mean daily RMSE across models. The all-struggle case maximizes minimum
model daily RMSE. Each strongest sole-win example maximizes second-best minus
winning RMSE, breaking equal diagnostic values by chronological order. These
post-score selections are qualitative diagnostics, not evidence for overall wins
or explanations of weather conditions.

## Runtime, caching and reproducibility

The two predeclared validation timing origins estimated 133.5 seconds for all
245-origin predict calls plus 11.4 seconds of training preparation, loading and
engine overhead. No accuracy was inspected on the timing sample. All 245 origins
were scheduled; the test period was not reduced for speed.

Each model's predictions/origins/call timings are saved immediately after its
backtest. A canonical cache identity covers the full configuration, expanded
parent, data manifest, checkpoints/context, source hashes, runtime versions,
hardware and stage. Timing caches cannot be mistaken for test caches. Per-file
hashes and the exact scheduled test population are checked before evaluation.
`--evaluate-only` rebuilds diagnostics/figures from saved predictions without model
loading or inference. Existing run directories cannot be overwritten by inference.
No resume scheduler was added because the measured run is short; completed model
caches and explicit setup/failure records survive errors.

All models run sequentially on the same Windows CPU machine. The common timing
boundary is `Forecaster.predict`, including input/output translation and LightGBM
feature generation, excluding model load/download and backtester context selection.
TSFM native-call timing is additionally recorded but is a narrower scope.
LightGBM dataset preparation and fitting are separate from inference. Torch uses
four CPU threads; LightGBM retains its original one thread. These are observed
costs of the frozen settings, not controlled equal-thread kernel benchmarks.
Reliable peak RAM/VRAM measurements were unavailable and are omitted; checkpoint
bytes must not be interpreted as memory use.

Run from the repository root, with the original local snapshot and official
checkpoint caches already present:

```powershell
python -m pip install -e ".[dev,experiments,chronos,t0]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m mypy scripts/benchmark_tsfm.py
python scripts/benchmark_tsfm.py --config configs/experiments/tsfm_zero_shot_v1.toml --validate-only
python scripts/benchmark_tsfm.py --config configs/experiments/tsfm_zero_shot_v1.toml --timing-only
python scripts/benchmark_tsfm.py --config configs/experiments/tsfm_zero_shot_v1.toml
# Rebuild scoring and figures without inference:
python scripts/benchmark_tsfm.py --config configs/experiments/tsfm_zero_shot_v1.toml --evaluate-only
```

Use the frozen package versions in the TOML/environment record. The actual run
uses `.venv/chronos-smoke/Scripts/python.exe`; its CPU Torch wheel was previously
installed from the official PyTorch CPU index. A fresh inference reproduction
needs a new output directory in a copied config; it must rerun its timing sample.
No Python-source editing is required. Preserve the original artifacts for audit.

## Primary all-interval leaderboard

All four models score the same **23,520** finite pairs. Skill is relative to the
48h baseline and shown as a percentage here (CSV stores the fraction).

| RMSE rank | Model | MAE MW | RMSE MW | Bias MW | RMSE skill |
|---|---|---:|---:|---:|---:|
| 1 | t0-gamma | 439.03 | 866.33 | +84.38 | +15.647% |
| 2 | Chronos-2 | 466.98 | 938.13 | +41.02 | +8.656% |
| 3 | LightGBM | 549.23 | 1026.88 | -187.69 | +0.015% |
| 4 | 48h persistence | 511.42 | 1027.03 | -0.72 | 0% |

t0 reduces RMSE by **160.55 MW / 15.63%** relative to LightGBM; Chronos reduces it
by **88.75 MW / 8.64%**. t0 is 71.80 MW below Chronos. These are observed differences
in this period, without significance testing. LightGBM's 0.153 MW advantage over
persistence is essentially a tie, not meaningful superiority; its MAE and signed
bias are worse. Thus all advanced models have numerically lower primary RMSE,
but it would be misleading to say all decisively outperform the seasonal baseline
across metrics. The rerun reproduces original LightGBM and 48h predictions exactly
(maximum absolute prediction difference **0 MW** for both).

## Daylight leaderboard

The common daylight population contains **13,580** intervals. Night remains in
the separate primary population above.

| RMSE rank | Model | MAE MW | RMSE MW | Bias MW | RMSE skill |
|---|---|---:|---:|---:|---:|
| 1 | t0-gamma | 756.18 | 1140.10 | +146.39 | +15.649% |
| 2 | Chronos-2 | 795.63 | 1234.50 | +57.91 | +8.665% |
| 3 | LightGBM | 939.21 | 1351.34 | -337.05 | +0.021% |
| 4 | 48h persistence | 885.74 | 1351.62 | -1.25 | 0% |

The ordering and relative TSFM improvement persist during daylight. This result
is not solely a consequence of including many nighttime zeros.

Local-only research figure: **All-interval and daylight RMSE**, `artifacts/experiments/tsfm_zero_shot_v1/rmse_comparison.png`.
This original generated artifact is not distributed with the public source.

## Availability, context and raw outputs

| Model | Requested origins | Successful | Failed | Forecast availability | Valid pairs |
|---|---:|---:|---:|---:|---:|
| 48h persistence | 245 | 245 | 0 | 100% | 23,520 |
| LightGBM | 245 | 245 | 0 | 100% | 23,520 |
| Chronos-2 | 245 | 245 | 0 | 100% | 23,520 |
| t0-gamma | 245 | 245 | 0 | 100% | 23,520 |

There are no failure reasons to summarize. Every TSFM call receives **1342**
observed slots with **zero** missing slots. Each processes 243 normal 150-step
rollouts, one 146-step spring rollout and one 154-step autumn rollout, returning
96/92/100 targets respectively. The test snapshot contains no missing actuals,
so the common-support intersection loses no pairs. Native gap handling is tested
in adapter tests but is not stress-tested by this complete test-period context.

Raw negative predictions are retained: 5,397 t0 values (minimum -39.34 MW),
17 Chronos values (minimum -9.12 MW), and 73 LightGBM values (minimum -93.00 MW).
Persistence has none. These counts are not clipped away before scoring.

## Lead time and Belgian local hour

Common scored counts below are **per model**, with lower-inclusive, upper-exclusive
elapsed lead-time bins. The first target is 13 hours after issue; the final target
is 35.75/36.75/37.75 hours after issue on spring/normal/autumn target days.

| Lead bin h | Count | 48h RMSE | LightGBM RMSE | Chronos RMSE | t0 RMSE | Lowest |
|---|---:|---:|---:|---:|---:|---|
| [12,18) | 4,900 | 0.00 | 15.97 | 20.20 | 7.91 | 48h |
| [18,24) | 5,880 | 859.65 | 734.40 | 732.63 | 684.73 | t0 |
| [24,30) | 5,880 | 1794.11 | 1861.38 | 1665.29 | 1534.17 | t0 |
| [30,36) | 5,880 | 511.25 | 462.17 | 458.30 | 423.70 | t0 |
| [36,42) | 980 | 0.00 | 15.97 | 18.89 | 7.40 | 48h |

Rankings change in the nighttime lead bins: persistence exactly matches zero
observations, while raw learned outputs retain small offsets. Reference skill
is undefined in zero-reference-RMSE groups; RMSE itself remains valid. In the
daytime bins t0 leads. Lead time and time of day are coupled by the fixed daily
issue schedule, so these cannot be interpreted as independent horizon effects.

The full 24-row local-hour table is saved as `by_local_hour.csv`. Around the early
ramp (06:00–06:59), Chronos/t0 are nearly tied at 62.3/62.6 MW RMSE. t0 has the
lowest hourly RMSE from 07:00 through 20:59. Midday errors are largest: at 13:00
local, RMSE is 1704.6 MW for t0, 1841.2 for Chronos, 1994.0 for persistence and
2113.0 for LightGBM. Persistence leads at night and at the 21:00 local-hour group.
No weather condition is inferred from these error patterns.

## Monthly and seasonal results

| Target month | Count | 48h RMSE | LightGBM RMSE | Chronos RMSE | t0 RMSE | Lowest |
|---|---:|---:|---:|---:|---:|---|
| March | 2,972 | 954.16 | 1182.31 | 998.94 | 866.92 | t0 |
| April | 2,880 | 1229.38 | 1457.44 | 1080.60 | 1069.47 | t0 |
| May | 2,976 | 1133.79 | 1001.47 | 997.60 | 926.10 | t0 |
| June | 2,880 | 984.02 | 1030.79 | 773.92 | 783.10 | Chronos |
| July | 2,976 | 984.65 | 920.42 | 877.03 | 802.14 | t0 |
| August | 2,976 | 969.00 | 748.29 | 952.40 | 830.01 | LightGBM |
| September | 2,880 | 1169.12 | 978.57 | 1059.15 | 936.78 | t0 |
| October | 2,980 | 714.17 | 718.70 | 701.20 | 663.97 | t0 |

| Season | Count | 48h RMSE | LightGBM RMSE | Chronos RMSE | t0 RMSE |
|---|---:|---:|---:|---:|---:|
| MAM | 8,828 | 1110.36 | 1225.42 | 1025.85 | 956.70 |
| JJA | 8,832 | 979.20 | 905.88 | 871.86 | 805.56 |
| SON (Sep–Oct) | 5,860 | 964.95 | 856.33 | 895.19 | 809.62 |

t0 leads six months and all three represented seasonal groups. Chronos leads
June and LightGBM leads August; the small June difference is not a significance
claim. Chronos does not beat persistence in March. Seasonal tables are partial
coverage summaries, not evidence about winter or a full annual cycle.

## Daily wins, complementarity and selected examples

All 245 target days have complete common support; there are **no daily ties**.

| Model | Days won | Percent of days |
|---|---:|---:|
| LightGBM | 69 | 28.16% |
| Chronos-2 | 59 | 24.08% |
| 48h persistence | 59 | 24.08% |
| t0-gamma | 58 | 23.67% |

Aggregate RMSE winner t0 does not win the most days. Daily win frequency discards
the size of each win/loss, while pooled RMSE weights squared interval errors;
these answer different questions. Even the overall winner has substantial bad
days, so no universal day-level dominance is established.

| Pair | Signed interval error correlation | Daylight error correlation | Daily RMSE correlation |
|---|---:|---:|---:|
| LightGBM / 48h | 0.570 | 0.579 | 0.162 |
| LightGBM / Chronos | 0.588 | 0.601 | 0.080 |
| LightGBM / t0 | 0.667 | 0.695 | 0.137 |
| Chronos / 48h | 0.652 | 0.652 | 0.476 |
| t0 / 48h | 0.709 | 0.711 | 0.564 |
| Chronos / t0 | 0.918 | 0.920 | 0.840 |

The least-correlated **interval errors** are LightGBM/persistence. Among advanced
models, LightGBM/Chronos is least correlated; its daily-RMSE correlation is also
the smallest, 0.080. The two TSFMs share many error patterns. This supports
investigating complementarity later with separate training/validation data; low
correlation alone does not establish a useful or leakage-safe ensemble.

Predeclared diagnostic rules select:

| Selection | Target date | Criterion |
|---|---|---|
| Difficult-day plot | 6 July 2025 | Largest mean daily RMSE: 2301.78 MW |
| All-model struggle | 16 August 2025 | Largest minimum model daily RMSE: 1835.25 MW |
| Strongest LightGBM sole win | 13 April 2025 | 1460.92 MW below second-best daily RMSE |
| Strongest Chronos sole win | 24 April 2025 | 680.25 MW below second-best |
| Strongest t0 sole win | 18 April 2025 | 432.45 MW below second-best |
| Strongest persistence sole win | 20 September 2025 | 974.57 MW below second-best |

The plots below show all four raw forecasts, including both strong and weak TSFM
examples. On the difficult day, actual PV is much lower than all four forecast
profiles; this is an observation about power trajectories, not proof of cloud
conditions. The predeclared dates show materially different relative fits and
are not used as evidence in place of the complete tables.

Local-only research figure: **Three predeclared target days**, `artifacts/experiments/tsfm_zero_shot_v1/representative_periods.png`.
This original generated artifact is not distributed with the public source.

Local-only research figure: **Difficult day selected by mean model RMSE**, `artifacts/experiments/tsfm_zero_shot_v1/difficult_day.png`.
This original generated artifact is not distributed with the public source.

## Accuracy and compute

Tested environment: Windows 11 build 26200; Intel64 Family 6 Model 142 Stepping
12, GenuineIntel; Python 3.12.4; Torch 2.15.0+cpu; Transformers 5.23.0;
LightGBM 4.7.0; NumPy 2.5.4; pandas 3.1.1. Full package/hardware records are
in `freeze.json` and `environment.txt`. Torch uses four threads; LightGBM one.

| Model | RMSE MW | Mean predict call s | Cached local load s | Local training s | Model size indicator |
|---|---:|---:|---:|---:|---|
| 48h | 1027.03 | 0.00250 | 0 | 0 | No learned weights |
| LightGBM | 1026.88 | 0.03523 | N/A: fitted locally | 0.768 | 100 trees, at most 15 leaves/tree |
| Chronos | 938.13 | 0.25733 | 0.756 | None | 119.48M parameters; 477.9 MB weights |
| t0 | 866.33 | 0.30697 | 2.576 | None | 255.61M parameters; 1022.5 MB weights |

LightGBM preparation takes another **10.757 seconds** before fitting. TSFM
native-call means are 0.25355/0.30305 seconds; they exclude adapter translation,
so the common predict-boundary means above are the primary timing comparison.
All means include the first prediction after separately timed loading; no
best-run selection or discarded slow calls is used. Cached Hub resolutions are
about 0.0011/0.0008 seconds; initial download time is excluded.

Mean t0 calls are 8.71 times LightGBM calls (Chronos: 7.31 times), but both remain
below one second at a daily national-series cadence on this machine. Summed
backtest wall times, including context/engine overhead, are 4.07, 13.29, 66.84
and 78.89 seconds respectively. The first complete frozen run, including data,
training, loading and diagnostics, spanned about 223 seconds, from
17:14:20.789 to 17:18:03.970 UTC on 14 November 2026. These CPU observations do
not include provider pretraining cost, cold downloads or peak memory, and do not
generalize directly to other hardware or many-series deployment.

For this daily CPU workflow the additional inference time is modest relative
to the observed 8.64–15.63% RMSE reduction against fixed LightGBM. Whether the
storage/dependency, memory, latency and maintenance costs are economically
justified depends on the deployment and value of forecast errors; no business
cost/benefit threshold was tested. Statistical/economic value is not established
by a speed/accuracy table alone.

## Scientific acceptance audit

- [x] Real national Elia PV snapshot used; attribution and hashes retained.
- [x] Test dates/configuration frozen before this run's results; **reused known holdout**, not fresh.
- [x] All four models use the same 245 issue times.
- [x] All use the same next-Brussels-day target policy and timestamps.
- [x] Primary and daylight tables use all-four-model common support.
- [x] No local Belgian TSFM training/fine-tuning/weight updates; **provider pretraining overlap unknown**.
- [x] No weather/NWP/reanalysis supplied; deterministic LightGBM solar features disclosed.
- [x] No direct after-cutoff PV supplied to inference; contract/canary checks passed.
- [x] No Elia forecast fields used as inputs.
- [x] TSFM context was approved before final evaluation, with no context search.
- [x] LightGBM unchanged and not retuned; original predictions reproduced exactly.
- [x] Failed/missing predictions never replaced; this run has no failures.
- [x] Negative predictions preserved and counted.
- [x] Nighttime retained in the primary population; daylight shown separately.
- [x] Both DST target days included with correct 92/100 targets and bridge lengths.
- [x] Origin and interval forecast availability reported for every model.
- [x] Hardware/device, package versions, source/data hashes and timing scope recorded.

These checks certify the local temporal/input procedure, **not** historical
publication vintages or pretraining exclusion. Those two scientific uncertainties
prevent an operational-as-of or independently uncontaminated-holdout claim.

## Verification, artifacts and limitations

Before final inference: **351 offline tests passed, two native tests skipped**;
the combined-runtime model/experiment regression subset passed **100 tests**;
the leakage/cutoff/future/optional-import subset passed **22 tests** (122 deselected).
Ruff passed, 71 Python files passed format checking, strict mypy passed 41 source/
test files and the new runner, and frozen configuration validation confirmed
245 origins. The expensive benchmark is not part of CI. New offline tests cover
configuration drift/hash pinning, checkpoint/context/runtime/cache identity,
cache corruption, retained failures/common support, daily tie rules, timing
summary and deterministic case selection.

The explicit preflight regression commands were:

```powershell
.\.venv\chronos-smoke\Scripts\python.exe -m pytest tests/test_chronos.py tests/test_t0.py tests/test_lightgbm.py tests/test_lightgbm_data.py tests/test_tsfm_experiment.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_backtesting.py tests/test_features.py tests/test_lightgbm_data.py tests/test_lightgbm.py tests/test_chronos.py tests/test_t0.py -k "canary or future or cutoff or rolling or dependency or imports" -q
```

After inference, `--evaluate-only` ran successfully in the base environment
**without Torch/Chronos/t0 installed**. All cached file checksums passed, and every
derived CSV/PNG checksum matched the first evaluation exactly; only the results
manifest's evaluation timestamp changed. The initial manifest is retained under
`artifacts/audits/tsfm_zero_shot_v1_first_manifest.json` for that audit. All three
figures were visually inspected for readable labels, legends and aligned curves.

The output directory is `artifacts/experiments/tsfm_zero_shot_v1/` (48 files,
approximately 22.7 MB before later rescoring). It contains:

- `config.toml`, `parent_config.toml`, `freeze.json`, `environment.txt`, `snapshot.json`, `data_quality.json`, `training.json`;
- `predictions_<model>.csv`, `origins_<model>.csv`, `calls_<model>.csv`, `cache_<model>.json` for four models;
- `leaderboard.csv`, `daylight_leaderboard.csv`, `aligned_predictions.csv`, `availability.csv`, `failure_reasons_<model>.csv`;
- `horizon_metrics.csv`, `by_horizon.csv`, `by_local_hour.csv`, `by_month.csv`, `by_season.csv`, `daily_metrics.csv`, `daily_winners.csv`, `daily_win_summary.csv`, `selected_cases.csv`;
- `error_correlations.csv`, `daylight_error_correlations.csv`, `daily_rmse_correlations.csv`, `compute_summary.csv`;
- three PNGs and `results_manifest.json` with file checksums.

The timing-only directory is separate. The original history-only LightGBM experiment configuration,
all 24 recorded source files and all 34 recorded artifacts remain hash-identical.
The original run had no committed source revision; recorded source hashes identify
the implementation.

Assumptions are retrospective national MW estimates, 30-minute event-time safety
lag, fixed Brussels daylight/solar geometry, native TSFM P50, single local model
fit and fixed weights during test. Their limitations include revised data rather
than original issue-time vintages, unknown pretraining overlap, known holdout
reuse, no winter/multi-year replication, unequal feature representations and
mean-versus-median objectives, missing-feature shifts between LightGBM train/test,
no capacity normalization, no reliable peak memory measurement, one CPU runtime,
and no formal statistical significance assessment. No claim is made about tuned
classical alternatives, production forecasting, probabilistic calibration, Elia
forecast superiority, weather-informed skill or causal effects of pretraining.
The weights/runtime were selected in November 2026; no claim is made that these
same models and data vintages were available at the historical 2025 issue times.

**Conclusion:** pretrained zero-shot models add measurable error reduction over
this frozen local classical recipe in the tested period, with modest absolute
daily inference time on the tested CPU. They do not dominate every day or month.
The most valuable next scientific question is whether historically available
weather forecasts add information beyond PV history under the same operational
cutoff. Archived weather comparisons require forecast issue/availability evidence and coverage checks, with a fresh confirmatory period for new modeling decisions. Future observations or reanalysis cannot substitute for forecast weather.
