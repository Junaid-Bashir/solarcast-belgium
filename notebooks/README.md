# Interactive real Belgian PV forecast analysis

Open `01_real_belgian_forecast_analysis.ipynb`. The source notebook is clean and
self-contained; a fully executed copy and all derived outputs are kept separately
under ignored `artifacts/analysis/01_real_belgian_forecast_analysis/`.

This is analysis of the already-known 2025 development population. No model is
trained, tuned, loaded or used to regenerate forecasts. No ensemble is fitted.
Any resulting hypothesis needs a fresh confirmatory period.

## Windows setup and launch

From the repository root, use an analysis environment containing an editable
SolarCast installation. The commands below use `.venv/weather-smoke` as an example
relative environment name; create it first with `python -m venv .venv/weather-smoke`
and install `python -m pip install -e .` with that environment's interpreter. Notebook
tools are development-only requirements, separate from the package dependency metadata
and all core runtime dependencies:

```powershell
.\.venv\weather-smoke\Scripts\python.exe -m pip install -r notebooks/requirements.txt
.\.venv\weather-smoke\Scripts\python.exe -m ipykernel install --user --name solarcast-analysis --display-name "SolarCast — analysis (.venv/weather-smoke)"
.\.venv\weather-smoke\Scripts\python.exe -m jupyterlab notebooks/01_real_belgian_forecast_analysis.ipynb
```

Choose **SolarCast — analysis (.venv/weather-smoke)** as the kernel and run all cells.
To view the already executed results instead:

```powershell
.\.venv\weather-smoke\Scripts\python.exe -m jupyterlab artifacts/analysis/01_real_belgian_forecast_analysis/executed.ipynb
```

The executed copy exists only after local execution; it is not distributed with
the repository. The original frozen data/prediction bundles listed below are
required and are not downloaded or provided by this guide.

The source also exposes `selected_date` and a simple date/Chronos widget. Plotly
figures support zoom, hover and legend toggles. No dashboard framework is installed.
Standalone chart HTML files use the local `plotly.min.js` beside them, not a CDN;
keep that file with any exported HTML you move.

Execute non-interactively, without modifying the source notebook:

```powershell
.\.venv\weather-smoke\Scripts\python.exe notebooks/execute_analysis.py
```

The runner writes only the separate executed notebook. A fresh environment needs
the editable SolarCast package and these requirements; the analysis does not need
LightGBM, Torch, t0 or Chronos runtimes. The existing environment happens to contain
some optional libraries, but notebook execution explicitly verifies that model
runtimes were not imported.

## Required existing, real inputs

This notebook does not download missing bundles. It fails clearly on missing or
corrupted inputs and authenticates them before analysis:

- `artifacts/experiments/lightgbm_baseline_v1/snapshot.json` and its referenced
  23 CSV/JSON pairs in `data/elia_benchmark_v1/`: original real national Elia PV.
- `artifacts/experiments/weather_lightgbm_v1/aligned_predictions.csv`: original
  four-model common population, including recorded actuals. Every actual is
  checked against the authenticated raw snapshot.
- `predictions_chronos.csv` from that same weather experiment: existing fifth-model
  forecasts on exactly matching keys. Chronos is present in this frozen bundle.
- `test_weather_features.csv` and `weather_availability.csv`: existing GFS forecasts
  selected under `gfs_init_plus_6h_v1`, with no observed weather or reanalysis.
- `extended_leaderboard.csv`, `leaderboard.csv`, `daylight_leaderboard.csv`,
  `daily_metrics.csv`, `daily_winners.csv`, `daily_win_summary.csv`,
  `by_local_hour.csv`, `horizon_metrics.csv`, `by_month.csv`, `weather_regimes.csv`,
  `validation_ablation.csv`, and `error_correlations.csv`: original definitions and
  scoring populations.
- Frozen manifests/metadata in all three experiment directories, including the
  the weather-informed LightGBM experiment pre-fit/inference freezes and its separately recorded reporting amendment.

The notebook checks all 75 the weather-informed LightGBM experiment manifest entries, original LightGBM/TSFM experiment
artifact checksums, snapshot shards and sidecars, forecast key/truth equality,
weather eligibility under the declared assumption, and exact 48h persistence
against historical PV. It does not claim current source/environment identity is
identical to an old run. All clocks are timezone aware; per-day plot labels show
Brussels offsets so DST folds are distinguishable.

## Investigation and population labels

The notebook proceeds through actual PV, a calendar-selected week, original
leaderboards, interactive day profiles, every major model's best/worst days,
daily winners/errors, hourly/horizon/monthly behavior, midday consistency, forecast
radiation/cloud regimes, t0 and persistence counterexamples, bias/negative outputs,
shape/residual diagnostics, and signed/absolute-error complementarity.

The original primary population remains **23,520 quarters / 245 origins**; daylight
is **13,580**. Stored primary daily winners remain the four-model definition,
even though Chronos appears in the five-model leaderboard. Additional midday
([10:00,15:00) Brussels), shape, error-tail and astronomical twilight analyses are
explicitly labelled diagnostics. They do not replace the primary ranking.

Examples are selected transparently: first full Monday–Sunday week (3–9 March),
stored daily-RMSE minima/maxima, or chronological first/middle/last eligible
t0/persistence-winning days. No model-success-only showcase is substituted.
Radiation plots distinguish regional W/m² from national PV MW, without inventing
a national capacity-weighted GHI feature. Cloud thresholds remain .25/.75 and
spatial cloud standard deviation .15; spatial variation is not actual or temporal
weather variation.

## Findings worth inspecting

- Weather-LightGBM retains the original 624.19 MW RMSE leadership and wins monthly
  aggregate RMSE in all eight observed months, but only 94/245 days outright.
- **t0 beats weather in pairwise daily RMSE on 101 days**, distinct from its 62
  sole four-model victories. Calendar-spaced examples are 3 March, 28 June and
  27 October.
- Weather improves midday RMSE over history on **176/245 days**, and over t0 on
  **147/245**. The top 25 improvement days supply **45.15% of net midday SSE
  reduction**: gains are broad but uneven, not just one exceptional day.
- t0 is closer on **64.87% of all individual quarters**, but this is heavily
  influenced by **91.63% of non-daylight quarters**; its daylight fraction is
  **45.28%**. Point-win frequency is different from MAE/RMSE and error-tail size.
- Persistence-winning days have median observed 48h-profile correlation **.9908**
  versus **.9629** on other days. This is partly expected after selecting on its
  forecast error. Within-day forecast cloud spread is nearly equal (.185 versus
  .187), so the analysis does not establish a stable-weather explanation.
- History-LightGBM has mean error **−187.69 MW** despite median error **+15.99 MW**
  and overforecasting on about 67% of quarters: error signs/counts do not determine
  bias when magnitudes are asymmetric. Night/day and error-tail views matter.
- Negative predictions remain untouched: history 73 (minimum −93.00 MW), weather
  2 (−1.91 MW), t0 5,397 (−39.34 MW), Chronos 17 (−9.12 MW), persistence 0.
  Most t0 negatives are astronomical night (4,877); history's include 49 daylight
  intervals, so not all negative outputs can be described as night only.
- Weather/t0 signed-error Pearson correlation remains **.5428** on the identical
  primary population. When weather absolute errors are in their upper quartile,
  t0 is closer on **58.78%** of those quarters. This is outcome-selected diagnostic
  evidence for a later fresh-period ensemble study, not usable advance selection
  or a measured ensemble improvement. No weights or blended predictions are computed.

## Hypotheses, not conclusions

Future experiments can test whether radiation/cloud gains persist independently;
whether t0 adds value to the complementary cases; whether peak-amplitude/fleet
nonstationarity contributes to failures; and whether uncertainty is calibrated.
None of these plots isolates NWP error from PV-mapping error, fleet changes,
observation revisions or unknown TSFM pretraining overlap. GFS publication/as-of
fidelity is still an explicit six-hour approximation, not observed evidence.

## Verification and outputs

`NotebookClient` executes all code cells from a fresh kernel. The notebook blocks
external socket connections and writes/deletes/renames under experiment, source
data and frozen-configuration roots. Before/after SHA-256 signatures verify
**2,960 protected files unchanged**. It asserts no model runtime was imported.

Generated output includes the executed notebook, **51 interactive HTML figures**,
local Plotly JS, and separate CSV/JSON bias, negatives, shape, midday, persistence,
weather and verification tables. The executed notebook is about 36 MB and is
kept outside Git; the clean source is the versionable research artifact.
Rendering checks and representative screenshots are also stored in the analysis
directory. No frozen model, prediction cache, configuration or manifest is overwritten.

The notebook performs no model fitting, forecast regeneration or publication.
