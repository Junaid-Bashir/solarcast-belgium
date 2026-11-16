# LightGBM: initial supervised formulation

One pooled regressor maps an issue/target row to national PV MW. Inputs are
deterministic target calendar/solar features, elapsed horizon in hours, and PV
lag/rolling features computed from that origin's bounded ForecastContext. It is
not 96 independent models and does not recursively predict future PV inputs.

Training origins are an explicit BacktestSpec (daily 11:00 Brussels in the example,
configurable there). The dataset builder reconstructs each context using the same
availability and context-window rules as backtesting. Labels are looked up separately
and must be complete/eligible by the fit-time policy cutoff. Missing labels are
excluded and counted; missing features remain NaN. All target intervals, including
nighttime, are eligible for training.

Train once with a fixed estimator count, then evaluate on later held-out issues.
Prediction before the declared fit time is rejected. Validation is not used for
early stopping or tuning in this formulation; an optional separate validation period
in the example is reserved and unused. No test data informs model selection.

The default history-only recipe uses no capacity, Elia forecasts, observed weather or NWP. Availability remains
an explicitly labelled approximation, not verified Elia publication latency.

## Installation and native interface

```powershell
python -m pip install -e ".[dev,lightgbm]"
```

The `lightgbm` extra adds `lightgbm>=4.6,<5`; base installs do not require it.
The adapter uses the [native Dataset/train/Booster API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html),
so no scikit-learn dependency is added. Importing its configuration/module does
not import LightGBM; missing support raises an installation hint at fit time.
For the released package, use `python -m pip install "solarcast-belgium[lightgbm]"`.
The editable installation above is for source development.

```python
from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest
from solarcast.models.lightgbm import LightGBMForecaster
from solarcast.models.lightgbm_data import build_training_dataset

policy = EventTimeCutoff("30min")
train_spec = BacktestSpec(train_issues, availability_policy=policy, context_window="14D")
training = build_training_dataset(pv_data, train_spec, fit_time=fit_time, training_window="365D")
model = LightGBMForecaster()
model.fit(training)
test_spec = BacktestSpec(test_issues, availability_policy=policy, context_window="14D")
result = backtest(model=model, data=pv_data, spec=test_spec)
```

The shared inference contract does not require a generic training protocol. This adapter
adds `fit(TrainingDataset)` while satisfying the existing `predict(context)`
Forecaster protocol. Baselines still need no fit. A shared `_build_context` helper
is extracted from backtesting and reused for training origins; no LightGBM-specific
branch is added to backtesting or evaluation.

## Fixed feature schema

`lightgbm_features(context, location=...)` is shared by training and prediction.
It returns exactly these 13 float64 columns in this order:

| Feature | Category | Unit/encoding | Availability and reason |
|---|---|---|---|
| `time_of_day_sin` | Future-known | Brussels clock sine | Deterministic diurnal phase |
| `time_of_day_cos` | Future-known | Brussels clock cosine | Paired phase without midnight discontinuity |
| `day_of_year_sin` | Future-known | Leap-year-aware annual sine | Deterministic seasonal phase |
| `day_of_year_cos` | Future-known | Annual cosine | Paired annual phase |
| `solar_apparent_elevation_deg` | Future-known | Degrees | Representative solar height |
| `solar_azimuth_deg` | Future-known | Degrees clockwise from north | Representative solar orientation |
| `clearsky_ghi_w_m2` | Future-known | W/m², Haurwitz | Nonlinear clear-sky reference, not weather |
| `horizon_hours` | Future-known | Elapsed target-minus-issue hours | Pooled multi-horizon conditioning |
| `pv_lag_24h_mw` | Historical-observed | MW, exact elapsed 24h | Eligible daily profile or NaN |
| `pv_lag_48h_mw` | Historical-observed | MW, exact elapsed 48h | Earlier daily profile or NaN |
| `pv_lag_7d_mw` | Historical-observed | MW, exact elapsed 168h | Earlier weekly profile or NaN |
| `pv_trailing_24h_mean_mw` | Historical-observed | MW | Complete recent-context summary |
| `pv_trailing_24h_max_mw` | Historical-observed | MW | Complete recent-peak summary |

Raw hour/minute, quarter-hour, month, zenith and daylight flag substantially
duplicate selected phases or elevation and are not also included. This initial
recipe is not a feature-selection result from the test set. It inherits all
[feature conventions](../features.md), including the rounded Brussels reference
(50.85°N, 4.35°E) and fixed refraction assumptions, not observed weather inputs.

Externally supplied `context.features` is deliberately ignored. The model owns
this audited recipe; raw PV labels, capacity, Elia forecasts, provenance and other
extra columns do not enter X. Names, order and float64 dtypes are strict at fit
and prediction. Missing/extra/reordered schema or infinity fails, not silently
reordered or dropped. Arbitrary valid target counts, including 92/96/100, work.

## Training rows and temporal bounds

The builder returns owned `X`, `y`, audit rows (`issue_time`, `target_time`,
`history_cutoff`) and policy/window/location/fit metadata. Treat these scalar-valued
tables as a snapshot; do not edit them to disguise feature provenance. This is
a model-specific dataset, not a universal ML data framework.

Every training origin uses the same availability policy and context-window path
as backtesting. Future target features are deterministic; historical features use
only that origin's eligible history. y is joined separately from labels permitted
by the same policy evaluated at **fit time**. All requested target intervals must
finish by its label cutoff; split-crossing schedules raise instead of silently
truncating days. Training origins must precede fit time. Missing, absent or
policy-unavailable labels are excluded and counted; no labels remaining is an error.
Invalid training origins fail, even if spec.on_error is continue.

`training_window=None` uses all scheduled origins. A positive elapsed window limits
origin times to `[fit_time-window, fit_time)`. This differs from context_window,
which limits history at each origin. Early examples may still need observations
older than the training-origin window for their historical features.

Fit validates feature schema, aligned finite labels and audit temporal bounds.
Prediction before the model's declared fit time raises a possible-training-leakage
error. A failed refit clears old state rather than silently retaining stale weights.
Only the booster and compact training metadata remain on the fitted model, not
the complete training matrix. Computational training latency is not simulated;
choose an appropriate fit/evaluation gap in experiments.

The example defines chronological training origins, a fit boundary, a reserved
unused validation period and a later untouched test period. Validation target
intervals must finish before the first test issue. With fixed estimator counts,
neither validation nor test data selects features, parameters or stopping iteration.
Later eligible observations, including earlier test days, may enter subsequent
inference contexts under the policy; model weights stay fixed. This is ordinary
rolling-origin inference, not retraining.

## Availability and missing values

The existing approximation is unchanged: cutoff=floor_15min(issue-safety_lag),
and an eligible interval must finish by cutoff. The builder also applies this
policy to labels at fit time. The example explicitly shares one policy and context
window between training/evaluation. Caller-defined experiments must maintain
compatible policies; ForecastContext does not encode full policy identity.

Current retrospective values can still differ from historically published ones.
Elia latency/revisions remain unresolved; no available_at is fabricated. Capacity
vintages remain excluded.

LightGBM's [documented missing-value handling](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html#missing-value-handle)
uses NaN. `use_missing=True`, `zero_as_missing=False` are explicit here: genuine
zero PV stays zero. Day-ahead 24h lags or limited-context lag/rolling features
may be NaN. They are not filled, interpolated or fetched from outside context.
Nighttime rows remain in training. Raw predictions, including negatives, are
returned without clipping, nighttime masking or capacity normalization.

## Configuration and reproducibility

`LightGBMConfig` exposes n_estimators=100, learning_rate=0.05, num_leaves=15,
min_child_samples=20, random_state=0. Training uses squared-error regression,
one CPU thread, no row/column sampling, deterministic=True, force_col_wise=True
and quiet verbosity. See [official parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
No GPU, callback framework, early stopping or tuning search is exposed.

`training_info` retains fit/label cutoff, actual target bounds, row/exclusion counts,
ordered features, config, training spec/window/location, target/unit, and LightGBM/
SolarCast versions. Forecast model_version records the LightGBM version, not a
unique trained-artifact ID. Serialization and remote artifacts are not implemented.

Same data, recipe, configuration, seed and library/build should produce equivalent
predictions; tests check this. Compiler/platform/library floating-point differences
may change results across environments. Record dependencies and provenance; no
cross-version/cross-hardware bitwise guarantee is made. Logging reports row counts,
periods, feature count and configuration rather than printing library output.

## Running experiments

```powershell
python examples/benchmark_lightgbm.py
python examples/benchmark_lightgbm.py --help
python examples/benchmark_lightgbm.py --input-csv data/normalized_pv.csv
```

Defaults use only a small **synthetic daily profile**: training origins 9–28 June
2026 at 11:00 Brussels, fit boundary 30 June 08:00 UTC, reserved validation origins
30 June–2 July, test origins 4–6 July, shared 30min safety lag and 14D context.
The example fixes 60 trees before evaluation. The first run trained 1,920 rows and
scored 288 common test observations; 48h persistence scored better than LightGBM.
These are software checks, not evidence of real Belgian PV forecasting quality.

Explicit real-data use accepts either --input-csv or both --fetch-start and
--fetch-end with aware timestamps. CSV times must contain offsets. No network
request occurs on import, by default, or when loading CSV. There is no implicit
years-long fetch or fallback source. Provide sufficient history and check data
licensing before redistribution. No real-data run was performed here because
there was no local Elia cache.

Configure --train-start/--train-end, --fit-time, --validation-start/--validation-end,
--test-start/--test-end, --issue-time, --safety-lag, --context-window,
--training-window, --n-estimators and --seed. Date bounds are Brussels **origin
dates**, start inclusive/end exclusive. The issue clock is local; ambiguous or
nonexistent DST issue times raise. Dates/paths live in the example, not core code.

## Verification and limitations

Offline tests cover target/future leakage canaries, origin-specific features,
label cutoffs, missing targets/features, arbitrary horizons/DST, native fitting,
determinism, raw negatives, schema errors, stale-refit prevention, optional imports
and model-agnostic evaluation. Base-only runs skip native tests when the extra is
absent; builder tests still run. Optional CI jobs exercise Windows/Linux native fits.

The original history-only formulation uses no NWP/weather, capacity or Elia-forecast
features, hyperparameter optimization, probabilistic forecasts, periodic retraining
or TSFMs. Its later real-data experiments and optional NWP extension are documented
separately and below.

## Optional history + NWP configuration

The default `LightGBMForecaster()` retains the original history-only feature recipe
and model ID. The same implementation now supports `feature_set="history+radiation"`,
`"history+radiation+cloud"`, or `"history+nwp"`. The last adds all seven approved
weather features to the original 13; the intermediate choices support a small
validation ablation. Hyperparameters remain independently controlled by the same
`LightGBMConfig` and are unchanged in the first weather experiment.

```python
training = build_training_dataset(
    pv,
    training_spec,
    fit_time=fit_time,
    feature_set="history+nwp",
    weather_provider=local_weather,
)
model = LightGBMForecaster(config, feature_set="history+nwp")
model.fit(training)
result = backtest(model=model, data=pv, spec=test_spec, weather_provider=local_weather)
```

Training reconstructs each historical issue using the shared context builder and
the historical GFS ingestion layer vintage selector before joining fit-time-eligible labels. Every NWP
feature must be finite; a missing run/time/variable fails the whole origin.
Strict training specs raise; continue specs exclude that origin and retain
`origin_failures`, requested-row counts and per-row vintage/precision audits.
No model-side fetch, observation/reanalysis substitution, model-specific weather
interpolation, implicit filling or PV prediction clipping occurs. History feature
NaNs retain their existing meaning. Feature-set mismatches and post-issue weather
availability in training audits are rejected.

Inference requires a target/issue-aligned `context.weather` only for weather
configurations. History-only inference ignores optional weather and remains usable
without the weather extra. The [controlled experiment](../experiments/weather-lightgbm-v1.md)
documents forecast-only training, paired training support, validation selection,
the explicit radiation packing-bound policy and immutable TSFM cache reuse.
