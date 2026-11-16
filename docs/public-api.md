# Public Python API

SolarCast 0.1.0 provides a small benchmarking workflow over the existing research
APIs. It does not download PV/NWP, choose a winning model, tune models, save runs,
or offer an operational forecast service. Existing lower-level imports remain valid.

```python
from solarcast import (
    benchmark,
    available_models,
    BenchmarkConfig,
    BenchmarkResult,
    ModelInfo,
    __version__,
)
```

## First benchmark (offline, synthetic)

This runnable fixture verifies installation. Constant 100 MW is fictional and is
neither realistic solar generation nor evidence of Belgian forecasting performance.

```python
import pandas as pd
from solarcast import benchmark, BenchmarkConfig

pv_data = pd.DataFrame(
    {
        "target_time": pd.date_range("2025-01-01", "2025-01-12", freq="15min", tz="UTC"),
        "pv_generation_mw": 100.0,
    }
)
config = BenchmarkConfig(
    issue_times=pd.date_range("2025-01-09 11:00", periods=2, freq="D", tz="Europe/Brussels"),
    observation_safety_lag="30min",
    reference_model="seasonal_persistence_48h",
)
results = benchmark(
    target="belgium_pv",
    models=["seasonal_persistence_48h", "last_value_persistence"],
    data=pv_data,
    config=config,
    data_source="SYNTHETIC constant fixture",
)
print(results.leaderboard())
```

Fixture results (selected columns):

| rank | model_id | mae_mw | rmse_mw | bias_mw | n_scored | forecast_availability |
|---|---|---|---|---|---|---|
| 1 | last_value_persistence | 0 | 0 | 0 | 192 | 1.0 |
| 1 | seasonal_persistence_48h | 0 | 0 | 0 | 192 | 1.0 |

Both skill values are undefined (NaN): the reference's RMSE is zero. Exact ties
share rank. The default model list contains only `seasonal_persistence_48h`.
Run `python examples/quickstart.py` for a smaller offline smoke test.

## Data: explicit local inputs

Only `belgium_pv` is supported: the national Belgian PV generation estimate, MW,
using Elia Belgium/Total where the source schema has grid categories. Supplying
this target name does not verify the provenance of arbitrary caller data.

`data` is a pandas DataFrame containing:

- `target_time`: timezone-aware quarter-hour **interval starts**, unique and non-null;
- `pv_generation_mw`: real numeric, nullable MW values.

SolarCast copies and sorts the snapshot and normalizes timestamps to UTC. Naive,
duplicate, misaligned timestamps and infinite values are rejected. Missing and
negative observations stay unchanged. Include enough warm-up history for model
lags, training labels when fitting, and evaluation actuals. Missing seasonal lag
inputs fail an entire origin; missing actuals reduce scoring support, never become zero.
Extra source/provenance columns can remain in the input; they do not enter model context.

Prepare normalized CSV/Parquet using your own storage, or deliberately fetch a
bounded interval with the existing `solarcast.data.EliaPVClient.fetch_history`.
See [Elia source contract](data/elia.md). Fetching is a separate, explicit step;
`benchmark()` has no managed-data or automatic fetch mode. Retrospective snapshots
cannot reconstruct historical revisions or certify publication availability.

## Models and discovery

```python
for model in available_models():
    print(model.model_id, model.status, model.extra, model.missing_dependencies)
```

| Identifier | Extra | Behavior/prerequisite |
|---|---|---|
| `last_value_persistence` | base | Latest eligible non-null MW, unchanged |
| `seasonal_persistence_48h` | base | Exact target minus 48 elapsed UTC hours |
| `seasonal_persistence_7d` | base | Exact target minus seven elapsed UTC days |
| `lightgbm` | `lightgbm` | Fixed history-only recipe; explicit chronological fit |
| `lightgbm_history_radiation_cloud` | `lightgbm` | Existing 19-feature recipe; explicit fit and local historical NWP |
| `t0` | `t0` | Pinned t0-gamma, raw native P50; prepared weights |
| `chronos` | `chronos` | Pinned Chronos-2, raw native P50; prepared weights |

These are existing adapter result IDs. Ambiguous aliases `seasonal_persistence`
and `lightgbm_weather` are intentionally omitted: the lag and feature recipe are
visible in the name. Other seasonal lags/feature recipes remain available through
the advanced APIs. There is no `best` or `ensemble` identifier.

Discovery returns immutable `ModelInfo` records, including unavailable models:
`status` is `built_in`, `installed`, or `unavailable`; `extra`,
`missing_dependencies`, `requires_training`, and `requires_weather` are inspectable.
It checks distribution metadata without importing ML runtimes. Installed means
the declared distributions are present, **not** that compatible binaries, devices,
weights, licenses, or sufficient data are verified. Runtime errors retain their cause.

From a checkout, install only what you need:

```shell
python -m pip install -e "."                  # persistence
python -m pip install -e ".[lightgbm]"        # history or prepared-weather LightGBM
python -m pip install -e ".[t0]"              # independent t0 support
python -m pip install -e ".[chronos]"         # independent Chronos support
python -m pip install -e ".[lightgbm,weather]" # optional native NWP preparation tools
```

The released package is available with `python -m pip install solarcast-belgium`.
For example, `python -m pip install "solarcast-belgium[lightgbm]"` adds optional
LightGBM support. Local wheel installations can use the same extras on the wheel
path. Base imports and persistence do not require weather or
TSFM extras. Missing optional distributions cause an actionable ImportError before
running any model. Unknown model/target/prerequisite mistakes raise ValueError.

## Configuration and expensive operations

`BenchmarkConfig` keeps scientific choices visible without a long function signature:

| Field | Semantics/default |
|---|---|
| `issue_times` | Required aware, ordered, unique DatetimeIndex; no implicit period |
| `observation_safety_lag` | Required nonnegative elapsed duration string |
| `context_window` | `"14D"`, elapsed and issue-anchored; None means all eligible history |
| `primary_metric` | `"rmse"` or `"mae"`, default RMSE |
| `reference_model` | None; explicit requested model required for RMSE skill |
| `subset` | `"all_intervals"` including night; optional `"reference_point_daylight"` |
| `on_error` | `"continue"` records failed origins; `"strict"` raises BacktestOriginError |
| `training_issue_times`, `fit_time` | None; both required for either LightGBM recipe |
| `lightgbm` | Existing LightGBMConfig, unchanged algorithm/parameter defaults |
| `t0`, `chronos` | Existing configs, with **local_files_only=True** for this public API |

The only public target policy is the next Brussels calendar day. It generates
92/96/100 quarters across DST; schedules are never inferred from available labels.
Advanced users can keep using BacktestSpec and custom target/availability policies.
The completion cutoff is floor_15min(issue minus safety lag). This explicitly
approximate policy does not establish real Elia publication latency.

LightGBM trains once, using the existing origin-aware dataset builder. Historical
features and NWP are reconstructed at each training issue; labels are joined only
if eligible at fit time. Whole training target grids must finish by that label cutoff.
All training issues must precede fit_time, and fit_time must not follow the first
evaluation issue. Invalid or crossing schedules raise; no automatic splitting,
validation selection, periodic retraining, or tuning occurs. Runtime can be substantial.
See [explicit local-data example](../examples/advanced_benchmark.py).

For TSFMs, install the extra and deliberately prepare the pinned checkpoint cache.
The default public configs never download weights. If acquisition is intended,
explicitly pass `T0Config(local_files_only=False)` as `config.t0` or the equivalent
ChronosConfig. This may perform a large Hub download/cache write at first inference.
See [optional t0 example](../examples/optional_tsfm_benchmark.py), [t0](models/t0.md)
and [Chronos](models/chronos.md). Device/runtime compatibility and pretrained
evaluation contamination remain caller responsibilities. No real inference is run in CI.

Standard Python logging reports model and origin progress. To see it, call
`logging.basicConfig(level=logging.INFO)` before a run.

## Weather: advanced, prepared local GFS only

Use the existing [NWP preparation workflow](data/weather.md) and
[bounded retrieval example](../examples/fetch_weather_vintage.py) separately.
Install weather tooling only if decoding, storing or preparing native subsets.
Prepare operational GFS runs covering **training and evaluation origins**, with
the correct radiation interval bounds, vintage identity, spatial weights and
availability evidence/assumptions. Retain source/cache provenance.

Given `pv_data`, `train_issues`, `test_issues`, `local_runs`, and prepared `weights`:

```python
from solarcast import benchmark, BenchmarkConfig
from solarcast.data.weather import NWPAvailability, WeatherStore

# This six-hour delay is an assumption, NOT certified historical as-of evidence.
weather = WeatherStore(tuple(local_runs), weights, policy=NWPAvailability(mode="assumed"))
config = BenchmarkConfig(
    issue_times=test_issues,
    observation_safety_lag="30min",
    training_issue_times=train_issues,
    fit_time="2025-01-27T08:00Z",  # choose your own valid chronological boundary
)
results = benchmark(
    data=pv_data,
    config=config,
    models=["seasonal_persistence_48h", "lightgbm_history_radiation_cloud"],
    weather=weather,
)
```

This is a preparation-dependent example, not a self-contained beginner download.
Observed mode requires actual availability evidence on each run; retrieval time
does not qualify. Radiation defaults remain strict. Latest eligible **complete**
vintages are selected by the same WeatherStore during training and inference.
No observed-weather, reanalysis, history-only or cross-vintage fallback is introduced.
Missing NWP fails a whole origin. Training failures and missing labels are recorded
in model metadata; no valid training data raises. Evaluation failures remain in
origins/predictions. Training support across history and weather recipes is not
automatically paired here: inspect row counts and origin failures before claiming
an isolated weather information-value comparison. Use the controlled research
workflow when paired training support is required.

## Results, scoring and reproducibility

`BenchmarkResult` is the existing evaluation type, with backward-compatible fields.
The public runner populates the new config/metadata/predictions/origins fields;
evaluator-only research calls leave those new fields empty:

- `leaderboard(primary_metric=None)`: ordinary pandas DataFrame; lower RMSE/MAE wins,
  exact ties share minimum rank, deterministic model-ID ordering, undefined scores
  have undefined rank. Changing this argument is an explicitly requested ranking view.
- `predictions`: **all requested rows**, including failed origins and missing forecasts,
  with evaluator actuals. No raw negatives are clipped.
- `origins`: success/failure, exception information, cutoff/history counts and weather audit.
- `metrics` and `summary`: common-support metric/accounting tables (`metrics` returns a copy).
- `aligned_predictions`: only the exact common finite scoring population, in long form.
- `config`: effective BenchmarkConfig on public runs; None on research evaluator-only runs.
- `metadata`: provenance assertion, canonical numerical PV SHA256, bounds/row counts,
  package/numerical versions, schedules/config snapshot, model recipes/configs and
  training/checkpoint information. Weather includes policy, weights identity, run
  versions/metadata and native numerical hashes. These fingerprints do not certify provenance.
- `grouped(by)`, `backtest_specs`, `primary_metric`, `reference_model`, `subset_name`:
  existing research diagnostics remain available.

The expected population is the union of requested issue/target keys (after an
optional model-independent subset). Metrics and ranks use the intersection where
**every requested model** has a valid forecast and the same finite actual.
Failures shrink everyone's scoring support; read `n_scored`, `n_expected`,
`forecast_availability`, `n_failed_origins` together. No common observations means
NaN metrics/ranks, never fabricated zero errors. There are no independent-population
metrics in the public leaderboard. Availability is valid predictions / expected keys;
it is not the fraction of actual observations present. RMSE skill is
1 minus model RMSE / explicit reference RMSE on the same support.

Daylight uses the existing representative Brussels point apparent elevation >0
at interval start, not fleet-weighted illumination. Daylight changes scoring only;
raw predictions remain complete. Common targets do not establish equal information
sets: history, NWP and pretrained adapters have different declared inputs.

The result is an in-memory object with caller-owned mutable tables. Metadata/config
contain pandas objects and paths; they are inspectable Python values, not a promised
portable JSON manifest or serialized model. Preserve source revisions, input vintages,
environment and externally acquired model artifacts for full reproduction. Save
tables explicitly, e.g. `results.leaderboard().to_csv("leaderboard.csv", index=False)`.
Nothing is automatically written by the public runner; optional weight acquisition
is the explicit exception described above. Existing frozen experiment artifacts
are not loaded or changed by this API.

## Scope and naming decisions

`benchmark`, `available_models`, `BenchmarkConfig` and `BenchmarkResult` describe
the user task directly, are qualified by the `solarcast` namespace, and avoid
colliding with research names by reusing the existing result type. `ModelInfo`
types the small discovery surface; there is no registry/plugin hierarchy.

The aspirational two-argument call omits data, schedule and availability assumptions.
Those inputs are required here to prevent hidden downloads, inference from test
coverage and undocumented scientific choices. No local experiment paths are required.
`plot()` is deferred: use the existing analysis notebook or ordinary pandas/plotting
tools. `forecast()` is planned, not exported: live NWP, model artifacts, lifecycle and
operational availability need a separate design. No fake operational behavior is offered.

The March–October 2025 development results are known and are not a fresh holdout.
This API adds no new performance evidence. Do not tune from those results or infer
deployment readiness; use a fresh confirmatory period. Frozen experiment source
guards deliberately reject changed source trees, including new public exports;
replay old frozen runs with their archived source rather than editing their freezes.
