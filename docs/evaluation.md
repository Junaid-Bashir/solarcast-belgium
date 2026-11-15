# Deterministic evaluation and benchmark comparison

Evaluation consumes existing rolling-backtest results, not models. It scores raw
MW predictions without clipping, filling or interpolating. The default population
includes **all intervals, including nighttime**. Metrics do not establish whether
the underlying observation-availability assumptions were operationally correct.

## Numerical metrics

```python
from solarcast.evaluation import mae, rmse, bias, rmse_skill
```

Given complete finite pairs `actual[i]`, `prediction[i]`, let
`error[i] = prediction[i] - actual[i]`:

- **MAE (MW):** `mean(abs(error))`.
- **RMSE (MW):** `sqrt(mean(error**2))`.
- **Bias (MW):** `mean(error)`; positive means overforecasting, negative means
  underforecasting.
- **RMSE skill (dimensionless):** `1 - RMSE(model) / RMSE(reference)`, calculated
  from the same actual/prediction pairs for both models. Positive beats the
  reference, zero matches it, negative is worse.

Low-level functions require real, finite, nonempty one-dimensional vectors of
equal length. They raise on missing/non-finite input instead of choosing a mask
independently. The evaluation layer below chooses masks and reports exclusions.
Scaled numerical calculations avoid unnecessary squaring/summation overflow.

If the reference error is zero, skill is mathematically undefined, including when
both forecasts are perfect. Return NaN, with summary `skill_status=zero_reference_error`.
No epsilon, infinity, fabricated zero or automatic baseline substitution is used.

**nRMSE is deferred:** no normalization denominator has been agreed. Mean, range,
maximum production and monitored capacity have different interpretations; capacity
availability is also unresolved. Do not report an ambiguous normalized score.
**MAPE is omitted** because zero/near-zero PV makes it unsuitable as a primary score.

## Input contract and scorable rows

`evaluate_forecast(result)` accepts one BacktestResult or its prediction DataFrame.
`compare_models(results)` accepts a sequence of those same existing objects/tables.
No new per-observation format is required. Required columns are `issue_time`,
`target_time`, `model_id`, `prediction_mw`, `actual_mw`, `horizon`, and `status`.

Each input must contain one unambiguous model/run. Duplicate model IDs across inputs
or mixed non-null versions within a run raise; use distinct IDs for configurations.
Duplicate issue/target keys, naive/null/invalid timestamps, nonfuture or misaligned
targets, invalid statuses and inconsistent horizons raise rather than silently
dropping structural errors. Unsorted rows are copied and sorted. Inputs are not
mutated; normal UTC normalization is applied.

For structurally valid rows, a candidate scorable pair requires:

- finite, present numeric actual;
- finite, present numeric prediction;
- `status == "ok"`.

Null/NaN/infinity are excluded from scoring, never changed into zero. Failed-origin
rows cannot score even if a manually edited table contains a number. Negative
predictions and actuals remain unchanged. Numeric strings, booleans and complex
values are invalid types, not silently converted.

`is_scorable` from input is recomputed because output tables can have been edited
or exported. The returned aligned rows carry that recomputed finite/status mask.
Model-independent subset filtering and common support further select scored pairs.

## Common-support comparison

Alignment keys are **`(issue_time, target_time)`**, not target time alone: different
issues imply different information and forecast horizons. Forecasts for the same
target from two issues are separate observations.

The comparison population is the **union** of supplied requested keys, optionally
restricted by one explicit subset. The scored support is the intersection of keys
with valid pairs for **every compared model**. Every aggregate, grouped score and
reference skill uses this same support. Models cannot improve their ranking by
being scored independently on easier timestamps.

Example: A supplies keys 1–4 and B supplies 2–5. Expected population is 1–5; both
are scored on 2–4. Each has one absent requested row relative to the union and one
otherwise-valid row excluded by alignment. Forecast availability is 4/5 for each.
If one model has no valid common predictions, metrics and ranks are undefined
(NaN / nullable rank), while failure and availability counts remain visible.

Shared keys must have identical finite actuals and agree on actual validity.
Conflicting actual values, or a finite actual in one run versus an unavailable
actual in another, raise. This avoids comparing forecasts against different truth
snapshots. Non-finite actuals are unscorable regardless of their numeric spelling.

Union size is inferred from supplied rows. An observation omitted from **every**
input cannot be reconstructed from a DataFrame. Preserve the backtester's failed
target rows; do not discard origins before evaluation.

Matching keys and truth do not certify identical model information access, training,
weather vintages or resources. Inspect retained `backtest_specs` and experimental
provenance; compare compatible information sets in scientifically justified tracks.

## API and leaderboard

```python
from solarcast.evaluation import compare_models

benchmark = compare_models(
    results=[last_value_result, seasonal_48h_result, seasonal_7d_result],
    primary_metric="rmse",
    reference_model="seasonal_persistence_7d",
)
leaderboard = benchmark.leaderboard()
summary = benchmark.summary
scored_rows = benchmark.aligned_predictions
```

Default ranking uses **lower RMSE**, emphasizing larger errors while retaining MW
interpretability. `primary_metric="mae"` is supported; signed bias is reported but
not used as a lower-is-better ranking metric. Leaderboard columns explicitly include
the ranking metric, subset and reference model. Exact ties share competition rank
(1, 1, 3), with alphabetical model ID determining display order. Unscored models
have no rank and appear last. Input order does not determine ties or references.

No reference is selected by default: skill stays NaN with `not_requested`. An
explicit reference absent from inputs raises. A group with no common data reports
`no_common_observations` instead of a skill. The named reference is not automatically
the worst or best model; choosing it is part of the experiment.

The leaderboard is an ordinary DataFrame and can use pandas `to_csv` when desired.
No plotting or reporting dependency, final top-level API or file export manager
is added. Output tables are owned mutable results; treat the benchmark object as
a snapshot and do not edit its summary before calling its methods.

## Counts, availability and evaluation periods

`benchmark.summary` exposes per-model metrics and these diagnostics within the
selected population:

| Field | Meaning |
|---|---|
| `n_total`, `n_expected` | Supplied own rows; union of requested issue/target keys |
| `n_missing_rows` | Expected keys absent from this model's supplied table |
| `n_actual_missing`, `n_actual_nonfinite` | Null actuals; non-null infinite actuals |
| `n_prediction_missing`, `n_prediction_invalid` | Null predictions; non-null predictions invalid due to infinity or failed status |
| `n_valid_pairs` | Own finite/status-valid pairs before common-support alignment |
| `n_excluded_alignment` | Otherwise-valid own pairs removed by common support |
| `n_scored` | Identical common-support count used by all compared models |
| `n_origins`, `n_scored_origins`, `n_failed_origins` | Own supplied, scored, and explicitly failed origin counts |
| `forecast_availability` | Finite/status-valid predictions divided by union expected keys |
| `first_target`, `last_target` | Own supplied target period |
| `first_scored_target`, `last_scored_target` | Actual scored target period |

Missing-actual and missing-prediction diagnostics can overlap; they are not exclusive
reasons whose sum must equal total exclusions. NaN is treated as missing, infinity
as non-finite. Missing rows are counted separately. Forecast availability counts
valid forecasts even when actuals are missing; it is **not** probabilistic interval
coverage or the proportion scored. An empty selected population gives NaN availability.

All counts refer to forecast observations (issue/target pairs), not deduplicated
target instants across multiple origins. Do not suppress failed runs in reporting.

## Grouped analysis

```python
by_horizon = benchmark.grouped("horizon")
by_hour = benchmark.grouped("local_hour")
by_month = benchmark.grouped("month")
by_season = benchmark.grouped("season")
```

These return numerical summaries with the same common support and diagnostics:

- Horizon: exact elapsed Timedelta, with no hidden bins.
- Local hour: target hour in Europe/Brussels, not UTC.
- Month: Brussels **year-month** (`YYYY-MM`) so years do not merge silently.
- Season: meteorological DJF (Dec–Feb), MAM (Mar–May), JJA (Jun–Aug), SON (Sep–Nov),
  pooled across years as explicit seasonal categories.

DST does not change key semantics or impose 96 rows. Repeated autumn clock hours
can contribute eight quarter-hours to the local-hour=2 group while remaining eight
distinct UTC observations. Grouped evaluation does not rerun models or reselect a
different model-specific scoring mask.

## Nighttime and explicit subsets

The default includes all intervals. To examine daylight separately, supply a
model-independent boolean subset and an explicit name. The callback receives
only unique issue/target keys and horizon, not model predictions or errors:

```python
import pandas as pd
from solarcast.features import solar_features


def daylight(keys: pd.DataFrame) -> pd.Series:
    flags = solar_features(pd.DatetimeIndex(keys.target_time.unique()))
    flags = flags.set_index("target_time").is_daylight
    return keys.target_time.map(flags).astype(bool)


daylight_benchmark = compare_models(
    results,
    subset=daylight,
    subset_name="reference_point_daylight",
    reference_model="seasonal_persistence_7d",
)
```

The mask must be non-null boolean with the same index. It is applied once before
alignment, equally to all models. This reuses the existing representative-point
apparent-elevation > 0° definition and its national approximation; it is not a new
daylight model. Daylight-only selection must be labelled and reported alongside
all-interval results when appropriate, not substituted silently.

## Example, assumptions and limitations

Run `python examples/evaluate_baselines.py`. It backtests last-value, 48h and 7D
persistence over three origins and compares 288 common observations. Inputs are a
synthetic linear ramp, **not realistic PV or Elia data**; the ranking only checks
the software pipeline. The 48h choice is explicit because fixed 24h persistence
cannot cover all day-ahead targets under the issue cutoff. No fallback is hidden.

Inputs assume MW, interval-start timestamps and compatible backtest conditions.
Error metrics weight each selected quarter-hour forecast observation equally;
they do not integrate energy or give every local day equal weight across DST.
Training/selection leakage and actual revisions are not repaired by scoring.
Historical Elia publication availability remains approximated. No probabilistic
evaluation, significance tests/bootstrap, advanced models, postprocessing or plots
are implemented. Record experiment provenance and preserve a held-out test period.
