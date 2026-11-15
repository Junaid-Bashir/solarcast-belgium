# Rolling-origin backtesting

`solarcast.backtesting.backtest` simulates an ordered sequence of historical forecast
issues using the existing Forecaster, ForecastRequest, ForecastContext and Forecast
contracts. It returns numerical predictions/actuals and origin status tables. It
does not calculate metrics, train models or download data.

## API

```python
import pandas as pd
from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest
from solarcast.models import SeasonalPersistenceForecaster

issues = pd.date_range("2026-06-08 11:00", periods=3, freq="D", tz="Europe/Brussels")
spec = BacktestSpec(
    issues,
    availability_policy=EventTimeCutoff(safety_lag="30min"),
    context_window="14D",
    on_error="strict",
)
result = backtest(model=SeasonalPersistenceForecaster("7D"), data=pv_data, spec=spec)
predictions = result.predictions
origins = result.origins
```

`pv_data` is a normalized PV snapshot. It must have unique, non-null aware
quarter-hour `target_time` and real numeric nullable `pv_generation_mw`. Naive,
duplicate, misaligned timestamps and infinite PV values are errors. Missing and
negative values are retained. The engine copies, normalizes to UTC and sorts data
internally, preserving caller input. Additional provenance columns can be read
by trusted availability policies but are never automatically passed to models.

Issue times must be an aware, nonempty, unique, ordered DatetimeIndex; they are
canonicalized to UTC. Issues need not fall on quarter-hour boundaries or at a fixed
local hour. Origin order is chronological; there is no shuffling or random splitting.

## Target policy and DST

The default `day_ahead_local_day(issue_time)` returns quarter-hour interval starts
for the next **Europe/Brussels calendar date**, from local midnight inclusive to
the following local midnight exclusive. Each boundary is localized independently
before conversion to UTC; no fixed 24-hour arithmetic determines the target day.

- Normal local day: 96 targets.
- Spring-forward local day: 92 targets.
- Autumn-backward local day: 100 targets.

Repeated autumn clock labels are distinct UTC instants. Target counts are never
fixed by the engine. A different small callable can replace the default:

```python
def next_hour(issue: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(issue.floor("15min") + pd.Timedelta(hours=1), periods=4, freq="15min")
```

Pass it as `target_policy=next_hour`. ForecastRequest rejects invalid, empty,
unordered, duplicate, naive or nonfuture target grids. Invalid target policies
abort even in continue mode: these are configuration errors, not model failures.

## Explicit availability approximation

`availability_policy` is a required spec argument. The supplied policy is named
`event_time_cutoff`, implemented by `EventTimeCutoff`. Its `safety_lag` defaults
visibly to zero when that policy is constructed without an argument. Zero is an
**evaluation assumption**, not evidence of zero Elia publication delay.

For issue `T` and nonnegative elapsed safety lag `L`:

```text
cutoff = floor_to_15_minutes(T - L)
eligible if observation.target_time + 15 minutes <= cutoff
```

The interval beginning at 10:45 represents `[10:45, 11:00)`. At issue 11:00 with
zero lag it is eligible by the completion approximation; the 11:00 interval is not.
With 30-minute lag, the cutoff is 10:30 and the latest eligible start is 10:15.
For issue 11:07 and lag 17 minutes, floor(10:50) gives 10:45 and latest start 10:30.

Flooring permits existing quarter-hour feature/context contracts to remain unchanged
while arbitrary issue times and safety lags are supported. It never rounds forward.
No `available_at` is fabricated, and retrieval timestamps are not used as publication
times. Exact Elia latency and revisions remain unresolved. A safety lag does not
by itself make a retrospective snapshot an operationally correct vintage archive.

## Context window

`context_window=None` means all policy-eligible history. A positive elapsed duration,
such as `"7D"`, applies an additional **issue-anchored** lower bound:

```text
observation.target_time >= issue_time - context_window
observation.target_time + 15 minutes <= cutoff
```

The lower bound is inclusive. Safety lag shortens the usable span at the upper
end; the engine does not shift the lower bound backward to compensate. For issue
11:00, one-hour window and 30-minute lag, starts 10:00 and 10:15 are eligible.
Windows are elapsed UTC durations, not local calendar dates. Empty, incomplete
or gapped history remains possible; each model decides whether it can forecast.

## Information flow and policy substitution

```text
normalized snapshot -> trusted availability selection -> context-window bound
                    -> ForecastContext -> model.predict(context) -> Forecast

normalized snapshot -> evaluator-side exact actual lookup -> result table only
```

Context construction independently enforces cutoff <= issue and completed intervals.
It projects only historical target/PV columns and defensively copies them. No
future labels, full dataset, capacity, Elia forecasts or evaluator object reaches
the model. Features are not automatically generated: `context.features` is None.
Adapters can derive features from supplied history and target times; an eventual
feature-builder hook must obey the same bounds.

A policy structurally satisfies `AvailabilityPolicy` by exposing `name` and
`select(data, issue_time) -> HistorySelection(history, cutoff)`. It receives a copy
of the prepared snapshot and can filter using genuine supplied publication metadata
later. The cutoff/context constraints still apply. Policies are trusted scientific
code, not model inputs; arbitrary policy/model closures or external data access
cannot be sandboxed by this engine.

The current input contract requires one row per target time. Historical revisions
will require upstream as-of snapshot selection or a future explicitly designed
version-aware ingestion path; the current snapshot schema does not reconstruct lost vintages.

Returned Forecast issue, target grid, model ID and PV variable are checked against
the current origin. The engine uses the same inference call for every adapter,
with no model-name/type branches. Actual values are looked up only after that call.

## Result tables and missing data

`BacktestResult` holds `predictions`, `origins`, the spec and model ID. Tables are
owned mutable output tables, not views into caller data. No wall-clock execution
timestamps or random values are introduced. Run configuration is retained in the
spec; no serialization/fingerprint framework is added.

`predictions` has one row per requested target **including failed origins**:

| Column | Semantics |
|---|---|
| `issue_time`, `target_time` | UTC `datetime64[ns, UTC]` |
| `model_id`, `model_version` | String identity; version nullable |
| `prediction_mw`, `actual_mw` | Nullable `Float64` MW |
| `horizon` | Proper elapsed target-minus-issue timedelta |
| `status` | `ok` or `forecast_failed` |
| `is_scorable` | True only if prediction and actual are both non-missing |

The composite observation key is issue/model/target. The same target can appear
under different issues; rows are not overwritten or globally deduplicated.

Absent actual timestamps and explicit null actuals stay missing; neither is converted
to zero or interpolated. `is_scorable` is only an eligibility flag for future metrics,
not a metric or permission to ignore failures when reporting coverage.

`origins` records issue/model, validated `history_cutoff`, history/target counts,
status, exception type and exception message. If context validation failed before a
cutoff could be established, the recorded cutoff is missing, not invented.

## Failure policy

- **`on_error="strict"` (default):** wrap an origin failure in BacktestOriginError
  with model and issue context and preserve the original exception as its cause.
  The run stops; no partial result is returned.
- **`on_error="continue"`:** log and record the origin exception, retain all its
  target rows with missing predictions and `is_scorable=False`, then attempt later
  issues. Actuals can still be present. There is no hidden fill or fallback model.

Continue mode covers availability/context/model/result-validation failures; it
does not hide invalid source data, schedules or target-policy configuration.
KeyboardInterrupt/SystemExit are not caught. Summaries log origin counts, not
individual quarter-hour rows. No progress or parallelization dependency is added.

Exact 24h seasonal persistence can fail for tomorrow afternoon at an earlier issue
time. This remains visible; use an explicitly selected longer lag when appropriate.

## Determinism, training and limitations

The engine runs sequentially and reuses the supplied model instance. Stateless
baselines are deterministic. Other adapters must manage declared state correctly;
the engine cannot erase pretrained knowledge or hidden caller-supplied state.
Use equivalently prepared fresh models for independent comparisons. No automatic
fit, reset, retraining or weight loading is performed.

The [LightGBM adapter](models/lightgbm.md) now supports explicit origin-aware
training before evaluation, guarded by a fit-time boundary. Periodic retraining,
transformer fitting and broader state-isolation policies remain future work.
Training on all data before this loop would leak. Shared context selection is
reused by training construction without a LightGBM-specific backtest branch.

Tests include a `999999` future-value canary recorded by a fake forecaster, a check
that those same values appear only as evaluator actuals, advancing context windows,
safety-lag boundaries, missing actuals, strict/continue/all-failed origins, custom
availability selection, invalid model results, baseline integration, deterministic
ordering and input immutability. All are offline.

Run `python examples/backtest_baseline.py` for normal/spring/autumn smoke runs,
each with three consecutive origins. [Deterministic evaluation](evaluation.md) now
consumes these results. Trainable retraining, multi-model run
orchestration and performance optimization remain deferred.
This loop provides temporal isolation under a stated approximation; it does not
claim fully verified historical operational realism.

Optional `backtest(..., weather_provider=WeatherStore(...))` selects the newest
eligible complete local NWP vintage for each request before prediction. The
resulting numerical `context.weather` validates exact target/issue agreement;
weather-enabled origin rows retain run/availability/policy/weights provenance.
Coverage failures obey existing strict/continue handling. No provider preserves
the history-only context and result schema. Models never query archives. See the
[weather contract](data/weather.md#22-historical-gfs-ingestion) for the explicit
six-hour availability approximation, strict observed mode and mandatory canary.
