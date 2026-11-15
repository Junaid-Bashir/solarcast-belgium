# Persistence baselines

Baselines provide transparent reference forecasts before more complex models are
compared. No performance advantage or operational realism has been established;
comparisons use the shared rolling evaluator. These models are stateless: neither
requires nor exposes a dummy `fit()` method. They use the existing
[Forecaster contract](forecasting-interface.md), unchanged.

## Last-value persistence

`LastValuePersistenceForecaster` selects the row with the latest UTC target time
among **non-null** PV observations in `context.history`, then repeats that MW value
for every requested target. Context validation already requires finite-or-null
PV and completed intervals at/before the explicit cutoff.

```text
forecast(t) = latest eligible non-null historical PV value
```

Trailing nulls are explicitly skipped. If there is no valid observation, raise
`ForecastDataError`. Zero and negative values are valid raw values. No staleness
threshold is imposed: a very old latest valid observation is still used.

This is mainly a sanity check. A constant value cannot reproduce the changing
daytime profile and may repeat positive PV overnight or zero PV through daytime.
Those weaknesses should remain visible to future evaluation rather than being
hidden by physical corrections.

## Seasonal persistence

`SeasonalPersistenceForecaster(lag="24h")` uses an exact elapsed-time lookup:

```text
forecast(t) = observed_PV(t - lag), using supplied context only
```

The default is 24 elapsed hours; `lag="7D"` means 168 elapsed hours. Other positive
whole-quarter-hour timedeltas are supported, including `"48h"` and Python
`datetime.timedelta` / pandas Timedelta values. Reject zero/negative, missing,
invalid, or non-quarter-hour lags. No nearest lookup, row shift, recursive use of
new predictions, interpolation or fallback to another lag is performed.

Equivalent durations have stable identities: `"1D"` and `"24h"` both produce
`seasonal_persistence_24h`; `"7D"` and `"168h"` both produce
`seasonal_persistence_7d`. Whole-week multiples use days, other whole hours use
hours, and remaining quarter-hour durations use total minutes (for example 90m).
Last-value identity is `last_value_persistence`.

These lookup semantics match `pv_lag_24h_mw` and `pv_lag_7d_mw` from the feature
layer. Features may remain missing; a complete Forecast cannot, so the baseline
raises for unavailable matches rather than converting missing features to values.

## Eligibility and missing history

Models receive no full source DataFrame or downloader, only the validated
`ForecastContext`. They use its copied history and ignore optional feature tables.
No solar geometry, Elia forecasts, weather or monitored capacity enters prediction.
All selected source intervals must already be complete by `history_cutoff`, which
must not exceed the request issue. No availability timestamp is fabricated.

For seasonal persistence, any absent or null exact match raises `ForecastDataError`
for the **whole request**; there is no partial result or silent fill. The error gives
the model, missing count, up to five required historical UTC timestamps and the
cutoff/issue. Larger missing sets are marked as truncated. This follows the strict
finite/non-missing Forecast contract.

Important day-ahead limitation: a forecast issued at 11:00 today cannot use today's
afternoon observations as tomorrow afternoon's 24h-lag values. Even if the full
retrospective source contains them, they must not be in context. Exact 24h
persistence can therefore fail for part of a day-ahead target day. Pick an explicit
longer-lag model such as 48h or 7D when its entire profile is eligible; do not
silently cycle backward to an earlier day. A future backtester must report failures
and coverage along with accuracy, rather than scoring only convenient successes.

## UTC, DST and local-day alternative

Targets retain the ordered UTC keys from the request; all lags are elapsed UTC
durations. Models accept any valid target count, including 92/96/100 intervals for
Belgian local days. Repeated autumn clock times remain distinct UTC instants.

For example, 12:00 Brussels on 29 March 2026 is 10:00 UTC. Its 24h elapsed match is
10:00 UTC on 28 March, or 11:00 local then, not the previous day's 12:00 local.
The corresponding autumn comparison shifts in the opposite direction. No fixed
96-row shift is used, and gaps do not displace later matches.

A previous-local-calendar-day model may be useful for a separate comparison,
but it requires explicit choices for repeated/nonexistent source clock times and
day-ahead eligibility. It is deferred to avoid an implicit or fragile DST policy.
Elapsed semantics are a reproducible initial choice, not a claim of superior accuracy.

## Usage

```python
import pandas as pd
from solarcast.models import (
    ForecastContext,
    ForecastRequest,
    LastValuePersistenceForecaster,
    SeasonalPersistenceForecaster,
)

request = ForecastRequest(
    issue_time="2026-06-10T09:00Z",
    target_times=pd.date_range("2026-06-10T10:00Z", periods=4, freq="15min"),
)
# history is an already-filtered normalized PV table.
context = ForecastContext(request, history, history_cutoff="2026-06-10T09:00Z")
model = SeasonalPersistenceForecaster(lag="7D")
forecast = model.predict(context)
table = forecast.to_frame()
```

Both models return the standard Forecast with the original request, issue time,
target order, model ID, variable and MW unit. Raw negative predictions are not
clipped, and nighttime predictions are not forced to zero. Inputs are not mutated.
The baselines satisfy the shared inference contract without a training method.

Run `python examples/persistence_baselines.py` for a small synthetic offline example
of last-value, 24h and 7D models using the same context and typed inference call.
The synthetic values illustrate the interface, not realistic PV production.

## Limitations and next work

Seasonal profiles do not model changing weather, capacity growth or evolving solar
geometry; last-value persistence does not represent a daily solar cycle. Historical
Elia publication latency and revisions remain unresolved: completed intervals alone
are not proof of issue-time availability. Caller-side context selection and a future
defensible availability policy remain necessary. Capacity normalization and solar
postprocessing are separate future experiments, not hidden corrections here.

Offline tests cover exact lookup, feature agreement, nulls/gaps, no valid history,
cutoff isolation, DST, arbitrary horizons, metadata, lag configuration and input
immutability. No new dependencies, network access, model training or benchmark
metrics are required. Next: a rolling-origin backtester with explicit information
cutoffs and leakage protection is now available in [backtesting](backtesting.md).
See [evaluation](evaluation.md) for deterministic metrics and fair benchmark tables.
