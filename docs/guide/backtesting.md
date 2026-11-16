# Backtesting

Rolling-origin evaluation simulates successive historical predictions:

```text
eligible history ---- cutoff ---- issue time ---- next Belgian target day
                         |              |                   |
                  bounded context   predict()         score actuals
```

Each origin exposes only eligible supplied history. Scoring actuals stay outside prediction context. Later origins advance cutoffs and targets. Failures retain target rows so missing forecasts remain visible.

The public API requires an explicit observation lag. Event-time completion plus lag approximates availability; it is not verified as-of evidence. Chronological training finishes before evaluation. Known holdout diagnostics cannot be reused as untouched confirmation.

See the [backtesting contract](../backtesting.md) for schedules, cutoffs, failure policies and context selection, and the [forecasting interface](../forecasting-interface.md) for direct control.
