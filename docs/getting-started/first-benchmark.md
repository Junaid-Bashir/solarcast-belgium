# Your first benchmark

Begin with the complete [quickstart](quickstart.md).

```text
data + forecast origins + models + BenchmarkConfig
                         |
                    benchmark()
                         |
                  BenchmarkResult
```

## Prepare data

Supply unique timezone-aware quarter-hour `target_time` interval starts and numeric `pv_generation_mw` values in MW. Include warm-up history and evaluation actuals; preserve missing observations. `benchmark()` does not fetch data. Declare provenance with `data_source`. See [Belgian PV data](../guide/belgian-pv.md).

## Choose forecast origins

An **issue time** is when a historical prediction is simulated. Eligible **history** satisfies the cutoff; **target times** are the next Belgian day's quarters. Scoring actuals are kept out of prediction context.

Set `observation_safety_lag` explicitly. A 30-minute event-time lag is an approximation, not evidence of original publication time or revision history. Belgian days contain 92, 96 or 100 quarters around DST.

Start with `seasonal_persistence_48h`. LightGBM additionally needs chronological training issues and a fit time before evaluation; TSFMs require runtimes and weights. See [models](../guide/models.md).

## Read results

This continuation runs after the quickstart in the same session:

```python
print(results.leaderboard())
print(results.origins.head())
print(results.predictions.head())
print(results.grouped(by="local_hour"))
```

Rankings use shared finite issue/target support. Inspect availability and failed origins alongside errors. `predictions` retains requested rows including failures; `aligned_predictions` is scoring support. Outputs are unclipped and night is included by default. See [evaluation](../guide/evaluation.md) and [result reference](../reference/result.md).
