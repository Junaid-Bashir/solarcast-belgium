# BenchmarkResult

Returned by `benchmark()`; normally do not construct this yourself. Use `leaderboard()`, `grouped()`, `metrics`, `predictions`, `aligned_predictions`, `origins` and `metadata`.

`predictions` preserves failures; `aligned_predictions` contains scoring support. Leaderboards include rank, identifier, scoring policy, MAE/RMSE/bias in MW, explicit-reference skill, scored/expected counts, availability and failed origins. Lower MAE/RMSE is better; inspect coverage before trusting ranks. See [evaluation](../guide/evaluation.md).

::: solarcast.BenchmarkResult
