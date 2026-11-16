# Evaluation

| Quantity | Meaning |
| --- | --- |
| MAE | Mean absolute error in MW; lower is better |
| RMSE | Square root of mean squared error in MW; emphasizes large misses |
| Bias | Mean prediction minus actual in MW; positive means overprediction |
| RMSE skill | `1 - RMSE(model) / RMSE(reference)`; explicit reference required |
| Forecast availability | Coverage of expected targets, reported alongside failures |

Rankings use common finite issue/target support across compared models. A model can fail outside that support: inspect `n_expected`, `n_scored`, `forecast_availability` and `n_failed_origins` too.

Defaults include night and raw outputs without clipping. There is no ambiguous normalized RMSE. Daylight filtering requires an explicit reference-point policy. A zero-error reference cannot produce meaningful finite RMSE skill.

See the [evaluation contract](../evaluation.md) for exact support, formulas and accounting, and the [result reference](../reference/result.md) for tables.
