# Published development-benchmark summary

`weather-lightgbm-v1.csv` is a byte-for-byte copy of the original frozen
`extended_leaderboard.csv`. It contains aggregate errors/support/availability only,
not actual PV observations, forecast caches or model weights. Its SHA256 and the
original manifest/freeze identity are recorded in `weather-lightgbm-v1.json`.
All 75 original manifest entries were checked before exporting this summary.

The [one public chart](../assets/development-benchmark.svg) and README table use
these same numerical values. This preparation did not train, tune, regenerate
forecasts or alter the scoring population. There are 245 origins and 23,520 common
quarters per model, March–October 2025, including night. Lower RMSE is better.

This is a **known-period development comparison**, not independent confirmation.
Weather-LightGBM uses forecast radiation/cloud, while the other models use history
only. Publication/revision and pretrained-model limitations remain. Read the
[full report](../experiments/weather-lightgbm-v1.md), [reproduction limits](../reproducibility.md)
and [third-party resource notes](../../THIRD_PARTY.md).

Original raw-input and source/prediction bundles are not distributed. Hashes provide
provenance anchors, not a public artifact download or a guarantee of provider as-of
fidelity. The current public source tree is not the frozen experiment source tree.
