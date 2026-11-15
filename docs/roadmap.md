# Roadmap

## Current capabilities

SolarCast benchmarks day-ahead Belgian aggregated PV using normalized Elia data,
timezone-aware contracts, deterministic features, persistence, chronological
LightGBM training and optional pinned t0/Chronos zero-shot inference. Historical
operational GFS ingestion and vintage selection support weather-informed LightGBM.
The public benchmark API wraps shared rolling-origin backtesting, raw-output
common-support evaluation and failure accounting.

The 2025 development results are known. See [experiments](README.md#development-experiments)
and [reproducibility](reproducibility.md) for evidence and limitations.

## Planned research and engineering

- Fresh confirmatory evaluation with stronger historical publication/revision evidence.
- Out-of-sample investigation of complementary model errors before ensemble fitting.
- Probabilistic forecasts, calibration and interval evaluation.
- Additional operational weather sources with verifiable vintage/availability semantics.
- Additional energy targets and European datasets where suitable source contracts exist.
- Broader environment validation and reproducibility tooling.
- Operational forecasting only after model lifecycle, current inputs and availability
  requirements are designed explicitly.

These are contribution directions, not implemented features, promised dates or
preselected scientific outcomes. Keep additions small and compare candidates under
disclosed information sets. Follow [contributing](../CONTRIBUTING.md) and the
[release process](release/README.md).
