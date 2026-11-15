# SolarCast documentation

## Getting started

- [Project overview and offline quickstart](../README.md)
- [Public benchmark API](public-api.md): inputs, configuration, discovery and results
- [Example catalog](../examples/README.md): prerequisites, network and output behavior
- [Reproducibility](reproducibility.md): new runs versus original frozen research runs

## Data and time

- [Data sources and local storage](data/README.md)
- [Elia national PV contract](data/elia.md): schemas, units, DST and availability limits
- [Historical operational GFS/NWP](data/weather.md): preparation and vintage selection
- [Features](features.md): deterministic and cutoff-limited historical inputs

## Forecasting and evaluation

- [Forecasting interface](forecasting-interface.md): request, bounded context and result
- [Persistence baselines](baselines.md): elapsed lags and missing-history behavior
- [Rolling-origin backtesting](backtesting.md): schedules and availability policies
- [Evaluation](evaluation.md): common support, failures, ranking and skill

## Models

- [LightGBM](models/lightgbm.md): pre-test fitting and weather feature recipes
- [t0-gamma](models/t0.md) and [Chronos-2](models/chronos.md): optional pinned zero-shot adapters
- [TSFM landscape](models/tsfm-landscape.md): dated investigation, not a support list

## Development experiments

These reports describe known development periods, not independent confirmation.

- [First real Elia LightGBM/baseline comparison](experiments/lightgbm-baseline-v1.md)
- [Controlled zero-shot TSFM comparison](experiments/tsfm-zero-shot-v1.md)
- [Weather-informed LightGBM comparison](experiments/weather-lightgbm-v1.md)
- [Verified aggregate benchmark snapshot](benchmarks/README.md)
- [Interactive real-data notebook](../notebooks/README.md): requires local artifact bundles

## Contributing and maintenance

- [Architecture](architecture.md) and [roadmap](roadmap.md)
- [Contributing](../CONTRIBUTING.md)
- [Release process](release/README.md)
- [Security reporting](../SECURITY.md), [third-party resources](../THIRD_PARTY.md)
- [License](../LICENSE), [citation](../CITATION.cff), [changelog](../CHANGELOG.md)
