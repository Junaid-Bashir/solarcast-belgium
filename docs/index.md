# SolarCast Belgium

Benchmark energy forecasting models under common historical conditions, starting with Belgian aggregated solar PV.

**Same data. Same horizon. Same backtest. Let the results decide.**

```python
from solarcast import available_models

for model in available_models():
    print(model.model_id, model.status)
```

## What SolarCast does

Supply a PV snapshot, historical forecast dates and model identifiers. SolarCast runs rolling-origin forecasts and returns predictions, failure accounting and a comparable leaderboard. It is a Python package, not an operational forecasting service.

| Approach | Information | Preparation |
| --- | --- | --- |
| Persistence | PV history | Core installation |
| LightGBM | PV history and deterministic features | Optional runtime and chronological training |
| Weather-LightGBM | History plus forecast radiation/cloud | Training and prepared historical NWP |
| t0 and Chronos | PV history, zero-shot | Optional runtimes and pretrained weights |

## Why benchmarking matters

Accuracy depends on both the model and its information at issue time. SolarCast compares common targets, keeps missing forecasts visible and discloses availability assumptions. The published study is a **known-period development benchmark**, not operational validation.

## Start here

- [Installation](getting-started/installation.md)
- [Five-minute quickstart](getting-started/quickstart.md)
- [API cheat sheet](getting-started/cheat-sheet.md)
- [Supported models](guide/models.md)
- [Development benchmark](research/development-benchmark.md)

These unversioned docs track current main; the public package is 0.1.0. Read the [limitations](research/limitations.md) before interpreting results.

## Development experiments

Read the [development benchmark](research/development-benchmark.md) and linked reports for frozen configurations, information sets and known-period caveats.
