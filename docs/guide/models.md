# Forecasting models

These are exact SolarCast identifiers; checkpoint names are documented on adapter pages.

| Identifier | Type | Local training | Weather | Requirements |
| --- | --- | --- | --- | --- |
| `last_value_persistence` | Last observation | No | No | Core |
| `seasonal_persistence_48h` | Exact elapsed 48-hour lag | No | No | Core |
| `seasonal_persistence_7d` | Exact elapsed 7-day lag | No | No | Core |
| `lightgbm` | History-based ML | Yes | No | `lightgbm` extra |
| `lightgbm_history_radiation_cloud` | 19 history/NWP features | Yes | Yes | `lightgbm`, prepared WeatherStore; `weather` for preparation |
| `t0` | Zero-shot t0-gamma | No | No | `t0` extra and weights |
| `chronos` | Zero-shot Chronos-2 | No | No | `chronos` extra and weights |

```python
from solarcast import available_models

for model in available_models():
    print(model.model_id, model.status, model.missing_dependencies)
```

`built_in` needs no optional runtime; `installed` means dependency distributions were found; `unavailable` lists missing distributions. Installed does not guarantee binary/device compatibility or cached weights.

Read [persistence](../baselines.md), [LightGBM](../models/lightgbm.md), [Weather-LightGBM](../models/weather-lightgbm.md), [t0](../models/t0.md) and [Chronos](../models/chronos.md). No automatic winner selection, tuning, ensemble or periodic retraining occurs. Public benchmark TSFM defaults prohibit downloads; explicit acquisition can download large pinned weights. Native P50 stays unclipped.
