# benchmark()

Supply PV data, a configuration and [model identifiers](../guide/models.md). Data must contain unique timezone-aware quarter-hour `target_time` and numeric `pv_generation_mw`. Include warm-up history and evaluation actuals. No observations are fetched automatically.

LightGBM trains once before evaluation; TSFMs can perform expensive inference. Weather requires a prepared local store. Rankings use common finite support and retain separate failure/availability accounting. See the complete executable [quickstart](../getting-started/quickstart.md), explanatory [first benchmark](../getting-started/first-benchmark.md) and [real-data notebook](https://github.com/Junaid-Bashir/solarcast-belgium/blob/main/notebooks/00_real_data_quickstart.ipynb).

::: solarcast.benchmark
