# Quickstart

Install with `python -m pip install solarcast-belgium`, then run this complete offline example:

```python
import pandas as pd
from solarcast import BenchmarkConfig, benchmark

pv = pd.DataFrame(
    {
        "target_time": pd.date_range("2025-01-01", "2025-01-12", freq="15min", tz="UTC"),
        "pv_generation_mw": 100.0,
    }
)
config = BenchmarkConfig(
    issue_times=pd.date_range("2025-01-09 11:00", periods=2, freq="D", tz="Europe/Brussels"),
    observation_safety_lag="30min",  # explicit availability approximation
)
results = benchmark(data=pv, config=config, models=["seasonal_persistence_48h"])
print(results.leaderboard().to_string(index=False))
```

Expect one row, 192 scored targets, zero RMSE and full availability. Constant synthetic data only checks installation; these are not Belgian forecasting results.

Continue with [your first benchmark](first-benchmark.md) or the [real-data notebook](https://github.com/Junaid-Bashir/solarcast-belgium/blob/main/notebooks/00_real_data_quickstart.ipynb).
