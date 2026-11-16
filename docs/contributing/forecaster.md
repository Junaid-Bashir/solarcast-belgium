# Adding a forecaster

The structural `Forecaster` protocol requires a stable `model_id` and `predict(context) -> Forecast`. No inheritance or dummy fit method is required. This complete offline example exercises the real contract:

```python
import pandas as pd
from solarcast.models import Forecast, ForecastContext, ForecastRequest


class ConstantForecaster:
    model_id = "example_constant"

    def predict(self, context: ForecastContext) -> Forecast:
        return Forecast(
            context.request,
            pd.DataFrame(
                {
                    "target_time": context.request.target_times,
                    "prediction_mw": 100.0,
                }
            ),
            model_id=self.model_id,
        )


request = ForecastRequest(
    issue_time="2025-01-02T11:00:00Z",
    target_times=pd.date_range("2025-01-03", periods=4, freq="15min", tz="UTC"),
)
history = pd.DataFrame(
    {
        "target_time": pd.date_range("2025-01-02", periods=4, freq="15min", tz="UTC"),
        "pv_generation_mw": 100.0,
    }
)
context = ForecastContext(request, history, history_cutoff="2025-01-02T10:30:00Z")
forecast = ConstantForecaster().predict(context)
print(forecast.predictions)
```

This demonstrates an interface, not a useful PV model. Custom objects can use the lower-level [backtesting engine](../backtesting.md). Public `benchmark()` accepts implemented string identifiers; there is no dynamic plugin registration.

Use only bounded supplied context and declared model state. Never fetch hidden observations or expose scoring actuals. Predictions must match every requested target, in order, in finite MW; fail explicitly when unable to forecast. Training labels must be eligible at fit time.

Test cutoffs, target equality, missing inputs and future-data sentinels. Keep heavyweight imports behind optional boundaries and test using offline fakes. Read the [forecasting contract](../forecasting-interface.md).
