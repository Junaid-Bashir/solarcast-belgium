"""Offline interface smoke example; the constant output is not a real baseline."""

import pandas as pd

from solarcast.models import Forecast, ForecastContext, Forecaster, ForecastRequest


class FakeForecaster:
    """Demonstrate structural typing without fit or any real prediction logic."""

    model_id = "fake_contract_example"

    def predict(self, context: ForecastContext) -> Forecast:
        return Forecast(
            context.request,
            pd.DataFrame({"target_time": context.request.target_times, "prediction_mw": 1.0}),
            model_id=self.model_id,
        )


def main() -> None:
    """Request -> bounded context -> fake adapter -> validated numerical result."""
    request = ForecastRequest(
        issue_time="2026-06-10T09:07Z",
        target_times=pd.date_range("2026-06-10T22:00Z", periods=4, freq="15min"),
    )
    history = pd.DataFrame(
        {
            "target_time": pd.date_range("2026-06-10T08:00Z", periods=4, freq="15min"),
            "pv_generation_mw": pd.array([1.0, 2.0, None, 4.0], dtype="Float64"),
        }
    )
    context = ForecastContext(request, history, history_cutoff="2026-06-10T09:00Z")
    model: Forecaster = FakeForecaster()
    print(model.predict(context).to_frame().to_string(index=False))


if __name__ == "__main__":
    main()
