"""Small offline baseline example; synthetic values are not Belgian observations."""

import pandas as pd

from solarcast.models import (
    ForecastContext,
    Forecaster,
    ForecastRequest,
    LastValuePersistenceForecaster,
    SeasonalPersistenceForecaster,
)


def main() -> None:
    """Use the same context and inference call for three stateless configurations."""
    targets = pd.date_range("2026-06-10T10:00Z", periods=4, freq="15min")
    times = pd.date_range("2026-06-03T00:00Z", "2026-06-10T09:00Z", freq="15min", inclusive="left")
    history = pd.DataFrame(
        {"target_time": times, "pv_generation_mw": pd.array(range(len(times)), dtype="Float64")}
    )
    request = ForecastRequest("2026-06-10T09:00Z", targets)
    context = ForecastContext(request, history, history_cutoff="2026-06-10T09:00Z")
    models: list[Forecaster] = [
        LastValuePersistenceForecaster(),
        SeasonalPersistenceForecaster("24h"),
        SeasonalPersistenceForecaster("7D"),
    ]
    for model in models:
        print(model.predict(context).to_frame().to_string(index=False))


if __name__ == "__main__":
    main()
