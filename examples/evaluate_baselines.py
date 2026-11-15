"""Synthetic linear-ramp fixture benchmark; not evidence about Belgian PV accuracy."""

import pandas as pd

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest
from solarcast.evaluation import compare_models
from solarcast.models import (
    Forecaster,
    LastValuePersistenceForecaster,
    SeasonalPersistenceForecaster,
)


def main() -> None:
    """Compare three baseline configurations over exactly the same offline origins."""
    times = pd.date_range("2026-06-01T00:00Z", "2026-06-13T00:00Z", freq="15min")
    synthetic = pd.DataFrame(
        {"target_time": times, "pv_generation_mw": pd.array(range(len(times)), dtype="Float64")}
    )
    spec = BacktestSpec(
        pd.date_range("2026-06-09T09:00Z", periods=3, freq="D"),
        availability_policy=EventTimeCutoff("0min"),
        context_window="14D",
    )
    # 24h can be unavailable for day-ahead targets; use an explicit 48h profile,
    # rather than filling its missing future inputs or silently changing its lag.
    models: list[Forecaster] = [
        LastValuePersistenceForecaster(),
        SeasonalPersistenceForecaster("48h"),
        SeasonalPersistenceForecaster("7D"),
    ]
    results = [backtest(model=model, data=synthetic, spec=spec) for model in models]
    benchmark = compare_models(
        results, reference_model="seasonal_persistence_7d", primary_metric="rmse"
    )
    print("SYNTHETIC linear-ramp test data; all intervals; common support; rank by RMSE.")
    print(benchmark.leaderboard().to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("Example only: this ramp is not a realistic solar profile or an Elia benchmark.")


if __name__ == "__main__":
    main()
