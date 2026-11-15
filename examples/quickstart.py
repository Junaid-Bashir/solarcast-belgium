"""Canonical offline public API quickstart; synthetic data, no performance claim."""

import pandas as pd

from solarcast import BenchmarkConfig, benchmark


def main() -> None:
    """Run the same bounded fixture shown in README."""
    # Synthetic installation check, not Belgian PV or accuracy evidence.
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


if __name__ == "__main__":
    main()
