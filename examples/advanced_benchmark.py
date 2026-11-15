"""Explicit local-data LightGBM fit/evaluation. Requires the lightgbm extra."""

import argparse
from pathlib import Path

import pandas as pd

from solarcast import BenchmarkConfig, benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv", type=Path, help="Normalized national PV with aware target_time"
    )
    args = parser.parse_args()
    data = pd.read_csv(args.input_csv)
    # Reject naive strings before UTC conversion, which would otherwise assume UTC.
    stamps = [pd.Timestamp(value) for value in data.target_time]
    if any(pd.isna(value) or value.tzinfo is None for value in stamps):
        parser.error("Every target_time must carry an explicit offset/timezone")
    data["target_time"] = pd.to_datetime(stamps, utc=True)
    config = BenchmarkConfig(
        issue_times=pd.date_range("2025-02-01 11:00", periods=2, freq="D", tz="Europe/Brussels"),
        observation_safety_lag="30min",
        context_window="14D",
        training_issue_times=pd.date_range(
            "2025-01-15 11:00", periods=10, freq="D", tz="Europe/Brussels"
        ),
        fit_time="2025-01-27T08:00Z",
        reference_model="seasonal_persistence_48h",
        on_error="strict",
    )
    results = benchmark(
        data=data,
        config=config,
        models=["seasonal_persistence_48h", "lightgbm"],
        data_source="Caller-prepared local national PV CSV; preserve its provenance separately",
    )
    print(results.leaderboard().to_string(index=False))
    print(results.config)
    print(results.metadata["model_metadata"])


if __name__ == "__main__":
    main()
