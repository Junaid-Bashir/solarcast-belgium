"""Local-only t0 benchmark: install .[t0] and prepare pinned weights separately."""

import argparse
from pathlib import Path

import pandas as pd

from solarcast import BenchmarkConfig, benchmark
from solarcast.models.t0 import T0Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Prepared Hugging Face cache")
    args = parser.parse_args()
    data = pd.read_csv(args.input_csv)
    stamps = [pd.Timestamp(value) for value in data.target_time]
    if any(pd.isna(value) or value.tzinfo is None for value in stamps):
        parser.error("target_time must contain explicit timezone offsets")
    data["target_time"] = pd.to_datetime(stamps, utc=True)
    results = benchmark(
        data=data,
        models=["seasonal_persistence_48h", "t0"],
        config=BenchmarkConfig(
            issue_times=pd.date_range(
                "2025-02-01 11:00", periods=2, freq="D", tz="Europe/Brussels"
            ),
            observation_safety_lag="30min",
            t0=T0Config(cache_dir=args.cache_dir, local_files_only=True),
            on_error="strict",
        ),
    )
    print(results.leaderboard().to_string(index=False))


if __name__ == "__main__":
    main()
