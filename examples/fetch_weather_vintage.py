"""Opt-in two-hour tomorrow GFS smoke; no bulk archive or model experiment.

Run with ``python examples/fetch_weather_vintage.py --live`` after installing
``.[weather]``. Use --full-day explicitly for all 92/96/100 tomorrow targets.
Without --live this example exits before network access.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from solarcast.backtesting import day_ahead_local_day
from solarcast.data.weather import (
    GFSClient,
    NWPAvailability,
    WeatherCache,
    belgium_weights,
    select_nwp_vintage,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="allow bounded NODD HTTP retrieval")
    parser.add_argument("--issue", default="2025-06-10T11:00:00+02:00")
    parser.add_argument("--cache", type=Path, default=Path("data/weather_gfs_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/research/weather/gfs_vintage")
    )
    parser.add_argument("--full-day", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.exit(message="No network access. Pass --live for the bounded historical smoke.\n")
    issue = pd.Timestamp(args.issue)
    if issue.tzinfo is None:
        parser.error("--issue must have an explicit timezone/offset")
    issue = issue.tz_convert("UTC")
    policy = NWPAvailability()
    # Construct a candidate initialization using the declared approximation;
    # selection below still checks availability and complete target coverage.
    reference = (issue - policy.delay).floor("6h")
    targets = day_ahead_local_day(issue)
    if not args.full_day:
        local = targets.tz_convert("Europe/Brussels")
        targets = targets[(local.hour >= 14) & (local.hour < 16)]
    first = int((targets.min().floor("h") - reference) / pd.Timedelta(hours=1))
    last = int(
        ((targets.max() + pd.Timedelta(minutes=15)).ceil("h") - reference) / pd.Timedelta(hours=1)
    )
    cache = WeatherCache(args.cache)
    with GFSClient(cache=cache) as client:
        run = client.fetch_forecast_run(reference, steps=list(range(first, last + 1)))
    # Prove this path uses persisted native data, not the in-memory downloaded frame.
    reloaded = cache.load(run.metadata["identity"])
    assert reloaded is not None and reloaded.version == run.version
    weights = belgium_weights()
    weather = select_nwp_vintage([reloaded], issue, targets, weights=weights, policy=policy)
    args.output.mkdir(parents=True, exist_ok=True)
    weather.features.to_csv(args.output / "features.csv", index=False)
    report = {
        "issue_time": issue.isoformat(),
        "reference_time": weather.reference_time.isoformat(),
        "available_at": weather.available_at.isoformat(),
        "availability_basis": "assumed",
        "policy": policy.name,
        "observed_available_at": None,
        "run_version": run.version,
        "weights_id": weights.identity,
        "positive_area_cells": len(weights.table),
        "native_rows": len(run.data),
        "bounding_grid_cells": len(weights.expected_cells),
        "targets": len(targets),
        "first_target": targets[0].isoformat(),
        "last_target": targets[-1].isoformat(),
        "identity": run.metadata["identity"],
        "source_transfer_bytes_in_cached_bundle": run.metadata["transferred_bytes"],
        "cache_reload": True,
        "feature_min": weather.features.iloc[:, 1:].min().to_dict(),
        "feature_max": weather.features.iloc[:, 1:].max().to_dict(),
        "limitation": "Lag approximation; archived object is not a verified historical as-of copy.",
    }
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
