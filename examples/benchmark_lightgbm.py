"""Explicit chronological experiment; defaults to synthetic data, never implicit downloads."""

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest, day_ahead_local_day
from solarcast.data import EliaPVClient
from solarcast.evaluation import compare_models
from solarcast.models import (
    Forecaster,
    LastValuePersistenceForecaster,
    SeasonalPersistenceForecaster,
)
from solarcast.models.lightgbm import LightGBMConfig, LightGBMForecaster
from solarcast.models.lightgbm_data import build_training_dataset


def daily_origins(start: str, end: str, local_clock: str) -> pd.DatetimeIndex:
    """Explicit date-only bounds mean Brussels origin dates, [start, end)."""
    first, stop = date.fromisoformat(start), date.fromisoformat(end)
    clock = time.fromisoformat(local_clock)
    if first >= stop or clock.tzinfo is not None:
        raise ValueError("Need increasing local dates and an offset-free local issue clock")
    dates = [first + timedelta(days=i) for i in range((stop - first).days)]
    # Ambiguous/nonexistent clocks raise instead of silently moving DST issue times.
    return pd.DatetimeIndex(
        [
            pd.Timestamp(datetime.combine(day, clock)).tz_localize(
                "Europe/Brussels", ambiguous="raise", nonexistent="raise"
            )
            for day in dates
        ]
    ).tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, help="Normalized local PV CSV; no network")
    parser.add_argument("--fetch-start", help="Explicit aware Elia historical fetch start")
    parser.add_argument("--fetch-end", help="Explicit aware Elia historical fetch end, exclusive")
    parser.add_argument("--train-start", default="2026-06-09")
    parser.add_argument("--train-end", default="2026-06-29", help="Exclusive training origin date")
    parser.add_argument("--fit-time", default="2026-06-30T08:00Z")
    parser.add_argument("--validation-start", default="2026-06-30")
    parser.add_argument("--validation-end", default="2026-07-03")
    parser.add_argument("--test-start", default="2026-07-04")
    parser.add_argument("--test-end", default="2026-07-07")
    parser.add_argument("--issue-time", default="11:00", help="Brussels local clock, not UTC")
    parser.add_argument("--safety-lag", default="30min")
    parser.add_argument("--context-window", default="14D")
    parser.add_argument("--training-window", default="365D", help="Elapsed training-origin window")
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if bool(args.fetch_start) != bool(args.fetch_end) or (args.input_csv and args.fetch_start):
        parser.error("Use one source: --input-csv or both --fetch-start and --fetch-end")
    train_issues = daily_origins(args.train_start, args.train_end, args.issue_time)
    validation = daily_origins(args.validation_start, args.validation_end, args.issue_time)
    test_issues = daily_origins(args.test_start, args.test_end, args.issue_time)
    fit_time = pd.Timestamp(args.fit_time)
    if fit_time.tzinfo is None:
        parser.error("--fit-time needs an explicit timezone/offset")
    fit_time = fit_time.tz_convert("UTC")
    if not (train_issues[-1] < fit_time <= validation[0] < test_issues[0]):
        parser.error("Require training origins < fit time <= validation start < test start")
    validation_end = day_ahead_local_day(validation[-1])[-1] + pd.Timedelta(minutes=15)
    if validation_end > test_issues[0]:
        parser.error("Validation target intervals must finish before the first test issue")
    if args.input_csv:
        data = pd.read_csv(args.input_csv)
        stamps = [pd.Timestamp(value) for value in data.target_time]
        if any(pd.isna(value) or value.tzinfo is None for value in stamps):
            parser.error("CSV target_time values must contain explicit offsets")
        data["target_time"] = pd.to_datetime(stamps, utc=True)
        data["pv_generation_mw"] = pd.to_numeric(data.pv_generation_mw).astype("Float64")
        source_name = f"local CSV: {args.input_csv}"
    elif args.fetch_start:
        with EliaPVClient() as client:
            data = client.fetch_history(start=args.fetch_start, end=args.fetch_end)
        source_name = "Elia historical data (explicit fetch; approximate availability)"
    else:
        times = pd.date_range(
            train_issues[0] - pd.Timedelta(days=8),
            day_ahead_local_day(test_issues[-1])[-1],
            freq="15min",
        )
        local = times.tz_convert("Europe/Brussels")
        clock_hours = local.hour.to_numpy() + local.minute.to_numpy() / 60
        days = (times - times[0]).total_seconds().to_numpy() / 86400
        # Fictional daily pattern and smooth amplitude; not observed weather or real PV.
        generation = np.maximum(0, np.sin(np.pi * (clock_hours - 6) / 12)) * (
            1000 + 150 * np.sin(days / 3)
        )
        data = pd.DataFrame({"target_time": times, "pv_generation_mw": generation})
        source_name = "SYNTHETIC daily-profile fixture; NOT Belgian performance evidence"
    policy = EventTimeCutoff(args.safety_lag)
    train_spec = BacktestSpec(
        train_issues, availability_policy=policy, context_window=args.context_window
    )
    dataset = build_training_dataset(
        data, train_spec, fit_time=fit_time, training_window=args.training_window
    )
    model = LightGBMForecaster(
        LightGBMConfig(n_estimators=args.n_estimators, random_state=args.seed)
    )
    model.fit(dataset)
    test_spec = BacktestSpec(
        test_issues, availability_policy=policy, context_window=args.context_window
    )
    models: list[Forecaster] = [
        LastValuePersistenceForecaster(),
        SeasonalPersistenceForecaster("48h"),
        SeasonalPersistenceForecaster("7D"),
        model,
    ]
    results = [backtest(model=adapter, data=data, spec=test_spec) for adapter in models]
    benchmark = compare_models(results, reference_model="seasonal_persistence_7d")
    print(source_name)
    print(f"Train origins [{train_issues[0]}, {train_issues[-1]}]; fit boundary {fit_time}")
    print(
        f"Validation [{validation[0]}, {validation[-1]}] reserved and unused; "
        "no early stopping/tuning"
    )
    print(f"Test origins [{test_issues[0]}, {test_issues[-1]}]; model trained once")
    print(
        f"Training rows={len(dataset.y)}; missing labels excluded={dataset.n_missing_labels}; "
        f"LightGBM={model.training_info.lightgbm_version}; config={model.config}"
    )
    print(benchmark.leaderboard().to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
