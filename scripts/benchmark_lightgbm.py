"""Real Elia experiment only. Run from repository root; never substitutes synthetic data."""

import argparse
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from solarcast.backtesting import backtest
from solarcast.evaluation import compare_models
from solarcast.experiments import (
    ExperimentConfig,
    daylight,
    grouped_diagnostics,
    load_snapshot,
    quality_summary,
    sha256,
    write_json,
)
from solarcast.models import (
    Forecaster,
    LastValuePersistenceForecaster,
    SeasonalPersistenceForecaster,
)
from solarcast.models.lightgbm import LightGBMForecaster
from solarcast.models.lightgbm_data import FEATURE_NAMES, build_training_dataset

logger = logging.getLogger(__name__)


def plots(
    output: Path, days: list[str], reference: str, profile_name: str = "actual_vs_forecast.png"
) -> None:
    """Rebuild figures exclusively from saved aligned predictions; no model refit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    frame = pd.read_csv(output / "aligned_predictions.csv")
    frame["target_time"] = pd.to_datetime(frame.target_time, utc=True)
    local = frame.target_time.dt.tz_convert("Europe/Brussels")
    figure, axes = plt.subplots(len(days), 1, figsize=(11, 3.2 * len(days)), squeeze=False)
    for ax, day in zip(axes[:, 0], days, strict=True):
        selected = frame.loc[local.dt.strftime("%Y-%m-%d") == day]
        actual = selected.drop_duplicates("target_time").sort_values("target_time")
        ax.plot(actual.target_time, actual.actual_mw, color="black", label="Actual Elia PV", lw=2)
        for model, color in ((reference, "#e08214"), ("lightgbm", "#2166ac")):
            rows = selected.loc[selected.model_id == model].sort_values("target_time")
            ax.plot(rows.target_time, rows.prediction_mw, label=model, color=color, lw=1.5)
        ax.set(title=day, ylabel="PV power (MW)")
        # Matplotlib's DateFormatter constructor has no typed signature.
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M", tz=local.dt.tz)  # type: ignore[no-untyped-call]
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1, 0].set_xlabel("Belgian local target time (Europe/Brussels)")
    figure.suptitle("Real Elia holdout: actual vs raw forecast — Elia Transmission Belgium SA")
    figure.tight_layout()
    figure.savefig(output / profile_name, dpi=160)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(8, 4))
    for model in (reference, "seasonal_persistence_7d", "lightgbm"):
        own = frame.loc[frame.model_id == model]
        errors = np.sort(np.abs(own.prediction_mw - own.actual_mw))
        ax.plot(errors, np.arange(1, len(errors) + 1) / len(errors), label=model)
    ax.set(
        xlabel="Absolute error (MW)",
        ylabel="Empirical cumulative fraction",
        title="All-interval errors on identical common support",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "absolute_error_ecdf.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Failures remain in origins.csv; avoid hundreds of expected 24h error log lines.
    logging.getLogger("solarcast.backtesting").setLevel(logging.ERROR)
    config = ExperimentConfig.load(args.config)
    root = Path.cwd()
    output = root / config.output_dir
    if args.validate_only:
        print("Configuration valid; no data downloaded or performance inspected")
        return
    if args.plots_only:
        if sha256(args.config) != (output / "config.sha256").read_text().strip():
            raise ValueError("Plot configuration differs from recorded run")
        plots(output, config.representative_days, config.reference_model)
        cases = pd.read_csv(output / "selected_daily_cases.csv")
        worst_cases = cases.loc[cases.selection == "worst"].sort_values("rmse_mw", ascending=False)
        plots(output, worst_cases.local_date.tolist(), config.reference_model, "worst_days.png")
        return
    if not args.download_only and output.exists():
        raise FileExistsError(
            f"Output already exists: {output}; preserve it or choose a new output_dir"
        )
    started = datetime.now(UTC)
    timer = time.monotonic()
    data, shards = load_snapshot(config, root)
    quality = quality_summary(
        data,
        pd.Timestamp(config.data_start).tz_convert("UTC"),
        pd.Timestamp(config.data_end).tz_convert("UTC"),
    )
    print("REAL ELIA DATA QUALITY", quality, flush=True)
    if args.download_only:
        return
    output.mkdir(parents=True)
    (output / "config.toml").write_bytes(args.config.read_bytes())
    (output / "config.sha256").write_text(sha256(args.config) + "\n")
    write_json(output / "data_quality.json", quality)
    write_json(output / "snapshot.json", shards)
    local_month = data.target_time.dt.tz_convert(config.timezone).dt.strftime("%Y-%m")
    quality_by_month = (
        data.assign(local_month=local_month)
        .groupby("local_month")
        .agg(
            interval_rows=("target_time", "size"),
            null_pv=("pv_generation_mw", lambda values: values.isna().sum()),
            negative_pv=("pv_generation_mw", lambda values: (values < 0).sum()),
        )
    )
    quality_by_month.to_csv(output / "data_quality_by_month.csv")
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    (output / "environment.txt").write_text(frozen.stdout, encoding="utf-8")
    dataset = build_training_dataset(data, config.spec("train"), fit_time=config.fit_time)
    training_summary = {
        "candidate_rows": dataset.n_requested,
        "excluded_missing_targets": dataset.n_missing_labels,
        "rows_with_missing_features": int(dataset.X.isna().any(axis=1).sum()),
        "missing_feature_cells": int(dataset.X.isna().sum().sum()),
        "missing_by_feature": dataset.X.isna().sum().to_dict(),
        "final_rows": len(dataset.y),
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "fit_time": dataset.fit_time,
        "label_cutoff": dataset.label_cutoff,
        "first_target": dataset.rows.target_time.min(),
        "last_target": dataset.rows.target_time.max(),
    }
    write_json(output / "training.json", training_summary)
    dataset.rows.to_csv(output / "training_audit.csv", index=False)
    logger.info("Training summary: %s", training_summary)
    model = LightGBMForecaster(config.model)
    model.fit(dataset)
    adapters: list[Forecaster] = [
        SeasonalPersistenceForecaster(lag) for lag in config.baseline_lags
    ]
    if config.include_last_value:
        adapters.append(LastValuePersistenceForecaster())
    adapters.append(model)
    results = []
    for adapter in adapters:
        logger.info("Test: %s (%d origins)", adapter.model_id, len(config.origins("test")))
        result = backtest(model=adapter, data=data, spec=config.spec("test"))
        results.append(result)
        result.predictions.to_csv(output / f"predictions_{adapter.model_id}.csv", index=False)
        result.origins.to_csv(output / f"origins_{adapter.model_id}.csv", index=False)
    # A deliberately reported ineligible 24h model must not empty every useful model's support.
    compared_ids = {SeasonalPersistenceForecaster(lag).model_id for lag in config.comparison_lags}
    compared_ids.add("lightgbm")
    if config.include_last_value:
        compared_ids.add("last_value_persistence")
    compared = [result for result in results if result.model_id in compared_ids]
    benchmark = compare_models(compared, reference_model=config.reference_model)
    light = compare_models(
        compared,
        reference_model=config.reference_model,
        subset=daylight,
        subset_name="reference_point_daylight",
    )
    # Separate availability table includes every model, even the structurally infeasible 24h one.
    availability = pd.concat(
        [compare_models([result]).summary for result in results], ignore_index=True
    )
    # Do not expose individual-support error scores as if they were comparable.
    availability = availability[
        [
            "model_id",
            "n_expected",
            "n_origins",
            "n_failed_origins",
            "forecast_availability",
            "n_actual_missing",
            "n_prediction_missing",
            "n_valid_pairs",
        ]
    ]
    availability.to_csv(output / "availability.csv", index=False)
    benchmark.summary.to_csv(output / "metrics.csv", index=False)
    benchmark.leaderboard().to_csv(output / "leaderboard.csv", index=False)
    light.leaderboard().to_csv(output / "daylight_leaderboard.csv", index=False)
    benchmark.aligned_predictions.to_csv(output / "aligned_predictions.csv", index=False)
    for by in ("horizon", "local_hour", "month", "season"):
        benchmark.grouped(by).to_csv(output / f"by_{by}.csv", index=False)
    for by in ("horizon_6h", "local_date"):
        grouped_diagnostics(benchmark.aligned_predictions, config.reference_model, by).to_csv(
            output / f"by_{by}.csv", index=False
        )
    daily = pd.read_csv(output / "by_local_date.csv")
    own = daily.loc[daily.model_id == "lightgbm"].sort_values(["rmse_mw", "local_date"])
    cases = pd.concat([own.head(3).assign(selection="best"), own.tail(3).assign(selection="worst")])
    cases.to_csv(output / "selected_daily_cases.csv", index=False)
    worst = own.tail(3).sort_values("rmse_mw", ascending=False).local_date.tolist()
    case_rows = []
    aligned = benchmark.aligned_predictions
    dates = aligned.target_time.dt.tz_convert(config.timezone).dt.strftime("%Y-%m-%d")
    for day in worst:
        for identifier, group in aligned.loc[dates.isin([day])].groupby("model_id"):
            error = group.prediction_mw.astype(float) - group.actual_mw.astype(float)
            hours = group.target_time.dt.tz_convert(config.timezone).dt.hour
            midpoint = error.loc[hours.between(10, 15)]
            largest = error.abs().idxmax()
            case_rows.append(
                {
                    "local_date": day,
                    "model_id": identifier,
                    "n_scored": len(group),
                    "midday_mae_10_to_16_mw": float(midpoint.abs().mean()),
                    "midday_bias_10_to_16_mw": float(midpoint.mean()),
                    "largest_absolute_error_mw": float(error.abs().max()),
                    "largest_error_target_local": str(
                        group.loc[largest, "target_time"].tz_convert(config.timezone)
                    ),
                }
            )
    pd.DataFrame(case_rows).to_csv(output / "worst_case_diagnostics.csv", index=False)
    plots(output, config.representative_days, config.reference_model)
    # Worst days are selected by the declared diagnostic rule, never used to change the model.
    plots(output, worst, config.reference_model, "worst_days.png")
    git_commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True, check=False
    )
    git_status = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True, check=True
    )
    sources = {
        str(path.relative_to(root)): sha256(path)
        for folder in ("src", "scripts", "configs")
        for path in sorted((root / folder).rglob("*"))
        if path.suffix in (".py", ".toml")
    }
    metadata = {
        "started_utc": str(started),
        "finished_utc": str(datetime.now(UTC)),
        "elapsed_seconds": time.monotonic() - timer,
        "python": sys.version,
        "platform": platform.platform(),
        "config_sha256": sha256(args.config),
        "git_commit": git_commit.stdout.strip() if git_commit.returncode == 0 else None,
        "git_status": git_status.stdout,
        "source_sha256": sources,
        "packages": {
            name: version(name)
            for name in (
                "solarcast-belgium",
                "lightgbm",
                "pandas",
                "numpy",
                "pvlib",
                "tzdata",
                "matplotlib",
                "httpx",
            )
        },
        "model_config": asdict(config.model),
        "feature_names": FEATURE_NAMES,
        "validation": "reserved and unused; no tuning or early stopping",
        "training": "one fit before validation/test; no retraining",
        "availability": "completed intervals with safety lag; publication vintages unverified",
        "snapshot_manifest_sha256": sha256(output / "snapshot.json"),
        "test_target_counts": results[-1].origins.target_count.value_counts().to_dict(),
        "comparison_ids": sorted(compared_ids),
        "reference_model": config.reference_model,
        "unranked_baselines": sorted(set(r.model_id for r in results) - compared_ids),
        "tuning": "none; configuration frozen before first real test evaluation",
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "metadata.json", metadata)
    print(benchmark.leaderboard().to_string(index=False), flush=True)
    print("DAYLIGHT\n", light.leaderboard().to_string(index=False), flush=True)
    print(f"Saved real experiment to {output}", flush=True)


if __name__ == "__main__":
    main()
