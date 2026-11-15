"""Frozen real-Elia comparison; local snapshots/weights only, no test-set tuning."""

import argparse
import gc
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from solarcast.backtesting import backtest
from solarcast.evaluation import compare_models
from solarcast.experiments import (
    daylight,
    grouped_diagnostics,
    load_snapshot,
    quality_summary,
    sha256,
    write_json,
)
from solarcast.models import Forecaster, SeasonalPersistenceForecaster
from solarcast.models.chronos import ChronosConfig, ChronosForecaster
from solarcast.models.lightgbm import LightGBMForecaster
from solarcast.models.lightgbm_data import FEATURE_NAMES, build_training_dataset
from solarcast.models.t0 import T0Config, T0Forecaster
from solarcast.tsfm_experiment import (
    MODEL_IDS,
    TimedForecaster,
    TSFMExperimentConfig,
    cache_identity,
    compute_summary,
    daily_wins,
    error_correlations,
    load_cache,
    save_cache,
    selected_cases,
    verify_snapshot,
)


def source_hashes(root: Path) -> dict[str, str]:
    paths = [
        p
        for folder in ("src", "scripts", "configs")
        for p in (root / folder).rglob("*")
        if p.suffix in (".py", ".toml")
    ]
    paths.append(root / "pyproject.toml")
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def load_predictions(
    output: Path, freeze: dict[str, Any], config: TSFMExperimentConfig, root: Path
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    expected = config.parent(root).spec("test")
    keys = pd.MultiIndex.from_tuples(
        [
            (issue, target)
            for issue in expected.issue_times
            for target in expected.target_policy(issue)
        ],
        names=["issue_time", "target_time"],
    )
    frames, availability, compute = [], [], []
    for model in MODEL_IDS:
        predictions, origins, _, timing = load_cache(output, freeze["identity"], model)
        actual_keys = pd.MultiIndex.from_frame(predictions[["issue_time", "target_time"]])
        if not actual_keys.equals(keys):
            raise ValueError(
                f"Cached population does not match the frozen target schedule: {model}"
            )
        if len(origins) != len(expected.issue_times) or origins.issue_time.duplicated().any():
            raise ValueError("Cached origin population differs from the frozen schedule")
        frames.append(predictions)
        own = compare_models([predictions]).summary.iloc[0]
        failed = int(origins.status.eq("forecast_failed").sum())
        availability.append(
            {
                "model_id": model,
                "requested_origins": len(origins),
                "successful_origins": len(origins) - failed,
                "failed_origins": failed,
                "origin_availability_percent": 100 * (len(origins) - failed) / len(origins),
                "forecast_availability_percent": float(own.forecast_availability) * 100,
                "n_expected": int(own.n_expected),
                "n_valid_pairs": int(own.n_valid_pairs),
            }
        )
        compute.append({"model_id": model, **timing})
        failures = origins.loc[origins.status.eq("forecast_failed")]
        failures.groupby(["error_type", "error_message"], dropna=False).size().rename(
            "origins"
        ).to_csv(output / f"failure_reasons_{model}.csv")
    return frames, pd.DataFrame(availability), pd.DataFrame(compute)


def evaluate(output: Path, config: TSFMExperimentConfig, root: Path) -> None:
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    payload = freeze["identity_payload"]
    if (
        payload["configuration"] != asdict(config)
        or payload["sources"] != source_hashes(root)
        or freeze["identity"] != cache_identity(payload)
        or payload["stage"] != "test"
    ):
        raise ValueError("Evaluation configuration/source identity differs from the frozen run")
    verify_snapshot(root, root / config.snapshot_manifest)
    frames, availability, compute = load_predictions(output, freeze, config, root)
    reference = config.parent(root).reference_model
    benchmark = compare_models(frames, reference_model=reference)
    light = compare_models(
        frames, reference_model=reference, subset=daylight, subset_name="reference_point_daylight"
    )
    benchmark.leaderboard().to_csv(output / "leaderboard.csv", index=False)
    light.leaderboard().to_csv(output / "daylight_leaderboard.csv", index=False)
    benchmark.aligned_predictions.to_csv(output / "aligned_predictions.csv", index=False)
    availability.to_csv(output / "availability.csv", index=False)
    compute.merge(benchmark.summary[["model_id", "rmse_mw"]], on="model_id").to_csv(
        output / "compute_summary.csv", index=False
    )
    for by in ("horizon", "local_hour", "month", "season"):
        benchmark.grouped(by).to_csv(output / f"by_{by}.csv", index=False)
    daily = grouped_diagnostics(benchmark.aligned_predictions, reference, "local_date")
    daily.to_csv(output / "daily_metrics.csv", index=False)
    grouped_diagnostics(benchmark.aligned_predictions, reference, "horizon_6h").to_csv(
        output / "horizon_metrics.csv", index=False
    )
    detail, wins = daily_wins(daily, config.tie_atol_mw)
    detail.to_csv(output / "daily_winners.csv", index=False)
    wins.to_csv(output / "daily_win_summary.csv", index=False)
    selected_cases(daily, config.tie_atol_mw).to_csv(output / "selected_cases.csv", index=False)
    error_correlations(benchmark.aligned_predictions).to_csv(output / "error_correlations.csv")
    error_correlations(light.aligned_predictions).to_csv(output / "daylight_error_correlations.csv")
    daily.pivot(index="local_date", columns="model_id", values="rmse_mw").corr().to_csv(
        output / "daily_rmse_correlations.csv"
    )
    plots(output, config)
    write_json(
        output / "results_manifest.json",
        {
            "identity": freeze["identity"],
            "evaluated_utc": str(datetime.now(UTC)),
            "files": {
                p.name: sha256(p)
                for p in sorted(output.iterdir())
                if p.is_file() and p.name != "results_manifest.json"
            },
        },
    )
    print(benchmark.leaderboard().to_string(index=False), flush=True)
    print("DAYLIGHT\n" + light.leaderboard().to_string(index=False), flush=True)


def plots(output: Path, config: TSFMExperimentConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    frame = pd.read_csv(output / "aligned_predictions.csv")
    frame["target_time"] = pd.to_datetime(frame.target_time, utc=True)
    local = frame.target_time.dt.tz_convert("Europe/Brussels")
    dates = local.dt.strftime("%Y-%m-%d")
    cases = pd.read_csv(output / "selected_cases.csv")
    difficult = cases.loc[cases.selection.eq("difficult"), "local_date"].iloc[0]
    colors = ["#e08214", "#2166ac", "#359767", "#9b4d96"]
    for filename, days in (
        ("representative_periods.png", config.normal_plot_days),
        ("difficult_day.png", [difficult]),
    ):
        figure, axes = plt.subplots(len(days), 1, figsize=(11, 3.4 * len(days)), squeeze=False)
        for ax, day in zip(axes[:, 0], days, strict=True):
            selected = frame.loc[dates.eq(day)]
            actual = selected.drop_duplicates("target_time").sort_values("target_time")
            ax.plot(
                actual.target_time, actual.actual_mw, color="black", lw=2, label="Actual Elia PV"
            )
            for model, color in zip(MODEL_IDS, colors, strict=True):
                rows = selected.loc[selected.model_id.eq(model)].sort_values("target_time")
                ax.plot(rows.target_time, rows.prediction_mw, label=model, color=color, lw=1.3)
            ax.set(title=day, ylabel="PV power (MW)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=local.dt.tz))  # type: ignore[no-untyped-call]
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=2)
        axes[-1, 0].set_xlabel("Belgian local target time (Europe/Brussels)")
        figure.suptitle("Raw forecasts on common support — Elia Transmission Belgium SA")
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    labels = ["48h seasonal", "LightGBM", "Chronos-2", "t0-gamma"]
    for ax, filename, title in zip(
        axes,
        ["leaderboard.csv", "daylight_leaderboard.csv"],
        ["All intervals (primary)", "Daylight only"],
        strict=True,
    ):
        scores = pd.read_csv(output / filename).set_index("model_id").loc[MODEL_IDS]
        ax.bar(labels, scores.rmse_mw, color=colors)
        ax.set(title=title, ylabel="Common-support RMSE (MW)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "rmse_comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--timing-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    root = Path.cwd()
    config = TSFMExperimentConfig.load(args.config, root)
    parent = config.parent(root)
    if args.validate_only:
        print(f"Valid frozen config: {len(parent.origins('test'))} origins; no inference")
        return
    output = root / config.output_dir
    if args.evaluate_only:
        evaluate(output, config, root)
        return
    stage = "timing" if args.timing_only else "test"
    output = output.with_name(output.name + "_timing") if args.timing_only else output
    if output.exists():
        raise FileExistsError(f"Preserve existing run: {output}; use --evaluate-only to rescore")
    packages = {name: version(name) for name in config.packages}
    if packages != config.packages:
        raise ValueError(f"Environment differs from frozen package pins: {packages}")
    shards = verify_snapshot(root, root / config.snapshot_manifest)
    sources = source_hashes(root)
    if not args.timing_only:
        timing = json.loads(
            (output.with_name(output.name + "_timing") / "estimate.json").read_text()
        )
        if timing["config_sha256"] != sha256(args.config) or timing["sources"] != sources:
            raise ValueError(
                "Run the timing sample for this exact frozen configuration/source first"
            )
        print(
            f"Preflight estimate: {timing['estimated_test_predict_seconds']:.1f}s predict calls, "
            f"plus {timing['training_preparation_seconds']:.1f}s train preparation and loading",
            flush=True,
        )
    import torch

    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.seed)
    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "lightgbm_threads": 1,
        "device": "cpu",
    }
    payload = {
        "configuration": asdict(config),
        "parent_expanded": asdict(parent),
        "snapshot_sha256": config.snapshot_sha256,
        "sources": sources,
        "packages": packages,
        "hardware": hardware,
        "stage": stage,
    }
    identity = cache_identity(payload)
    output.mkdir(parents=True)
    (output / "config.toml").write_bytes(args.config.read_bytes())
    (output / "parent_config.toml").write_bytes((root / config.parent_config).read_bytes())
    write_json(
        output / "freeze.json",
        {
            "frozen_utc": str(datetime.now(UTC)),
            "identity": identity,
            "identity_payload": payload,
            "config_sha256": sha256(args.config),
            "python": sys.version,
            "solarcast_version": version("solarcast-belgium"),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            or None,
            "git_status": subprocess.run(
                ["git", "status", "--short"], capture_output=True, text=True, check=True
            ).stdout,
            "other_packages": {
                n: version(n) for n in ("pvlib", "tzdata", "huggingface-hub", "matplotlib")
            },
            "holdout_disclosure": (
                "History-only LightGBM scores and June smoke outputs already known; no tuning"
            ),
        },
    )
    (output / "environment.txt").write_text(
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout,
        encoding="utf-8",
    )
    write_json(output / "snapshot.json", shards)
    data, loaded_shards = load_snapshot(parent, root)
    if cache_identity(loaded_shards) != cache_identity(shards):
        raise ValueError("Loaded snapshot provenance differs from frozen manifest")
    write_json(
        output / "data_quality.json",
        quality_summary(
            data,
            pd.Timestamp(parent.data_start).tz_convert("UTC"),
            pd.Timestamp(parent.data_end).tz_convert("UTC"),
        ),
    )
    spec = config.timing_spec(root) if args.timing_only else parent.spec("test")
    estimates: list[float] = []
    preparation_seconds = 0.0
    for identifier in MODEL_IDS:
        model: Forecaster
        compute: dict[str, object] = {
            "device": "cpu",
            "context_window": parent.context_window,
            "load_seconds": 0.0,
            "cache_seconds": 0.0,
            "training_seconds": 0.0,
            "training_preparation_seconds": 0.0,
        }
        if identifier == MODEL_IDS[0]:
            model = SeasonalPersistenceForecaster("48h")
            compute.update(learned_model_size="no learned weights", checkpoint_bytes=0)
        elif identifier == "lightgbm":
            start = time.perf_counter()
            training = build_training_dataset(data, parent.spec("train"), fit_time=parent.fit_time)
            preparation_seconds = time.perf_counter() - start
            fitted = LightGBMForecaster(parent.model)
            start = time.perf_counter()
            fitted.fit(training)
            compute.update(
                training_seconds=time.perf_counter() - start,
                training_preparation_seconds=preparation_seconds,
                learned_model_size="100 trees; 15 leaves maximum each",
                checkpoint_bytes=None,
            )
            write_json(
                output / "training.json",
                {
                    "rows": len(training.y),
                    "candidate_rows": training.n_requested,
                    "missing_labels": training.n_missing_labels,
                    "features": FEATURE_NAMES,
                    "feature_missing_cells": int(training.X.isna().sum().sum()),
                    "fit_time": training.fit_time,
                    "label_cutoff": training.label_cutoff,
                    "configuration": asdict(parent.model),
                    "validation": "timing only; no tuning",
                },
            )
            model = fitted
            del training, fitted
        else:
            if identifier == "chronos":
                native: ChronosForecaster | T0Forecaster = ChronosForecaster(
                    ChronosConfig(
                        context_length=config.context_slots,
                        local_files_only=True,
                        device=config.device,
                        revision=config.chronos["revision"],
                    )
                )
                size, weight_bytes = "119.48M parameters", 477930472
            else:
                native = T0Forecaster(
                    T0Config(
                        context_length=config.context_slots,
                        local_files_only=True,
                        device=config.device,
                        revision=config.t0["revision"],
                    )
                )
                size, weight_bytes = "255.61M parameters", 1022492908
            try:
                native.load()
            except Exception as error:
                write_json(output / f"setup_failure_{identifier}.json", {"error": repr(error)})
                raise  # Completed model caches survive; never substitute predictions.
            compute.update(
                load_seconds=native.load_seconds,
                cache_seconds=native.cache_seconds,
                learned_model_size=size,
                checkpoint_bytes=weight_bytes,
                context_cap_slots=config.context_slots,
                checkpoint=native.metadata.checkpoint,
                revision=native.metadata.revision,
            )
            model = native
            del native
        timed = TimedForecaster(model)
        start = time.perf_counter()
        result = backtest(model=timed, data=data, spec=spec)
        compute["backtest_wall_seconds"] = time.perf_counter() - start
        calls = pd.DataFrame(timed.calls)
        compute.update(compute_summary(calls))
        save_cache(output, identity, identifier, result.predictions, result.origins, calls, compute)
        if args.timing_only:
            # Never compute accuracy on timing-validation origins.
            estimates.append(float(calls.predict_seconds.mean()) * len(parent.origins("test")))
        print(
            f"Saved {identifier}: {len(result.origins)} origins, "
            f"{result.origins.status.eq('forecast_failed').sum()} failures",
            flush=True,
        )
        del timed, model, result
        gc.collect()
    if args.timing_only:
        write_json(
            output / "estimate.json",
            {
                "config_sha256": sha256(args.config),
                "sources": sources,
                "estimated_test_predict_seconds": sum(estimates),
                "per_model_estimated_predict_seconds": dict(zip(MODEL_IDS, estimates, strict=True)),
                "training_preparation_seconds": preparation_seconds,
                "test_origins": len(parent.origins("test")),
                "note": (
                    "Two predeclared validation origins, timing only; "
                    "no accuracy or context selection"
                ),
            },
        )
        print(
            f"Estimated full predict-call runtime: {sum(estimates):.1f}s "
            "plus training/loading/engine overhead"
        )
    else:
        evaluate(output, config, root)


if __name__ == "__main__":
    main()
