"""Controlled weather-LightGBM experiment; explicit bounded/resumable data preparation."""

import argparse
import os
import platform
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from solarcast.backtesting import backtest
from solarcast.data.weather import WeatherError
from solarcast.evaluation import compare_models
from solarcast.experiments import daylight, grouped_diagnostics, load_snapshot, sha256, write_json
from solarcast.models.lightgbm import LightGBMForecaster
from solarcast.models.lightgbm_data import FeatureSet, build_training_dataset, feature_names
from solarcast.tsfm_experiment import (
    TimedForecaster,
    TSFMExperimentConfig,
    cache_identity,
    compute_summary,
    daily_wins,
    error_correlations,
    load_cache,
    save_cache,
    verify_snapshot,
)
from solarcast.weather_experiment import (
    PreparedWeather,
    WeatherExperimentConfig,
    cached_comparators,
    choose_weather,
    feature_entry,
    plan,
    prepare_origin,
    read_json,
)


def prepare(config: WeatherExperimentConfig, root: Path) -> None:
    parent = config.parent(root)
    issues = [issue for split in ("train", "validation", "test") for issue in parent.origins(split)]
    start = time.perf_counter()
    statuses = []
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = {
            pool.submit(prepare_origin, str(root), config, issue.isoformat()): issue
            for issue in issues
        }
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            if len(statuses) % 5 == 0 or status["status"] != "ok":
                print(
                    f"Weather {len(statuses)}/{len(issues)}: {status['issue_time']} "
                    f"{status['status']} {status.get('error_message', '')}; "
                    f"wall={time.perf_counter() - start:.1f}s",
                    flush=True,
                )
    write_json(
        root / config.output_dir / "preparation.json",
        {
            "wall_seconds": time.perf_counter() - start,
            "origins": len(statuses),
            "workers": config.workers,
            "statuses": sorted(statuses, key=lambda s: s["issue_time"]),
        },
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


def weather_inventory(config: WeatherExperimentConfig, root: Path) -> pd.DataFrame:
    records = []
    for split in ("train", "validation", "test"):
        for issue in config.parent(root).origins(split):
            entry = feature_entry(root, config, issue)
            status = read_json(entry / "status.json")
            if status["plan_identity"] != plan(config, root)["identity"]:
                raise ValueError("Prepared origin plan mismatch")
            if (
                status["status"] == "ok"
                and sha256(entry / "features.parquet") != status["features_sha256"]
            ):
                raise ValueError("Prepared weather feature hash mismatch")
            records.append({"split": split, **status})
    return pd.DataFrame(records)


def run_models(config: WeatherExperimentConfig, root: Path) -> None:
    output, parent = root / config.output_dir, config.parent(root)
    if (output / "freeze.json").exists():
        raise ValueError(
            "Final model run already frozen; use --evaluate-only, never overwrite inference"
        )
    inventory = weather_inventory(config, root)
    inventory.to_csv(output / "weather_availability.csv", index=False)
    prior_freeze = read_json(root / config.tsfm_output_dir / "freeze.json")
    snapshot = root / prior_freeze["identity_payload"]["configuration"]["snapshot_manifest"]
    verify_snapshot(root, snapshot)
    data, _ = load_snapshot(parent, root)
    comparators = cached_comparators(config, root)
    prior_config = TSFMExperimentConfig.load(root / config.tsfm_config, root)
    for package in ("lightgbm", "numpy", "pandas"):
        if version(package) != prior_config.packages[package]:
            raise ValueError(
                f"Controlled comparison requires {package}=={prior_config.packages[package]}"
            )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: version(name)
            for name in (
                "solarcast-belgium",
                "lightgbm",
                "numpy",
                "pandas",
                "pvlib",
                "eccodes",
                "pyarrow",
                "shapely",
                "pyproj",
                "matplotlib",
                "scipy",
            )
        },
    }
    payload: dict[str, Any] = {
        "plan": plan(config, root),
        "sources": source_hashes(root),
        "weather_availability_sha256": sha256(output / "weather_availability.csv"),
        "snapshot_sha256": sha256(snapshot),
        "environment": environment,
    }
    prefit = {
        "identity": cache_identity(payload),
        "payload": payload,
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    if (output / "pre_fit_freeze.json").exists():
        previous = read_json(output / "pre_fit_freeze.json")
        if previous["identity"] != prefit["identity"]:
            raise ValueError("Pre-fit freeze differs; inspect rather than silently refit")
    else:
        write_json(output / "pre_fit_freeze.json", prefit)
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, expected_hash in payload["sources"].items():
            if sha256(root / relative) != expected_hash:
                raise ValueError("Source changed while freezing the experiment")
            archive.write(root / relative, relative)
    write_json(
        output / "metadata.json",
        {
            "classification": config.classification,
            "prefit_identity": prefit["identity"],
            "environment": environment,
            "pv_availability": "event-time completion plus 30min assumption",
            "nwp_availability": (
                "initialization plus six hours assumption; archive as-of fidelity unverified"
            ),
            "cached_tsfm_timings": "Original tsfm_zero_shot_v1 CPU run; no TSFM rerun",
            "source_archive_sha256": sha256(output / "source_snapshot.zip"),
        },
    )
    import subprocess

    installed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
    )
    (output / "environment.txt").write_text(installed.stdout, encoding="utf8")
    print("Pre-fit source/data/environment freeze:", prefit["identity"], flush=True)
    store = PreparedWeather(root, config)
    start = time.perf_counter()
    complete = build_training_dataset(
        data,
        parent.spec("train"),
        fit_time=parent.fit_time,
        feature_set="history+nwp",
        weather_provider=store,
    )
    preparation_seconds = time.perf_counter() - start
    complete.rows.to_csv(output / "training_rows.csv", index=False)
    complete.origin_failures.to_csv(output / "training_weather_failures.csv", index=False)
    models: dict[str, LightGBMForecaster] = {}
    validation, timings = [], []
    for name in config.ablations:
        feature_set = cast(FeatureSet, name)
        dataset = replace(
            complete, X=complete.X[list(feature_names(feature_set))].copy(), feature_set=feature_set
        )
        model = LightGBMForecaster(parent.model, feature_set=feature_set)
        start = time.perf_counter()
        model.fit(dataset)
        train_seconds = time.perf_counter() - start
        models[name] = model
        timed = TimedForecaster(model)
        result = backtest(
            model=timed,
            data=data,
            spec=parent.spec("validation"),
            weather_provider=store if name != "history" else None,
        )
        result.predictions.to_csv(output / f"validation_{model.model_id}.csv", index=False)
        result.origins.to_csv(output / f"validation_origins_{model.model_id}.csv", index=False)
        validation.append(result)
        timings.append(
            {
                "model_id": model.model_id,
                "feature_set": name,
                "training_seconds": train_seconds,
                "feature_count": len(dataset.X.columns),
                "training_rows": len(dataset.y),
                "requested_rows": complete.n_requested,
                "missing_labels": complete.n_missing_labels,
                "failed_training_origins": len(complete.origin_failures),
                "training_preparation_seconds_shared": preparation_seconds,
            }
        )
        assert model._booster is not None
        model._booster.save_model(str(output / f"model_{model.model_id}.txt"))
    ablation = compare_models(validation, reference_model="lightgbm").leaderboard()
    model_to_features = {model.model_id: name for name, model in models.items()}
    ablation["feature_set"] = ablation.model_id.map(model_to_features)
    ablation.to_csv(output / "validation_ablation.csv", index=False)
    chosen = choose_weather(ablation)
    selection = {
        "feature_set": chosen,
        "model_id": models[chosen].model_id,
        "selection_rule": config.selection_rule,
        "validation_only": True,
        "ablation_sha256": sha256(output / "validation_ablation.csv"),
        "training_support": "identical finite-label rows across all four ablations",
        "feature_categories": {
            "future_known_deterministic": list(feature_names()[:8]),
            "historical_observed": list(feature_names()[8:]),
            "future_forecast_covariates": list(feature_names(chosen)[13:]),
        },
    }
    write_json(output / "selection.json", selection)
    pd.DataFrame(timings).to_csv(output / "training_compute.csv", index=False)
    write_json(
        output / "training.json",
        {
            "fit_time": parent.fit_time,
            "config": asdict(parent.model),
            "paired_training_origins": int(complete.rows.issue_time.nunique()),
            "failed_weather_origins": len(complete.origin_failures),
            "finite_training_rows": len(complete.y),
            "requested_rows": complete.n_requested,
            "missing_labels": complete.n_missing_labels,
            "feature_names": {
                name: list(feature_names(cast(FeatureSet, name))) for name in config.ablations
            },
        },
    )
    final_payload = {
        "prefit_identity": prefit["identity"],
        "sources": source_hashes(root),
        "plan": plan(config, root),
        "selection": selection,
        "validation_sha256": sha256(output / "validation_ablation.csv"),
        "model_sha256": {
            m.model_id: sha256(output / f"model_{m.model_id}.txt") for m in models.values()
        },
    }
    freeze: dict[str, Any] = {
        "identity": cache_identity(final_payload),
        "payload": final_payload,
        "frozen_before_test_inference_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    write_json(output / "freeze.json", freeze)
    print("Validation ablation:\n" + ablation.to_string(index=False), flush=True)
    print("Frozen weather selection:", chosen, flush=True)
    for name in ("history", chosen):
        model = models[name]
        timed = TimedForecaster(model)
        start = time.perf_counter()
        result = backtest(
            model=timed,
            data=data,
            spec=parent.spec("test"),
            weather_provider=store if name != "history" else None,
        )
        calls = pd.DataFrame(timed.calls)
        compute = {
            "timing_source": "weather_lightgbm_v1 current CPU run",
            "device": "cpu",
            "load_seconds": 0.0,
            "backtest_wall_seconds": time.perf_counter() - start,
            **compute_summary(calls),
        }
        save_cache(
            output,
            freeze["identity"],
            model.model_id,
            result.predictions,
            result.origins,
            calls,
            compute,
        )
    # Reuse frozen baselines/TSFMs exactly, with separate original-compute provenance.
    for predictions, compute in comparators:
        model_id = str(compute["model_id"])
        if model_id == "lightgbm":
            new, _, _, _ = load_cache(output, freeze["identity"], "lightgbm")
            error = np.max(
                np.abs(
                    new.prediction_mw.to_numpy(float) - predictions.prediction_mw.to_numpy(float)
                )
            )
            write_json(
                output / "history_regression.json",
                {
                    "identical_training_support": complete.origin_failures.empty,
                    "max_absolute_prediction_difference_mw": float(error),
                    "reason_if_different": (
                        "paired training rows exclude missing-NWP origins; parameters unchanged"
                    ),
                },
            )
            if complete.origin_failures.empty and error > 1e-9:
                raise ValueError(
                    "History-only predictions changed despite identical original training support"
                )
            predictions.to_csv(output / "original_history_predictions.csv", index=False)
            continue
        predictions.to_csv(output / f"predictions_{model_id}.csv", index=False)
        write_json(output / f"reused_{model_id}.json", compute)


def check_analysis_sources(
    frozen: dict[str, str], current: dict[str, str], output: Path, inference_identity: str
) -> None:
    """Allow only a separately authenticated reporting-script revision of cached inference.

    Model/data/feature/config sources must remain byte-identical. The original
    inference freeze and its source archive are never rewritten.
    """
    if current == frozen:
        return
    changed = {name for name in set(frozen) | set(current) if frozen.get(name) != current.get(name)}
    amendment_path = output / "analysis_amendment.json"
    if changed != {"scripts/benchmark_weather_lightgbm.py"} or not amendment_path.exists():
        raise ValueError("Inference sources changed without an authenticated analysis-only record")
    amendment = read_json(amendment_path)
    if (
        amendment["inference_identity"] != inference_identity
        or amendment["original_sources"] != frozen
        or amendment["analysis_sources"] != current
        or amendment["refit_or_reinference"] is not False
        or sha256(output / "analysis_source_snapshot.zip") != amendment["analysis_archive_sha256"]
        or any(
            sha256(output / name) != expected
            for name, expected in amendment["prediction_files"].items()
        )
    ):
        raise ValueError("Analysis-only source revision identity mismatch")


def load_frames(config: WeatherExperimentConfig, root: Path) -> tuple[list[pd.DataFrame], str]:
    output = root / config.output_dir
    freeze, selection = read_json(output / "freeze.json"), read_json(output / "selection.json")
    check_analysis_sources(
        freeze["payload"]["sources"], source_hashes(root), output, freeze["identity"]
    )
    if (
        freeze["identity"] != cache_identity(freeze["payload"])
        or freeze["payload"]["plan"] != plan(config, root)
        or freeze["payload"]["selection"] != selection
        or freeze["payload"]["validation_sha256"] != sha256(output / "validation_ablation.csv")
    ):
        raise ValueError("Final experiment source/config/selection identity mismatch")
    for model_id, expected_hash in freeze["payload"]["model_sha256"].items():
        if sha256(output / f"model_{model_id}.txt") != expected_hash:
            raise ValueError("Frozen fitted model artifact checksum mismatch")
    prefit = read_json(output / "pre_fit_freeze.json")
    if (
        prefit["identity"] != cache_identity(prefit["payload"])
        or freeze["payload"]["prefit_identity"] != prefit["identity"]
    ):
        raise ValueError("Pre-fit source/data/environment identity mismatch")
    metadata = read_json(output / "metadata.json")
    if sha256(output / "source_snapshot.zip") != metadata["source_archive_sha256"]:
        raise ValueError("Frozen source archive checksum mismatch")
    inventory = weather_inventory(config, root)
    inventory.to_csv(output / "weather_availability_verified.csv", index=False)
    if (
        sha256(output / "weather_availability_verified.csv")
        != prefit["payload"]["weather_availability_sha256"]
    ):
        raise ValueError("Prepared weather inventory changed since fitting")
    comparators = cached_comparators(config, root)
    cached = {str(pred.model_id.iloc[0]): pred for pred, _ in comparators}
    frames = [cached[config.reference_model]]
    for name in ("lightgbm", selection["model_id"]):
        pred, _, _, _ = load_cache(output, freeze["identity"], name)
        expected = pd.MultiIndex.from_frame(
            cached[config.reference_model][["issue_time", "target_time"]]
        )
        if not pd.MultiIndex.from_frame(pred[["issue_time", "target_time"]]).equals(expected):
            raise ValueError("Current prediction cache schedule differs from original")
        frames.append(pred)
    frames.extend([cached["t0"], cached["chronos"]])
    return frames, str(selection["model_id"])


def evaluate(config: WeatherExperimentConfig, root: Path) -> None:
    output = root / config.output_dir
    frames, weather_id = load_frames(config, root)
    primary = [frame for frame in frames if frame.model_id.iloc[0] != "chronos"]
    scores = compare_models(primary, reference_model=config.reference_model)
    daytime = compare_models(
        primary,
        reference_model=config.reference_model,
        subset=daylight,
        subset_name="reference_point_daylight",
    )
    if scores.summary.n_scored.eq(0).any():
        raise ValueError("No primary common support; cannot publish a leaderboard")
    scores.leaderboard().to_csv(output / "leaderboard.csv", index=False)
    daytime.leaderboard().to_csv(output / "daylight_leaderboard.csv", index=False)
    scores.summary.to_csv(output / "availability.csv", index=False)
    compare_models(frames, reference_model=config.reference_model).leaderboard().to_csv(
        output / "extended_leaderboard.csv", index=False
    )
    scores.aligned_predictions.to_csv(output / "aligned_predictions.csv", index=False)
    for by in ("horizon", "local_hour", "month", "season"):
        scores.grouped(by).to_csv(output / f"by_{by}.csv", index=False)
    grouped_diagnostics(scores.aligned_predictions, config.reference_model, "horizon_6h").to_csv(
        output / "horizon_metrics.csv", index=False
    )
    daily = grouped_diagnostics(scores.aligned_predictions, config.reference_model, "local_date")
    daily.to_csv(output / "daily_metrics.csv", index=False)
    detail, wins = daily_wins(daily, 1e-9)
    detail.to_csv(output / "daily_winners.csv", index=False)
    wins.to_csv(output / "daily_win_summary.csv", index=False)
    error_correlations(scores.aligned_predictions).to_csv(output / "error_correlations.csv")
    error_correlations(daytime.aligned_predictions).to_csv(
        output / "daylight_error_correlations.csv"
    )
    daily.pivot(index="local_date", columns="model_id", values="rmse_mw").corr().to_csv(
        output / "daily_rmse_correlations.csv"
    )
    summary = scores.summary.set_index("model_id")
    changes = []
    for first, second in (("lightgbm", weather_id), ("lightgbm", "t0"), ("t0", weather_id)):
        before = summary.loc[[first]].to_dict(orient="records")[0]
        after = summary.loc[[second]].to_dict(orient="records")[0]
        changes.append(
            {
                "from": first,
                "to": second,
                "mae_change_mw": float(after["mae_mw"] - before["mae_mw"]),
                "mae_relative_change_percent": float(
                    100 * (after["mae_mw"] / before["mae_mw"] - 1)
                ),
                "rmse_change_mw": float(after["rmse_mw"] - before["rmse_mw"]),
                "rmse_relative_change_percent": float(
                    100 * (after["rmse_mw"] / before["rmse_mw"] - 1)
                ),
                "rmse_skill_change": float(after["rmse_skill"] - before["rmse_skill"]),
            }
        )
    pd.DataFrame(changes).to_csv(output / "information_value.csv", index=False)
    store = PreparedWeather(root, config)
    features = []
    for issue in config.parent(root).origins("test"):
        try:
            targets = config.parent(root).spec("test").target_policy(issue).as_unit("ns")
            weather = store.for_issue(issue, targets)
        except WeatherError:
            continue
        features.append(weather.features.assign(issue_time=issue))
    weather_table = pd.concat(features, ignore_index=True)
    weather_table.to_csv(output / "test_weather_features.csv", index=False)
    regime = weather_table[["issue_time", "target_time"]].copy()
    regime["cloud_group"] = pd.cut(
        weather_table.nwp_cloud_belgium_mean_fraction,
        [-np.inf, 0.25, 0.75, np.inf],
        right=False,
        labels=["low_cloud", "medium_cloud", "high_cloud"],
    ).astype(str)
    regime["spatial_variability"] = np.where(
        weather_table.nwp_cloud_belgium_spatial_std_fraction >= 0.15,
        "spatially_variable",
        "spatially_uniform",
    )
    joined = scores.aligned_predictions.merge(
        regime, on=["issue_time", "target_time"], validate="many_to_one"
    )
    regime_tables = []
    for column in ("cloud_group", "spatial_variability"):
        for label, group in joined.groupby(column, sort=True):
            for light_only in (False, True):
                subset = group.loc[daylight(group)] if light_only else group
                if subset.empty:
                    continue
                comparison = compare_models(
                    [rows for _, rows in subset.groupby("model_id")],
                    reference_model=config.reference_model,
                ).summary
                comparison.insert(0, "regime", label)
                comparison.insert(0, "regime_variable", column)
                comparison["daylight_only"] = light_only
                regime_tables.append(comparison)
    pd.concat(regime_tables, ignore_index=True).to_csv(output / "weather_regimes.csv", index=False)
    weather_daylight = weather_table.loc[daylight(weather_table)].copy()
    weather_daylight["local_date"] = weather_daylight.target_time.dt.tz_convert(
        "Europe/Brussels"
    ).dt.strftime("%Y-%m-%d")
    allowed_days = set(daily.local_date)
    difficulty = (
        weather_daylight.loc[weather_daylight.local_date.isin(allowed_days)]
        .groupby("local_date")
        .nwp_cloud_belgium_spatial_std_fraction.mean()
    )
    difficult = str(difficulty.idxmax())
    write_json(
        output / "selected_cases.json",
        {
            "representative": config.parent(root).representative_days,
            "difficult": difficult,
            "criterion": config.difficult_rule,
            "forecast_cloud_spatial_std_mean": float(difficulty.max()),
        },
    )
    compute = []
    freeze = read_json(output / "freeze.json")
    for model_id in ("lightgbm", weather_id):
        _, _, _, timing = load_cache(output, freeze["identity"], model_id)
        compute.append({"model_id": model_id, **timing})
    for name in (config.reference_model, "t0", "chronos"):
        compute.append(read_json(output / f"reused_{name}.json"))
    pd.DataFrame(compute).to_csv(output / "compute_summary.csv", index=False)
    inventory = weather_inventory(config, root)
    write_json(
        output / "data_overhead.json",
        {
            "scheduled_runs": len(inventory),
            "successful_runs": int(inventory.status.eq("ok").sum()),
            "accounted_completed_run_source_bytes": int(inventory.transferred_bytes.sum()),
            "native_cache_bytes_for_planned_runs": int(
                inventory.native_cache_bytes.fillna(0).sum()
            ),
            "prepared_feature_cache_bytes": sum(
                p.stat().st_size
                for p in (root / config.weather_feature_dir / plan(config, root)["identity"]).rglob(
                    "*"
                )
                if p.is_file()
            ),
            "sum_origin_elapsed_seconds": float(inventory.elapsed_seconds.sum()),
            "preparation_wall_seconds": read_json(output / "preparation.json")["wall_seconds"],
            "workers": config.workers,
            "note": (
                "Completed-bundle bytes; partial failed transfers and initial strict-QC "
                "probe overhead are additional."
            ),
        },
    )
    plots(output, config, weather_id, difficult)
    write_json(
        output / "results_manifest.json",
        {
            "identity": freeze["identity"],
            "evaluated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "files": {
                p.name: sha256(p)
                for p in sorted(output.iterdir())
                if p.is_file() and p.name != "results_manifest.json"
            },
        },
    )
    print(scores.leaderboard().to_string(index=False), flush=True)
    print("DAYLIGHT\n" + daytime.leaderboard().to_string(index=False), flush=True)


def plots(output: Path, config: WeatherExperimentConfig, weather_id: str, difficult: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    frame = pd.read_csv(output / "aligned_predictions.csv")
    frame["target_time"] = pd.to_datetime(frame.target_time, utc=True)
    local = frame.target_time.dt.tz_convert("Europe/Brussels")
    labels = local.dt.strftime("%Y-%m-%d")
    models = [config.reference_model, "lightgbm", "t0", weather_id]
    for filename, days in (
        ("representative_periods.png", config.parent(Path.cwd()).representative_days),
        ("difficult_weather_day.png", [difficult]),
    ):
        figure, axes = plt.subplots(len(days), 1, figsize=(12, 3.5 * len(days)), squeeze=False)
        for ax, day in zip(axes[:, 0], days, strict=True):
            selected = frame.loc[labels.eq(day)]
            if selected.empty:
                ax.set_title(f"{day}: no common weather/model support")
                continue
            actual = selected.drop_duplicates("target_time").sort_values("target_time")
            ax.plot(
                actual.target_time, actual.actual_mw, color="black", lw=2, label="Actual Elia PV"
            )
            for model_id in models:
                rows = selected.loc[selected.model_id.eq(model_id)].sort_values("target_time")
                ax.plot(rows.target_time, rows.prediction_mw, lw=1.2, label=model_id)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=local.dt.tz))  # type: ignore[no-untyped-call]
            ax.set(title=day, ylabel="PV power (MW)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=2)
        axes[-1, 0].set_xlabel("Belgian local time (Europe/Brussels)")
        figure.suptitle(
            "Development comparison: raw forecasts on common support — Elia Transmission Belgium SA"
        )
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/weather_lightgbm_v1.toml")
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--prepared", action="store_true", help="use already prepared weather; no network"
    )
    args = parser.parse_args()
    root = Path.cwd()
    config = WeatherExperimentConfig.load(args.config, root)
    output = root / config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frozen_plan = plan(config, root)
    if (output / "plan.json").exists():
        if read_json(output / "plan.json") != frozen_plan:
            raise ValueError("Existing experiment plan differs; no automatic overwrite")
    else:
        write_json(output / "plan.json", frozen_plan)
    frozen_config = output / "config.toml"
    if frozen_config.exists() and frozen_config.read_bytes() != args.config.read_bytes():
        raise ValueError("Frozen configuration text differs; inspect rather than overwrite")
    if not frozen_config.exists():
        frozen_config.write_bytes(args.config.read_bytes())
    if args.evaluate_only:
        evaluate(config, root)
        return
    if not args.prepared:
        prepare(config, root)
    if args.prepare_only:
        return
    run_models(config, root)
    evaluate(config, root)


if __name__ == "__main__":
    main()
