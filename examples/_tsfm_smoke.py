"""Shared two-origin TSFM smoke mechanics; no benchmark or accuracy claims."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest
from solarcast.evaluation import compare_models
from solarcast.experiments import sha256, write_json
from solarcast.models import Forecast, ForecastContext, ForecastRequest
from solarcast.models.chronos import ChronosConfig, ChronosForecaster
from solarcast.models.t0 import T0Config, T0Forecaster

Model = ChronosForecaster | T0Forecaster


class Recorder:
    """Observe adapter diagnostics while the unchanged backtester drives inference."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.calls: list[dict[str, object]] = []

    @property
    def model_id(self) -> str:
        return self.model.model_id

    def predict(self, context: ForecastContext) -> Forecast:
        forecast = self.model.predict(context)
        if self.model.last_inference is not None:
            self.calls.append(asdict(self.model.last_inference))
        return forecast


def main(family: str) -> None:
    parser = argparse.ArgumentParser(description=f"{family} technical smoke; synthetic by default")
    parser.add_argument(
        "--input-csv", type=Path, help="Normalized local Elia data; never implicit retrieval"
    )
    parser.add_argument("--issue", default="2025-06-14T11:00:00+02:00")
    parser.add_argument("--context-length", type=int, default=1344)
    parser.add_argument("--context-window", default="14D")
    parser.add_argument("--safety-lag", default="30min")
    parser.add_argument("--device", default="cpu")
    acquisition = parser.add_mutually_exclusive_group()
    acquisition.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="use prepared pinned weights only (default)",
    )
    acquisition.add_argument(
        "--allow-download",
        dest="local_files_only",
        action="store_false",
        help="explicitly permit potentially large pinned checkpoint downloads",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Optional new directory for smoke artifacts"
    )
    args = parser.parse_args()
    issue = pd.Timestamp(args.issue)
    if pd.isna(issue) or issue.tzinfo is None:
        parser.error("--issue requires an explicit timezone/offset")
    issue = issue.tz_convert("UTC")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("--output-dir already exists; preserve the previous smoke record")
    if args.input_csv:
        data = pd.read_csv(args.input_csv)
        stamps = [pd.Timestamp(value) for value in data.target_time]
        if any(pd.isna(stamp) or stamp.tzinfo is None for stamp in stamps):
            parser.error("Input timestamps need explicit offsets")
        data["target_time"] = pd.to_datetime(stamps, utc=True)
        data["pv_generation_mw"] = pd.to_numeric(data.pv_generation_mw).astype("Float64")
        source = "Local normalized Elia snapshot; technical smoke only"
    else:
        times = pd.date_range(issue - pd.Timedelta("16D"), issue + pd.Timedelta("4D"), freq="15min")
        local = times.tz_convert("Europe/Brussels")
        hours = local.hour.to_numpy() + local.minute.to_numpy() / 60
        data = pd.DataFrame(
            {
                "target_time": times,
                "pv_generation_mw": np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) * 1000,
            }
        )
        source = "SYNTHETIC representative profile; not observed Belgian data"
    spec = BacktestSpec(
        pd.DatetimeIndex([issue, issue + pd.Timedelta("1D")]),
        availability_policy=EventTimeCutoff(args.safety_lag),
        context_window=args.context_window,
        on_error="continue",
    )
    selection = spec.availability_policy.select(data.copy(), issue)
    history = selection.history.loc[
        selection.history.target_time >= issue - pd.Timedelta(args.context_window)
    ]
    request = ForecastRequest(
        issue, pd.date_range(issue.ceil("15min") + pd.Timedelta("15min"), periods=4, freq="15min")
    )
    context = ForecastContext(request, history, history_cutoff=selection.cutoff)
    model: Model
    if family == "chronos":
        model = ChronosForecaster(
            ChronosConfig(
                context_length=args.context_length,
                device=args.device,
                local_files_only=args.local_files_only,
            )
        )
        runtime_package = "chronos-forecasting"
    elif family == "t0":
        model = T0Forecaster(
            T0Config(
                context_length=args.context_length,
                device=args.device,
                local_files_only=args.local_files_only,
            )
        )
        runtime_package = "tfc-t0"
    else:
        raise ValueError("Unknown smoke model family")
    model.load()
    forecast = model.predict(context)
    short_timing = model.last_inference
    # Two origins only. Reuses the standard backtester/evaluator and the loaded model.
    recorder = Recorder(model)
    result = backtest(model=recorder, data=data, spec=spec)
    summary = compare_models([result]).summary
    record = {
        "source": source,
        "input_csv": None if args.input_csv is None else str(args.input_csv),
        "input_sha256": None if args.input_csv is None else sha256(args.input_csv),
        "model": asdict(model.metadata),
        "cache_resolution_or_download_seconds": model.cache_seconds,
        "local_model_load_seconds": model.load_seconds,
        "short_forecast": asdict(short_timing) if short_timing is not None else None,
        "last_backtest_forecast": asdict(model.last_inference) if model.last_inference else None,
        "backtest_origins": len(result.origins),
        "failed_origins": int(result.origins.status.eq("forecast_failed").sum()),
        "successful_origin_diagnostics": recorder.calls,
        "successful_origins_with_missing_context": sum(
            int(cast(int, call["missing_slots"]) > 0) for call in recorder.calls
        ),
        "packages": {
            name: version(name)
            for name in (
                "solarcast-belgium",
                runtime_package,
                "torch",
                "numpy",
                "pandas",
            )
            + (("transformers",) if family == "chronos" else ())
        },
        "note": "Shape/integration smoke only; no scientific accuracy conclusion",
    }
    print(record)
    print(forecast.predictions.to_string(index=False))
    print(
        summary[
            ["model_id", "n_expected", "n_scored", "forecast_availability", "n_failed_origins"]
        ].to_string(index=False)
    )
    if args.output_dir:
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / "smoke.json", record)
        forecast.to_frame().to_csv(args.output_dir / "short_forecast.csv", index=False)
        result.predictions.to_csv(args.output_dir / "predictions.csv", index=False)
        result.origins.to_csv(args.output_dir / "origins.csv", index=False)
        summary.to_csv(args.output_dir / "evaluation.csv", index=False)
