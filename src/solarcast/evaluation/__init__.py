"""Deterministic metrics and transparent common-support benchmark summaries."""

from solarcast.evaluation.benchmark import BenchmarkResult, compare_models, evaluate_forecast
from solarcast.evaluation.metrics import bias, mae, rmse, rmse_skill

__all__ = [
    "BenchmarkResult",
    "bias",
    "compare_models",
    "evaluate_forecast",
    "mae",
    "rmse",
    "rmse_skill",
]
