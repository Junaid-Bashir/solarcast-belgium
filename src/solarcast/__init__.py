"""SolarCast Belgium: a foundation for reproducible PV forecasting research."""

from importlib.metadata import version

__version__: str = version("solarcast-belgium")

from solarcast.api import BenchmarkConfig, ModelInfo, available_models, benchmark  # noqa: E402
from solarcast.evaluation import BenchmarkResult  # noqa: E402

__all__ = [
    "__version__",
    "benchmark",
    "available_models",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ModelInfo",
]
