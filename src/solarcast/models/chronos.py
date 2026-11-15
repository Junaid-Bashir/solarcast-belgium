"""Optional, univariate Chronos-2 inference with explicit UTC bridge alignment."""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from solarcast.models import Forecast, ForecastContext, ForecastDataError
from solarcast.models._tsfm import bridge, context_grid

logger = logging.getLogger(__name__)
CHECKPOINT = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


@dataclass(frozen=True)
class ChronosConfig:
    """Maximum elapsed quarter-hour context slots, not a required observed-row count."""

    context_length: int = 1344
    device: str = "cpu"
    revision: str = REVISION
    cache_dir: Path | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_length, bool)
            or not isinstance(self.context_length, int)
            or not 1 <= self.context_length <= 8192
        ):
            raise ValueError("context_length must be an integer in [1, 8192]")
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", self.device):
            raise ValueError("device must be cpu, cuda or cuda:N; no implicit device selection")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be an explicit 40-character Hub commit hash")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be boolean")


class _Tensor(Protocol):
    def detach(self) -> "_Tensor": ...
    def cpu(self) -> "_Tensor": ...
    def numpy(self) -> npt.NDArray[np.float32]: ...


class _Pipeline(Protocol):
    @property
    def model_context_length(self) -> int: ...
    @property
    def model_prediction_length(self) -> int: ...
    def predict_quantiles(
        self,
        inputs: list[npt.NDArray[np.float32]],
        *,
        prediction_length: int,
        quantile_levels: list[float],
        context_length: int,
        batch_size: int,
        cross_learning: bool,
        limit_prediction_length: bool,
    ) -> tuple[list[_Tensor], list[_Tensor]]: ...


@dataclass(frozen=True)
class _Loaded:
    pipeline: _Pipeline
    cache_seconds: float
    load_seconds: float


@dataclass(frozen=True)
class _Runtime:
    library_version: str
    load: Callable[[], _Loaded]
    synchronize: Callable[[], None]


def _runtime(config: ChronosConfig) -> _Runtime:
    """The only heavy import boundary; construction checks dependencies, not weights."""
    try:
        import torch
        from chronos import Chronos2Pipeline
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            'Chronos-2 requires optional support: pip install -e ".[dev,chronos]" '
            "from this repository. Install a platform-appropriate PyTorch CPU/CUDA wheel."
        ) from error
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; choose device='cpu'")

    def synchronize() -> None:
        if config.device.startswith("cuda"):
            torch.cuda.synchronize(config.device)

    def load() -> _Loaded:
        start = time.perf_counter()
        # Download/cache lookup is timed separately from local model construction.
        snapshot = snapshot_download(
            repo_id=CHECKPOINT,
            revision=config.revision,
            cache_dir=None if config.cache_dir is None else str(config.cache_dir),
            local_files_only=config.local_files_only,
            allow_patterns=["config.json", "model.safetensors"],
        )
        cache_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pipeline = Chronos2Pipeline.from_pretrained(
            snapshot,
            device_map=config.device,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        pipeline.model.eval()
        synchronize()
        return _Loaded(cast(_Pipeline, pipeline), cache_seconds, time.perf_counter() - start)

    return _Runtime(version("chronos-forecasting"), load, synchronize)


@dataclass(frozen=True)
class ChronosMetadata:
    checkpoint: str
    revision: str
    library_version: str
    context_length: int
    device: str
    model_id: str = "chronos"
    frequency: str = "15min UTC interval starts"
    dtype: str = "float32"
    point_policy: str = "native quantile 0.5 (P50)"


@dataclass(frozen=True)
class ChronosInference:
    issue_time: pd.Timestamp
    history_cutoff: pd.Timestamp
    first_context_time: pd.Timestamp
    context_slots: int
    observed_slots: int
    missing_slots: int
    rollout_steps: int
    requested_targets: int
    inference_seconds: float


class ChronosForecaster:
    """Reuse one frozen pretrained pipeline. No fit, covariates or hidden data access."""

    def __init__(self, config: ChronosConfig = ChronosConfig()) -> None:
        self._config = config
        self._runtime = _runtime(config)
        self._pipeline: _Pipeline | None = None
        self._last_inference: ChronosInference | None = None
        self._cache_seconds: float | None = None
        self._load_seconds: float | None = None

    @property
    def config(self) -> ChronosConfig:
        return self._config

    @property
    def model_id(self) -> str:
        return "chronos"

    @property
    def metadata(self) -> ChronosMetadata:
        return ChronosMetadata(
            CHECKPOINT,
            self.config.revision,
            self._runtime.library_version,
            self.config.context_length,
            self.config.device,
        )

    @property
    def last_inference(self) -> ChronosInference | None:
        """Last successful call only; reset before every attempted prediction."""
        return self._last_inference

    @property
    def cache_seconds(self) -> float | None:
        """Hub cache resolution/download, excluded from inference time."""
        return self._cache_seconds

    @property
    def load_seconds(self) -> float | None:
        """Local pipeline construction/device transfer, excluded from inference time."""
        return self._load_seconds

    def load(self) -> None:
        """Explicit optional warm-up. First predict also loads; later calls reuse weights."""
        if self._pipeline is not None:
            return
        try:
            loaded = self._runtime.load()
            if loaded.pipeline.model_context_length < self.config.context_length:
                raise ValueError(
                    "Checkpoint context limit is smaller than configured context_length"
                )
            if loaded.pipeline.model_prediction_length < 1:
                raise ValueError("Checkpoint prediction limit must be positive")
        except Exception as error:
            raise RuntimeError(
                f"Cannot load {CHECKPOINT}@{self.config.revision}: {error}"
            ) from error
        self._pipeline = loaded.pipeline
        self._cache_seconds, self._load_seconds = loaded.cache_seconds, loaded.load_seconds

    def predict(self, context: ForecastContext) -> Forecast:
        self._last_inference = None
        cutoff = context.history_cutoff
        prepared = context_grid(context, self.config.context_length, "Chronos-2")
        grid, values = prepared.times, prepared.values
        targets = context.request.target_times
        offsets, steps = bridge(context, 1024, "Chronos-2")
        self.load()
        assert self._pipeline is not None
        if steps > self._pipeline.model_prediction_length:
            raise ForecastDataError("Requested bridge exceeds this checkpoint's prediction limit")
        try:
            self._runtime.synchronize()
            start = time.perf_counter()
            quantiles, _ = self._pipeline.predict_quantiles(
                [values],
                prediction_length=steps,
                quantile_levels=[0.5],
                context_length=len(values),
                batch_size=1,
                cross_learning=False,
                limit_prediction_length=True,
            )
            self._runtime.synchronize()
            if len(quantiles) != 1:
                raise ValueError("Expected exactly one upstream series")
            raw = np.asarray(quantiles[0].detach().cpu().numpy())
            seconds = time.perf_counter() - start
            if raw.shape != (1, steps, 1):
                raise ValueError(f"Expected quantiles shape (1, {steps}, 1), got {raw.shape}")
            if not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
                raise ValueError("Upstream P50 must be floating and finite over the entire bridge")
            forecast = Forecast(
                context.request,
                pd.DataFrame({"target_time": targets, "prediction_mw": raw[0, offsets, 0]}),
                model_id=self.model_id,
                model_version=self._runtime.library_version,
            )
        except Exception as error:
            raise RuntimeError(
                f"Chronos-2 inference at {context.request.issue_time}, cutoff {cutoff}, "
                f"context {len(values)} slots, rollout {steps}: {error}"
            ) from error
        missing = int(np.isnan(values).sum())
        self._last_inference = ChronosInference(
            context.request.issue_time,
            cutoff,
            grid[0],
            len(values),
            len(values) - missing,
            missing,
            steps,
            len(targets),
            seconds,
        )
        logger.info("Chronos-2 inference: %s", self._last_inference)
        return forecast
