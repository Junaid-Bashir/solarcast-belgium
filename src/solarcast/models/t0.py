"""Optional history-only t0-gamma inference; native P50 with raw adjustments off."""

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
CHECKPOINT = "theforecastingcompany/t0-gamma"
REVISION = "fef67abad3eed4f227c7b536a6e8ee4e9ab1080d"


@dataclass(frozen=True)
class T0Config:
    """Elapsed slot cap; 8192 is a SolarCast guard, not a verified gamma hard limit."""

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
            raise ValueError("context_length must be an integer in [1, 8192] (SolarCast guard)")
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


class _Output(Protocol):
    @property
    def quantiles(self) -> _Tensor: ...


class _Model(Protocol):
    @property
    def max_horizon(self) -> int: ...
    def predict(
        self,
        model_input: npt.NDArray[np.float32],
        horizon: int,
        quantile_levels: list[float],
        *,
        context_length: int,
        trim_leading_missing: bool,
        symmetric_averaging: bool,
        clip_nonnegative: bool,
        snap_integers: bool,
        drop_constant_covariates: bool,
    ) -> _Output: ...


@dataclass(frozen=True)
class _Loaded:
    model: _Model
    cache_seconds: float
    load_seconds: float


@dataclass(frozen=True)
class _Runtime:
    library_version: str
    load: Callable[[], _Loaded]
    synchronize: Callable[[], None]


def _runtime(config: T0Config) -> _Runtime:
    try:
        import torch
        from huggingface_hub import snapshot_download
        from t0 import T0Forecaster as NativeForecaster
    except ImportError as error:
        raise ImportError(
            't0-gamma requires optional support: pip install -e ".[dev,t0]" '
            "from this repository. Install a platform-appropriate PyTorch CPU/CUDA wheel."
        ) from error
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; choose device='cpu'")

    def synchronize() -> None:
        if config.device.startswith("cuda"):
            torch.cuda.synchronize(config.device)

    def load() -> _Loaded:
        start = time.perf_counter()
        snapshot = snapshot_download(
            repo_id=CHECKPOINT,
            revision=config.revision,
            cache_dir=None if config.cache_dir is None else str(config.cache_dir),
            local_files_only=config.local_files_only,
            allow_patterns=["config.json", "model.safetensors"],
        )
        cache_seconds = time.perf_counter() - start
        start = time.perf_counter()
        # Hub mixin loads the local snapshot; no second unpinned network lookup.
        model = (
            NativeForecaster.from_pretrained(
                snapshot, map_location="cpu", dtype=torch.float32, local_files_only=True
            )
            .to(device=config.device, dtype=torch.float32)
            .eval()
        )
        model.requires_grad_(False)
        synchronize()
        return _Loaded(cast(_Model, model), cache_seconds, time.perf_counter() - start)

    return _Runtime(version("tfc-t0"), load, synchronize)


@dataclass(frozen=True)
class T0Metadata:
    checkpoint: str
    revision: str
    library_version: str
    context_length: int
    device: str
    model_id: str = "t0"
    frequency: str = "15min UTC interval starts"
    dtype: str = "float32"
    point_policy: str = "native quantile 0.5 (P50)"
    trim_leading_missing: bool = False
    symmetric_averaging: bool = False
    clip_nonnegative: bool = False
    snap_integers: bool = False
    drop_constant_covariates: bool = False


@dataclass(frozen=True)
class T0Inference:
    issue_time: pd.Timestamp
    history_cutoff: pd.Timestamp
    first_context_time: pd.Timestamp
    context_slots: int
    observed_slots: int
    missing_slots: int
    rollout_steps: int
    requested_targets: int
    inference_seconds: float


class T0Forecaster:
    """One reusable frozen checkpoint; no fitting, covariates or external history."""

    def __init__(self, config: T0Config = T0Config()) -> None:
        self._config = config
        self._runtime = _runtime(config)
        self._model: _Model | None = None
        self._last_inference: T0Inference | None = None
        self._cache_seconds: float | None = None
        self._load_seconds: float | None = None

    @property
    def config(self) -> T0Config:
        return self._config

    @property
    def model_id(self) -> str:
        return "t0"

    @property
    def metadata(self) -> T0Metadata:
        return T0Metadata(
            CHECKPOINT,
            self.config.revision,
            self._runtime.library_version,
            self.config.context_length,
            self.config.device,
        )

    @property
    def last_inference(self) -> T0Inference | None:
        """Last successful call only; reset before each attempt."""
        return self._last_inference

    @property
    def cache_seconds(self) -> float | None:
        return self._cache_seconds

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            loaded = self._runtime.load()
            if loaded.model.max_horizon < 1:
                raise ValueError("Checkpoint prediction limit must be positive")
        except Exception as error:
            raise RuntimeError(
                f"Cannot load {CHECKPOINT}@{self.config.revision}: {error}"
            ) from error
        self._model = loaded.model
        self._cache_seconds, self._load_seconds = loaded.cache_seconds, loaded.load_seconds

    def predict(self, context: ForecastContext) -> Forecast:
        self._last_inference = None
        prepared = context_grid(context, self.config.context_length, "t0-gamma")
        grid, values = prepared.times, prepared.values
        offsets, steps = bridge(context, 1024, "t0-gamma")
        self.load()
        assert self._model is not None
        if steps > self._model.max_horizon:
            raise ForecastDataError("Requested bridge exceeds this checkpoint's single-pass limit")
        try:
            self._runtime.synchronize()
            start = time.perf_counter()
            # [batch=1, elapsed slots]; NaNs are native missing observations, not PAD.
            output = self._model.predict(
                values[None, :],
                horizon=steps,
                quantile_levels=[0.5],
                context_length=len(values),
                trim_leading_missing=False,
                symmetric_averaging=False,
                clip_nonnegative=False,
                snap_integers=False,
                drop_constant_covariates=False,
            )
            self._runtime.synchronize()
            raw = np.asarray(output.quantiles.detach().cpu().numpy())
            seconds = time.perf_counter() - start
            if raw.shape != (1, steps, 1):
                raise ValueError(f"Expected quantiles shape (1, {steps}, 1), got {raw.shape}")
            if not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
                raise ValueError("Upstream P50 must be floating and finite over the entire bridge")
            forecast = Forecast(
                context.request,
                pd.DataFrame(
                    {
                        "target_time": context.request.target_times,
                        "prediction_mw": raw[0, offsets, 0],
                    }
                ),
                model_id=self.model_id,
                model_version=self._runtime.library_version,
            )
        except Exception as error:
            raise RuntimeError(
                f"t0-gamma inference at {context.request.issue_time}, "
                f"cutoff {context.history_cutoff}, "
                f"context {len(values)} slots, rollout {steps}: {error}"
            ) from error
        missing = int(np.isnan(values).sum())
        self._last_inference = T0Inference(
            context.request.issue_time,
            context.history_cutoff,
            grid[0],
            len(values),
            len(values) - missing,
            missing,
            steps,
            len(context.request.target_times),
            seconds,
        )
        logger.info("t0-gamma inference: %s", self._last_inference)
        return forecast
