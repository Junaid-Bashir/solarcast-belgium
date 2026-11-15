"""Small offline orchestration layer over SolarCast's research components."""

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import pandas as pd

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, _prepare_data, backtest
from solarcast.data.weather import WeatherStore
from solarcast.evaluation import BenchmarkResult, compare_models
from solarcast.evaluation.benchmark import PrimaryMetric
from solarcast.features import solar_features
from solarcast.models import (
    Forecaster,
    LastValuePersistenceForecaster,
    SeasonalPersistenceForecaster,
)
from solarcast.models.chronos import ChronosConfig, ChronosForecaster
from solarcast.models.lightgbm import LightGBMConfig, LightGBMForecaster
from solarcast.models.lightgbm_data import FeatureSet, build_training_dataset
from solarcast.models.t0 import T0Config, T0Forecaster
from solarcast.models.types import _aware_time

logger = logging.getLogger(__name__)
_MODELS = {
    "last_value_persistence": (None, ()),
    "seasonal_persistence_48h": (None, ()),
    "seasonal_persistence_7d": (None, ()),
    "lightgbm": ("lightgbm", ("lightgbm",)),
    "lightgbm_history_radiation_cloud": ("lightgbm", ("lightgbm",)),
    "t0": ("t0", ("tfc-t0", "torch", "huggingface-hub")),
    "chronos": ("chronos", ("chronos-forecasting", "torch", "huggingface-hub")),
}


@dataclass(frozen=True)
class ModelInfo:
    """Dependency discovery, not a guarantee of runtime/device/cache readiness.

    ``status`` is built_in, installed, or unavailable. ``missing_dependencies``
    names distributions absent from this interpreter; discovery imports no ML runtime.
    ``extra`` identifies the repository installation extra, if any.
    """

    model_id: str
    status: Literal["built_in", "installed", "unavailable"]
    extra: str | None
    missing_dependencies: tuple[str, ...]
    requires_training: bool
    requires_weather: bool


def available_models() -> tuple[ModelInfo, ...]:
    """Discover implemented identifiers without downloads or heavy imports.

    Returns dependency-level ModelInfo records, including unavailable adapters.
    An installed distribution may still have incompatible binaries or dependencies.
    Example: ``[m.model_id for m in available_models() if m.status == 'built_in']``.
    """
    records = []
    for name, (extra, dependencies) in _MODELS.items():
        missing = []
        for dependency in dependencies:
            try:
                version(dependency)
            except PackageNotFoundError:
                missing.append(dependency)
        records.append(
            ModelInfo(
                name,
                "built_in" if extra is None else "unavailable" if missing else "installed",
                extra,
                tuple(missing),
                name.startswith("lightgbm"),
                name == "lightgbm_history_radiation_cloud",
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Explicit day-ahead schedule and information assumptions.

    Parameters
    ----------
    issue_times : pandas.DatetimeIndex
        Ordered, unique, timezone-aware evaluation issues. Targets are the next
        Brussels calendar day, including DST changes.
    observation_safety_lag : str
        Required elapsed lag for EventTimeCutoff. This is an approximation, not
        verified publication availability or revision control.
    context_window : str or None
        Issue-anchored elapsed history window; None retains all eligible history.
    training_issue_times, fit_time
        Required for LightGBM. Training target intervals must finish by the
        fit-time label cutoff, and fit_time must not follow the first test issue.
    primary_metric, reference_model, subset, on_error
        Ranking, explicit RMSE-skill reference, scoring subset and failure policy.
        Default scoring includes night. Continue retains failed-origin rows.
    lightgbm, t0, chronos
        Existing adapter configurations. TSFM defaults prohibit weight downloads;
        explicitly set local_files_only=False to allow pinned Hub acquisition.

    Invalid schedules/durations raise ValueError. No training or I/O occurs here.
    Example: ``BenchmarkConfig(issues, observation_safety_lag='30min')``.
    """

    issue_times: pd.DatetimeIndex
    observation_safety_lag: str
    context_window: str | None = "14D"
    primary_metric: PrimaryMetric = "rmse"
    reference_model: str | None = None
    subset: Literal["all_intervals", "reference_point_daylight"] = "all_intervals"
    on_error: Literal["strict", "continue"] = "continue"
    training_issue_times: pd.DatetimeIndex | None = None
    fit_time: str | None = None
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    t0: T0Config = field(default_factory=lambda: T0Config(local_files_only=True))
    chronos: ChronosConfig = field(default_factory=lambda: ChronosConfig(local_files_only=True))

    def __post_init__(self) -> None:
        spec = self.spec()
        object.__setattr__(self, "issue_times", spec.issue_times)
        if self.primary_metric not in ("rmse", "mae"):
            raise ValueError("primary_metric must be rmse or mae")
        if self.subset not in ("all_intervals", "reference_point_daylight"):
            raise ValueError("subset must be all_intervals or reference_point_daylight")
        if (self.training_issue_times is None) != (self.fit_time is None):
            raise ValueError("Provide both training_issue_times and timezone-aware fit_time")
        if self.training_issue_times is not None:
            train = self.spec(training=True)
            assert self.fit_time is not None
            fit = _aware_time(self.fit_time, "fit_time")
            if train.issue_times[-1] >= fit or fit > spec.issue_times[0]:
                raise ValueError("Require training issues < fit_time <= first evaluation issue")
            object.__setattr__(self, "training_issue_times", train.issue_times)
            object.__setattr__(self, "fit_time", fit.isoformat())

    def spec(self, *, training: bool = False) -> BacktestSpec:
        """Return the existing research specification with the same cutoff policy."""
        issues = self.training_issue_times if training else self.issue_times
        if issues is None:
            raise ValueError("LightGBM requires training_issue_times and fit_time")
        return BacktestSpec(
            issues,
            availability_policy=EventTimeCutoff(self.observation_safety_lag),
            context_window=self.context_window,
            on_error=self.on_error,
        )


def _daylight(keys: pd.DataFrame) -> pd.Series:
    return pd.Series(
        solar_features(pd.DatetimeIndex(keys.target_time)).is_daylight.to_numpy(), index=keys.index
    )


def benchmark(
    *,
    data: pd.DataFrame,
    config: BenchmarkConfig,
    models: Sequence[str] = ("seasonal_persistence_48h",),
    target: str = "belgium_pv",
    data_source: str = "caller-supplied snapshot; provenance not independently verified",
    weather: WeatherStore | None = None,
) -> BenchmarkResult:
    """Run explicit local day-ahead benchmarks using existing adapters and scoring.

    ``data`` must contain unique aware quarter-hour target_time and nullable real
    pv_generation_mw (national Belgium, MW). Supply warm-up history, training
    labels if needed, and evaluation actuals; missing values remain missing.
    ``config`` declares schedule, cutoff assumptions and optional pre-test fit.
    ``models`` contains identifiers from available_models(); no automatic winner
    selection occurs. ``data_source`` is a caller assertion recorded in metadata.
    ``weather`` must be a prepared local WeatherStore for the NWP adapter.

    Returns the existing BenchmarkResult, augmented with complete predictions,
    origins, effective config and metadata. Leaderboards score raw predictions on
    common finite issue/target pairs, with availability/failure accounting.
    Nothing is saved, PV/NWP is never fetched, and TSFM downloads require explicit
    opt-in via adapter config. LightGBM is trained once when requested.

    Raises ValueError for invalid targets/models/data/prerequisites; ImportError
    for absent optional support. Strict origin failures retain their cause via
    BacktestOriginError; continue mode records failures in results.origins.

    Example: ``benchmark(data=pv_data, config=config).leaderboard()``.
    """
    if target != "belgium_pv":
        raise ValueError(f"Unsupported target {target!r}. Supported targets: belgium_pv")
    if isinstance(models, str) or not models or len(set(models)) != len(models):
        raise ValueError("models must be a nonempty sequence of unique model identifiers")
    discovery = {info.model_id: info for info in available_models()}
    for name in models:
        if name not in discovery:
            raise ValueError(
                f"Unknown SolarCast model {name!r}. Available models: {', '.join(_MODELS)}"
            )
    if config.reference_model is not None and config.reference_model not in models:
        raise ValueError("reference_model must be one of the requested models")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("data_source must be a nonempty provenance description")
    needs_weather = any(discovery[name].requires_weather for name in models)
    if needs_weather and (not isinstance(weather, WeatherStore) or not weather.runs):
        raise ValueError(
            "Weather LightGBM requires weather=WeatherStore(...) with prepared historical "
            "operational GFS runs, spatial weights and an explicit availability policy. "
            "Prepare local runs as described in docs/data/weather.md; no automatic fetch/fallback."
        )
    if weather is not None and not needs_weather:
        raise ValueError("weather is only used by lightgbm_history_radiation_cloud")
    if any(discovery[name].requires_training for name in models):
        config.spec(training=True)
    for name in models:
        info = discovery[name]
        if info.missing_dependencies:
            raise ImportError(
                f"Model {name!r} requires optional support "
                f"({', '.join(info.missing_dependencies)}). "
                f'Install from the repository: python -m pip install -e ".[{info.extra}]"'
            )
    prepared = _prepare_data(data)
    # Canonical content identity covers the numerical snapshot, not external provenance.
    content = prepared[["target_time", "pv_generation_mw"]].to_csv(index=False, lineterminator="\n")
    metadata: dict[str, object] = {
        "target": target,
        "target_definition": "Elia national Belgium PV estimate (Total grid where applicable)",
        "unit": "MW",
        "models": list(models),
        "data_source": data_source,
        "data_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "data_rows": len(prepared),
        "data_first_target": prepared.target_time.min(),
        "data_last_target": prepared.target_time.max(),
        "solarcast_version": version("solarcast-belgium"),
        "numerical_versions": {name: version(name) for name in ("pandas", "numpy", "pvlib")},
        "evaluation_issue_times": config.issue_times.copy(),
        "availability": "event_time_cutoff approximation; snapshot revisions not reconstructed",
        "target_policy": "next Europe/Brussels calendar day, 15min UTC interval starts",
        "comparison_policy": "common_support; raw outputs; no clipping or imputation",
        "effective_config": asdict(config),
        "model_metadata": {},
    }
    model_metadata: dict[str, object] = {}
    runs = []
    for name in models:
        logger.info("SolarCast benchmark model: %s", name)
        model: Forecaster
        provider = weather if discovery[name].requires_weather else None
        if name == "last_value_persistence":
            model = LastValuePersistenceForecaster()
            model_metadata[name] = {"recipe": "latest eligible non-null value"}
        elif name.startswith("seasonal_persistence_"):
            lag = "48h" if name.endswith("48h") else "7D"
            model = SeasonalPersistenceForecaster(lag)
            model_metadata[name] = {"lag": lag, "policy": "exact elapsed UTC; no fallback"}
        elif name.startswith("lightgbm"):
            feature_set: FeatureSet = (
                "history+radiation+cloud" if provider is not None else "history"
            )
            trained = LightGBMForecaster(config.lightgbm, feature_set=feature_set)
            assert config.fit_time is not None
            training = build_training_dataset(
                prepared,
                config.spec(training=True),
                fit_time=config.fit_time,
                feature_set=feature_set,
                weather_provider=provider,
            )
            trained.fit(training)
            model_metadata[name] = {
                "training": asdict(trained.training_info),
                "n_requested": training.n_requested,
                "n_missing_labels": training.n_missing_labels,
                "origin_failures": training.origin_failures.copy(deep=True),
            }
            model = trained
        else:
            tsfm = T0Forecaster(config.t0) if name == "t0" else ChronosForecaster(config.chronos)
            model_metadata[name] = asdict(tsfm.metadata)
            model = tsfm
        runs.append(
            backtest(model=model, data=prepared, spec=config.spec(), weather_provider=provider)
        )
    metadata["model_metadata"] = model_metadata
    if weather is not None:
        metadata["weather"] = {
            "availability_policy": asdict(weather.policy),
            "availability_assumption": weather.policy.name,
            "radiation_policy": weather.radiation_policy,
            "weights_id": weather.weights.identity,
            "runs": [
                {
                    "reference_time": run.reference_time,
                    "version": run.version,
                    "observed_available_at": run.observed_available_at,
                    "version_available_at": run.version_available_at,
                    "revision": run.revision,
                    "metadata": run.metadata,
                    "native_data_sha256": hashlib.sha256(
                        run.data.to_csv(index=False, lineterminator="\n").encode()
                    ).hexdigest(),
                }
                for run in weather.runs
            ],
        }
    result = compare_models(
        runs,
        primary_metric=config.primary_metric,
        reference_model=config.reference_model,
        subset=_daylight if config.subset == "reference_point_daylight" else None,
        subset_name=config.subset if config.subset == "reference_point_daylight" else None,
    )
    return replace(
        result,
        config=replace(config),
        metadata=metadata,
        predictions=pd.concat([run.predictions for run in runs], ignore_index=True),
        origins=pd.concat([run.origins for run in runs], ignore_index=True),
    )
