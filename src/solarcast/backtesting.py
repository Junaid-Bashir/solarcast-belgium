"""Sequential rolling-origin evaluation with explicit, approximate availability."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal, Protocol

import pandas as pd

from solarcast.data.weather.types import WeatherCovariates
from solarcast.features._validation import pv_history, target_index
from solarcast.models import Forecast, ForecastContext, Forecaster, ForecastRequest
from solarcast.models.types import _aware_time

logger = logging.getLogger(__name__)
TargetPolicy = Callable[[pd.Timestamp], pd.DatetimeIndex]


class WeatherProvider(Protocol):
    """Engine-owned selection of numerical covariates from local weather runs."""

    def for_issue(
        self, issue_time: pd.Timestamp, targets: pd.DatetimeIndex
    ) -> WeatherCovariates: ...


def day_ahead_local_day(issue_time: pd.Timestamp) -> pd.DatetimeIndex:
    """Return the next Brussels local calendar day's quarter-hour starts in UTC."""
    issue = _aware_time(issue_time, "issue_time")
    local_date = issue.tz_convert("Europe/Brussels").date()
    # Localize each calendar boundary independently; never add 24 elapsed hours.
    first = pd.Timestamp(local_date + timedelta(days=1), tz="Europe/Brussels")
    stop = pd.Timestamp(local_date + timedelta(days=2), tz="Europe/Brussels")
    return pd.date_range(first, stop, freq="15min", inclusive="left").tz_convert("UTC")


def _duration(value: str | timedelta, name: str, *, allow_zero: bool) -> pd.Timedelta:
    if not isinstance(value, (str, timedelta)):
        raise ValueError(f"{name} must be an elapsed duration string or timedelta")
    try:
        duration = pd.Timedelta(value)
        if (
            pd.isna(duration)
            or duration < pd.Timedelta(0)
            or (not allow_zero and duration == pd.Timedelta(0))
        ):
            raise ValueError("Invalid duration")
        _ = duration.value  # Fail clearly if not representable at nanosecond resolution.
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        ) from error
    return duration


@dataclass(frozen=True)
class HistorySelection:
    """A trusted policy's selected observations and explicit temporal upper bound."""

    history: pd.DataFrame
    cutoff: pd.Timestamp


class AvailabilityPolicy(Protocol):
    """Select observations for an issue; future policies may use real availability metadata.

    Policies must not mutate input, generate features, train models or consult hidden
    data. The engine/context additionally enforce completion and cutoff <= issue.
    """

    @property
    def name(self) -> str: ...

    def select(self, data: pd.DataFrame, issue_time: pd.Timestamp) -> HistorySelection: ...


@dataclass(frozen=True, init=False)
class EventTimeCutoff:
    """Approximation: completed intervals by floor_15min(issue - safety_lag).

    Zero safety lag is explicit but is NOT a verified publication-latency policy.
    """

    safety_lag: pd.Timedelta

    def __init__(self, safety_lag: str | timedelta = "0min") -> None:
        object.__setattr__(self, "safety_lag", _duration(safety_lag, "safety_lag", allow_zero=True))

    @property
    def name(self) -> str:
        return "event_time_cutoff"

    def select(self, data: pd.DataFrame, issue_time: pd.Timestamp) -> HistorySelection:
        cutoff = (_aware_time(issue_time, "issue_time") - self.safety_lag).floor("15min")
        selected = data.loc[data["target_time"] + pd.Timedelta(minutes=15) <= cutoff].copy()
        return HistorySelection(selected, cutoff)


@dataclass(frozen=True, init=False, eq=False)
class BacktestSpec:
    """An ordered issue schedule, target/availability policies and optional elapsed window."""

    _issue_times: pd.DatetimeIndex = field(repr=False)
    availability_policy: AvailabilityPolicy
    target_policy: TargetPolicy
    context_window: pd.Timedelta | None
    on_error: Literal["strict", "continue"]

    def __init__(
        self,
        issue_times: pd.DatetimeIndex,
        *,
        availability_policy: AvailabilityPolicy,
        target_policy: TargetPolicy = day_ahead_local_day,
        context_window: str | timedelta | None = None,
        on_error: Literal["strict", "continue"] = "strict",
    ) -> None:
        if not isinstance(issue_times, pd.DatetimeIndex) or issue_times.tz is None:
            raise ValueError("issue_times must be a timezone-aware DatetimeIndex")
        issues = issue_times.tz_convert("UTC").as_unit("ns")
        if (
            issues.empty
            or issues.hasnans
            or not issues.is_unique
            or not issues.is_monotonic_increasing
        ):
            raise ValueError("issue_times must be nonempty, non-null, ordered and unique")
        if on_error not in ("strict", "continue"):
            raise ValueError("on_error must be 'strict' or 'continue'")
        window = (
            None
            if context_window is None
            else _duration(context_window, "context_window", allow_zero=False)
        )
        object.__setattr__(self, "_issue_times", issues.copy(deep=True))
        object.__setattr__(self, "availability_policy", availability_policy)
        object.__setattr__(self, "target_policy", target_policy)
        object.__setattr__(self, "context_window", window)
        object.__setattr__(self, "on_error", on_error)

    @property
    def issue_times(self) -> pd.DatetimeIndex:
        return self._issue_times.copy(deep=True)


@dataclass(frozen=True)
class BacktestResult:
    """Owned result tables plus the run specification; no evaluation metrics.

    predictions retains every requested target, including missing predictions for
    failed origins. origins records success/failure and exception context per issue.
    Tables are caller-owned mutable results; inputs are never mutated.
    """

    predictions: pd.DataFrame
    origins: pd.DataFrame
    spec: BacktestSpec
    model_id: str


class BacktestOriginError(RuntimeError):
    """Strict-mode failure; the original exception is retained as __cause__."""


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    if not data.columns.is_unique or not {"target_time", "pv_generation_mw"} <= set(data.columns):
        raise ValueError("data needs unique target_time and pv_generation_mw columns")
    if not isinstance(data["target_time"].dtype, pd.DatetimeTZDtype):
        raise ValueError("data.target_time must be timezone-aware datetime data")
    times = target_index(pd.DatetimeIndex(data["target_time"]))
    validation_bound = (
        times.max() + pd.Timedelta(minutes=15) if len(times) else pd.Timestamp("1970-01-01T00:00Z")
    )
    # Reuse existing numeric/key validation without truncating the source snapshot.
    pv_history(data, validation_bound)
    prepared = data.copy(deep=True)
    prepared["target_time"] = times
    prepared["pv_generation_mw"] = prepared["pv_generation_mw"].astype("Float64")
    return prepared.sort_values("target_time", ignore_index=True)


def _build_context(
    prepared: pd.DataFrame,
    request: ForecastRequest,
    spec: BacktestSpec,
    weather_provider: WeatherProvider | None = None,
) -> ForecastContext:
    """Shared origin reconstruction for inference and supervised dataset building."""
    selection = spec.availability_policy.select(prepared.copy(deep=True), request.issue_time)
    allowed = selection.history
    if spec.context_window is not None:
        allowed = allowed.loc[allowed["target_time"] >= request.issue_time - spec.context_window]
    weather = (
        None
        if weather_provider is None
        else weather_provider.for_issue(request.issue_time, request.target_times)
    )
    return ForecastContext(request, allowed, history_cutoff=selection.cutoff, weather=weather)


def backtest(
    *,
    model: Forecaster,
    data: pd.DataFrame,
    spec: BacktestSpec,
    weather_provider: WeatherProvider | None = None,
) -> BacktestResult:
    """Simulate each issue sequentially, giving models only policy-selected PV history.

    No automatic feature construction, fitting, model reset or data access occurs.
    Custom policies are trusted code. Models must obey the Forecaster contract.
    """
    prepared = _prepare_data(data)
    actuals = prepared.set_index("target_time")["pv_generation_mw"]
    model_id = model.model_id
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model.model_id must be a nonempty string")
    logger.info(
        "Backtest start: %s, %d issues, availability=%s",
        model_id,
        len(spec.issue_times),
        spec.availability_policy.name,
    )
    rows: list[pd.DataFrame] = []
    origins: list[dict[str, object]] = []
    for issue in spec.issue_times:
        # Invalid target policies/schedules are configuration failures, never hidden
        # by continue mode. ForecastRequest is the existing invariant boundary.
        request = ForecastRequest(issue, spec.target_policy(issue))
        targets = request.target_times
        cutoff: pd.Timestamp | None = None
        history_count = 0
        error_type: str | None = None
        error_message: str | None = None
        version: str | None = None
        status = "ok"
        weather_audit: dict[str, object] = {}
        if weather_provider is not None:
            weather_audit = {
                "weather_run_version": None,
                "weather_reference_time": None,
                "weather_available_at": None,
                "weather_availability_basis": None,
                "weather_policy": None,
                "weather_weights_id": None,
                "weather_selection_notes": None,
                "weather_radiation_policy": None,
                "weather_radiation_adjusted_cell_hours": None,
            }
        predictions = pd.Series(pd.NA, index=range(len(targets)), dtype="Float64")
        try:
            context = _build_context(prepared, request, spec, weather_provider)
            if context.weather is not None:
                weather_audit = {
                    "weather_run_version": context.weather.run_version,
                    "weather_reference_time": context.weather.reference_time,
                    "weather_available_at": context.weather.available_at,
                    "weather_availability_basis": context.weather.availability_basis,
                    "weather_policy": context.weather.policy_name,
                    "weather_weights_id": context.weather.weights_id,
                    "weather_selection_notes": context.weather.selection_notes,
                    "weather_radiation_policy": context.weather.radiation_policy,
                    "weather_radiation_adjusted_cell_hours": (
                        context.weather.radiation_adjusted_cell_hours
                    ),
                }
            cutoff = context.history_cutoff
            history_count = len(context.history)
            forecast = model.predict(context)
            if not isinstance(forecast, Forecast):
                raise ValueError("model.predict must return a Forecast")
            if (
                forecast.model_id != model_id
                or model.model_id != model_id
                or forecast.issue_time != request.issue_time
                or not forecast.request.target_times.equals(targets)
                or forecast.target_variable != "pv_generation"
            ):
                raise ValueError(
                    "Returned forecast does not match issue, targets, model or PV variable"
                )
            predictions = forecast.predictions["prediction_mw"].astype("Float64")
            version = forecast.model_version
        except Exception as error:
            if spec.on_error == "strict":
                raise BacktestOriginError(f"{model_id} at issue {issue}: {error}") from error
            status = "forecast_failed"
            error_type = type(error).__name__
            error_message = str(error)
            logger.warning(
                "%s at issue %s failed: %s: %s", model_id, issue, error_type, error_message
            )
        # Evaluator-side actual lookup happens only after the model call. Actuals
        # are never supplied to the model or mixed into context/features.
        observed = actuals.reindex(targets).astype("Float64").array
        frame = pd.DataFrame(
            {
                "issue_time": issue,
                "target_time": targets,
                "model_id": model_id,
                "model_version": version,
                "prediction_mw": predictions.array,
                "actual_mw": observed,
                "horizon": request.horizons,
                "status": status,
            }
        )
        frame["is_scorable"] = frame["prediction_mw"].notna() & frame["actual_mw"].notna()
        rows.append(frame)
        origins.append(
            {
                "issue_time": issue,
                "model_id": model_id,
                "history_cutoff": cutoff,
                "history_count": history_count,
                "target_count": len(targets),
                "status": status,
                "error_type": error_type,
                "error_message": error_message,
                **weather_audit,
            }
        )
    table = pd.concat(rows, ignore_index=True)
    table["issue_time"] = table["issue_time"].astype("datetime64[ns, UTC]")
    for column in ("model_id", "model_version", "status"):
        table[column] = table[column].astype("string")
    origin_table = pd.DataFrame(origins)
    origin_table["issue_time"] = origin_table["issue_time"].astype("datetime64[ns, UTC]")
    origin_table["history_cutoff"] = pd.to_datetime(
        origin_table["history_cutoff"], utc=True
    ).astype("datetime64[ns, UTC]")
    for column in ("model_id", "status", "error_type", "error_message"):
        origin_table[column] = origin_table[column].astype("string")
    logger.info(
        "Backtest complete: %s, %d successful, %d failed origins",
        model_id,
        int((origin_table.status == "ok").sum()),
        int((origin_table.status == "forecast_failed").sum()),
    )
    return BacktestResult(table, origin_table, spec, model_id)
