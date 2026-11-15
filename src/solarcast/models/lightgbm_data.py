"""Origin-aware supervised rows and the shared pooled-regression feature contract."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from solarcast.backtesting import (
    BacktestSpec,
    WeatherProvider,
    _build_context,
    _duration,
    _prepare_data,
)
from solarcast.data.weather.types import FEATURES as WEATHER_FEATURES
from solarcast.data.weather.types import WeatherError
from solarcast.features import (
    BELGIUM_REFERENCE_LOCATION,
    SolarLocation,
    calendar_features,
    pv_lag_features,
    pv_rolling_features,
    solar_features,
)
from solarcast.features._validation import cutoff_timestamp, pv_history
from solarcast.models.types import ForecastContext, ForecastRequest, _aware_time

logger = logging.getLogger(__name__)
FEATURE_NAMES = (
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "solar_apparent_elevation_deg",
    "solar_azimuth_deg",
    "clearsky_ghi_w_m2",
    "horizon_hours",
    "pv_lag_24h_mw",
    "pv_lag_48h_mw",
    "pv_lag_7d_mw",
    "pv_trailing_24h_mean_mw",
    "pv_trailing_24h_max_mw",
)
FeatureSet = Literal["history", "history+radiation", "history+radiation+cloud", "history+nwp"]


def feature_names(feature_set: FeatureSet = "history") -> tuple[str, ...]:
    counts = {"history": 0, "history+radiation": 4, "history+radiation+cloud": 6, "history+nwp": 7}
    if feature_set not in counts:
        raise ValueError("Unknown LightGBM feature_set")
    return FEATURE_NAMES + WEATHER_FEATURES[: counts[feature_set]]


def validate_feature_matrix(features: pd.DataFrame, feature_set: FeatureSet = "history") -> None:
    """Require configured float64 columns; history NaNs stay, NWP must be complete."""
    if tuple(features.columns) != feature_names(feature_set):
        raise ValueError("LightGBM feature names/order mismatch; rebuild with lightgbm_features")
    if any(dtype != np.dtype("float64") for dtype in features.dtypes):
        raise ValueError("LightGBM features must all have float64 dtype")
    if np.isinf(features.to_numpy()).any():
        raise ValueError("LightGBM features must not contain infinity")
    weather_names = feature_names(feature_set)[len(FEATURE_NAMES) :]
    if weather_names and not np.isfinite(features[list(weather_names)].to_numpy()).all():
        raise WeatherError("LightGBM NWP features must be complete and finite")


def lightgbm_features(
    context: ForecastContext,
    *,
    location: SolarLocation = BELGIUM_REFERENCE_LOCATION,
    feature_set: FeatureSet = "history",
) -> pd.DataFrame:
    """Build exactly the same pooled features for training and prediction.

    Ignore precomputed context.features: use only the original history recipe and
    the configured approved NWP columns from validated, issue-aligned weather.
    """
    times = context.request.target_times
    history = context.history
    groups = (
        calendar_features(times),
        solar_features(times, location=location),
        pv_lag_features(history, times, history_cutoff=context.history_cutoff),
        pv_rolling_features(history, times, history_cutoff=context.history_cutoff),
    )
    features = pd.concat([group.drop(columns="target_time") for group in groups], axis=1)
    features["horizon_hours"] = context.request.horizons.total_seconds() / 3600
    features = features.loc[:, list(FEATURE_NAMES)].astype("float64")
    names = feature_names(feature_set)
    if feature_set != "history":
        if context.weather is None:
            raise WeatherError("weather-informed LightGBM requires eligible complete NWP")
        weather = context.weather.features
        if context.weather.issue_time != context.request.issue_time or not pd.DatetimeIndex(
            weather.target_time
        ).equals(times):
            raise WeatherError("NWP issue/targets differ from LightGBM request")
        for name in names[len(FEATURE_NAMES) :]:
            features[name] = weather[name].to_numpy(dtype=float)
    validate_feature_matrix(features, feature_set)
    return features


@dataclass(frozen=True)
class TrainingDataset:
    """Owned X/y plus audit rows; not a generic ML dataset or a serialization format."""

    X: pd.DataFrame
    y: pd.Series
    rows: pd.DataFrame
    fit_time: pd.Timestamp
    label_cutoff: pd.Timestamp
    spec: BacktestSpec
    location: SolarLocation
    training_window: pd.Timedelta | None
    n_requested: int
    n_missing_labels: int
    feature_set: FeatureSet = "history"
    origin_failures: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_training_dataset(
    data: pd.DataFrame,
    spec: BacktestSpec,
    *,
    fit_time: str | datetime,
    training_window: str | timedelta | None = None,
    location: SolarLocation = BELGIUM_REFERENCE_LOCATION,
    feature_set: FeatureSet = "history",
    weather_provider: WeatherProvider | None = None,
) -> TrainingDataset:
    """Reconstruct each historical origin, then join only fit-time-eligible labels.

    training_window limits origin times to [fit_time-window, fit_time), separately
    from spec.context_window. Whole training target grids must finish by the label
    cutoff; a split-crossing training schedule is rejected, not silently truncated.
    """
    boundary = _aware_time(fit_time, "fit_time")
    names = feature_names(feature_set)
    if (spec.issue_times >= boundary).any():
        raise ValueError("All training origins must precede fit_time")
    window = (
        None
        if training_window is None
        else _duration(training_window, "training_window", allow_zero=False)
    )
    issues = spec.issue_times
    if window is not None:
        issues = issues[issues >= boundary - window]
    if issues.empty:
        raise ValueError("No training origins remain inside training_window")
    prepared = _prepare_data(data)
    selection = spec.availability_policy.select(prepared.copy(deep=True), boundary)
    label_cutoff = cutoff_timestamp(selection.cutoff)
    if label_cutoff > boundary:
        raise ValueError("Label availability cutoff must not exceed fit_time")
    labels = pv_history(selection.history, label_cutoff)
    if len(labels) != len(selection.history):
        raise ValueError("Label policy returned incomplete/post-cutoff observations")
    matrices: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    audit: list[pd.DataFrame] = []
    requested = missing = 0
    failures: list[dict[str, object]] = []
    for issue in issues:
        request = ForecastRequest(issue, spec.target_policy(issue))
        if (request.target_times + pd.Timedelta(minutes=15) > label_cutoff).any():
            raise ValueError(
                f"Training target intervals at issue {issue} cross fit-time label cutoff"
            )
        requested += len(request.target_times)
        try:
            context = _build_context(
                prepared, request, spec, weather_provider if feature_set != "history" else None
            )
            features = lightgbm_features(context, location=location, feature_set=feature_set)
        except WeatherError as error:
            if spec.on_error == "strict":
                raise
            failures.append(
                {
                    "issue_time": issue,
                    "n_targets": len(request.target_times),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            continue
        y = pd.Series(
            labels.reindex(request.target_times).to_numpy(dtype=float, na_value=np.nan),
            name="pv_generation_mw",
        )
        valid = y.notna()
        missing += int((~valid).sum())
        matrices.append(features.loc[valid].reset_index(drop=True))
        targets.append(y.loc[valid].reset_index(drop=True))
        origin_audit = pd.DataFrame(
            {
                "issue_time": issue,
                "target_time": request.target_times,
                "history_cutoff": context.history_cutoff,
            }
        )
        if context.weather is not None:
            origin_audit["weather_reference_time"] = context.weather.reference_time
            origin_audit["weather_available_at"] = context.weather.available_at
            origin_audit["weather_run_version"] = context.weather.run_version
            origin_audit["weather_policy"] = context.weather.policy_name
            origin_audit["weather_availability_basis"] = context.weather.availability_basis
            origin_audit["weather_weights_id"] = context.weather.weights_id
            origin_audit["weather_radiation_policy"] = context.weather.radiation_policy
            origin_audit["weather_radiation_adjusted_cell_hours"] = (
                context.weather.radiation_adjusted_cell_hours
            )
        audit.append(origin_audit.loc[valid])
    if not matrices:
        raise WeatherError("No training origins have complete eligible NWP")
    X = pd.concat(matrices, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)
    rows = pd.concat(audit, ignore_index=True)
    if X.empty:
        raise ValueError("No finite available target labels remain for training")
    logger.info(
        "Training dataset: %d origins, %d rows, %d missing labels excluded, %d features; "
        "targets [%s, %s], fit_time=%s",
        len(issues),
        len(y),
        missing,
        len(names),
        rows.target_time.min(),
        rows.target_time.max(),
        boundary,
    )
    return TrainingDataset(
        X,
        y,
        rows,
        boundary,
        label_cutoff,
        spec,
        location,
        window,
        requested,
        missing,
        feature_set,
        pd.DataFrame(failures),
    )
