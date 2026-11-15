"""Optional CPU LightGBM adapter, trained once then used through Forecaster."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from solarcast.backtesting import BacktestSpec
from solarcast.features import SolarLocation
from solarcast.models.lightgbm_data import (
    FeatureSet,
    TrainingDataset,
    feature_names,
    lightgbm_features,
    validate_feature_matrix,
)
from solarcast.models.types import Forecast, ForecastContext

if TYPE_CHECKING:
    import lightgbm as lgb

logger = logging.getLogger(__name__)


class LightGBMNotFittedError(RuntimeError):
    """Prediction requires a successful explicit fit."""


@dataclass(frozen=True)
class LightGBMConfig:
    """Small fixed-estimator CPU configuration; no tuning, sampling or early stopping."""

    n_estimators: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    random_state: int = 0

    def __post_init__(self) -> None:
        for name in ("n_estimators", "num_leaves", "min_child_samples", "random_state"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.n_estimators < 1 or self.num_leaves < 2 or self.min_child_samples < 1:
            raise ValueError("Need positive estimator/min-child counts and at least two leaves")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be finite and in (0, 1]")


@dataclass(frozen=True)
class LightGBMTrainingInfo:
    fit_time: pd.Timestamp
    label_cutoff: pd.Timestamp
    first_training_target: pd.Timestamp
    last_training_target: pd.Timestamp
    n_rows: int
    n_missing_labels: int
    feature_names: tuple[str, ...]
    lightgbm_version: str
    solarcast_version: str
    config: LightGBMConfig
    training_spec: BacktestSpec
    training_window: pd.Timedelta | None
    location: SolarLocation
    target_variable: str = "pv_generation"
    unit: str = "MW"
    feature_set: FeatureSet = "history"


class LightGBMForecaster:
    """Pooled issue/target regression. Optional native library is imported only by fit.

    Precomputed context.features is ignored. Approved NWP is opt-in by feature_set.
    Missing historical features stay NaN;
    no raw prediction clipping/nighttime masking or external history lookup occurs.
    """

    def __init__(
        self, config: LightGBMConfig = LightGBMConfig(), *, feature_set: FeatureSet = "history"
    ) -> None:
        feature_names(feature_set)
        self.feature_set = feature_set
        self.config = config
        self._booster: lgb.Booster | None = None
        self._info: LightGBMTrainingInfo | None = None

    @property
    def model_id(self) -> str:
        return (
            "lightgbm"
            if self.feature_set == "history"
            else "lightgbm_" + self.feature_set.replace("+", "_")
        )

    @property
    def training_info(self) -> LightGBMTrainingInfo:
        if self._info is None:
            raise LightGBMNotFittedError("lightgbm must be fitted before inspecting training_info")
        return self._info

    def fit(self, training: TrainingDataset) -> None:
        """Fit once from explicit origin-aware examples; clear old state before attempting refit."""
        self._booster = None
        self._info = None
        if training.feature_set != self.feature_set:
            raise ValueError("Training feature_set differs from model feature_set")
        names = feature_names(self.feature_set)
        validate_feature_matrix(training.X, self.feature_set)
        if (
            training.X.empty
            or not training.X.index.equals(training.y.index)
            or not training.rows.index.equals(training.y.index)
            or len(training.rows) != len(training.y)
            or not np.isfinite(training.y.to_numpy(dtype=float)).all()
        ):
            raise ValueError("Training X/y/audit rows must align and labels must be finite")
        if (training.rows.target_time + pd.Timedelta(minutes=15) > training.label_cutoff).any():
            raise ValueError("Training labels extend past fit-time availability cutoff")
        if training.label_cutoff > training.fit_time:
            raise ValueError("Training label cutoff extends past fit_time")
        if self.feature_set != "history":
            required = {
                "weather_reference_time",
                "weather_available_at",
                "weather_run_version",
                "weather_policy",
                "weather_availability_basis",
                "weather_weights_id",
            }
            if (
                not required <= set(training.rows)
                or training.rows[list(required)].isna().any().any()
            ):
                raise ValueError("Weather training needs complete vintage audit rows")
            if (
                training.rows.weather_reference_time > training.rows.weather_available_at
            ).any() or (training.rows.weather_available_at > training.rows.issue_time).any():
                raise ValueError("Weather training availability exceeds historical issue time")
        if (
            (training.rows.issue_time >= training.fit_time).any()
            or (training.rows.history_cutoff > training.rows.issue_time).any()
            or (training.rows.target_time <= training.rows.issue_time).any()
        ):
            raise ValueError("Training audit rows violate issue/target/cutoff ordering")
        try:
            import lightgbm as lgb
        except ImportError as error:
            raise ImportError(
                'Install optional support with pip install -e ".[dev,lightgbm]" '
                "from this repository (the distribution is not published yet)"
            ) from error
        params = {
            "objective": "regression_l2",
            "metric": "None",
            "verbosity": -1,
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "min_data_in_leaf": self.config.min_child_samples,
            "seed": self.config.random_state,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "use_missing": True,
            "zero_as_missing": False,
        }
        logger.info(
            "Fitting lightgbm: %d rows, %d features, targets [%s, %s], config=%s",
            len(training.y),
            len(names),
            training.rows.target_time.min(),
            training.rows.target_time.max(),
            self.config,
        )
        dataset = lgb.Dataset(
            training.X, label=training.y.to_numpy(dtype=float), feature_name=list(names)
        )
        booster = lgb.train(params, dataset, num_boost_round=self.config.n_estimators)
        self._booster = booster
        self._info = LightGBMTrainingInfo(
            training.fit_time,
            training.label_cutoff,
            training.rows.target_time.min(),
            training.rows.target_time.max(),
            len(training.y),
            training.n_missing_labels,
            names,
            version("lightgbm"),
            version("solarcast-belgium"),
            self.config,
            training.spec,
            training.training_window,
            training.location,
            feature_set=self.feature_set,
        )

    def predict(self, context: ForecastContext) -> Forecast:
        if self._booster is None or self._info is None:
            raise LightGBMNotFittedError("lightgbm must be fitted before predict")
        if context.request.issue_time < self.training_info.fit_time:
            raise ValueError(
                "Forecast issue precedes fitted model's fit_time; possible training leakage"
            )
        features = lightgbm_features(
            context, location=self.training_info.location, feature_set=self.feature_set
        )
        validate_feature_matrix(features, self.feature_set)
        if tuple(self._booster.feature_name()) != self.training_info.feature_names:
            raise ValueError("Fitted LightGBM feature schema mismatch")
        values = np.asarray(self._booster.predict(features, validate_features=True), dtype=float)
        return Forecast(
            context.request,
            pd.DataFrame({"target_time": context.request.target_times, "prediction_mw": values}),
            model_id=self.model_id,
            model_version=self.training_info.lightgbm_version,
        )
