"""Forecasting contracts and stateless persistence baselines."""

from solarcast.models.base import Forecaster
from solarcast.models.persistence import (
    ForecastDataError,
    LastValuePersistenceForecaster,
    SeasonalPersistenceForecaster,
)
from solarcast.models.types import Forecast, ForecastContext, ForecastRequest

__all__ = [
    "Forecast",
    "ForecastContext",
    "ForecastDataError",
    "ForecastRequest",
    "Forecaster",
    "LastValuePersistenceForecaster",
    "SeasonalPersistenceForecaster",
]
