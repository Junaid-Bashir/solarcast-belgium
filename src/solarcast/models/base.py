"""The minimal inference capability; training is an adapter-specific concern."""

from typing import Protocol

from solarcast.models.types import Forecast, ForecastContext


class Forecaster(Protocol):
    """Structural contract satisfied without inheritance or a dummy fit method.

    Adapters must use only supplied inputs and their declared model state. They
    must not fetch hidden observations during prediction. Return a Forecast bound
    to context.request, using the same model_id. Model-specific state errors should
    be explicit (for example, predicting before required training).
    """

    @property
    def model_id(self) -> str:
        """Stable human-readable identifier used in results and comparisons."""
        ...

    def predict(self, context: ForecastContext) -> Forecast:
        """Forecast the explicit request targets from bounded supplied context."""
        ...
