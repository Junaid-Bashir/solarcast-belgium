"""Stateless raw persistence baselines using only the supplied historical context."""

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from solarcast.models.types import Forecast, ForecastContext


class ForecastDataError(ValueError):
    """The supplied context cannot provide a complete baseline forecast."""


class LastValuePersistenceForecaster:
    """Repeat the latest eligible non-null PV value; a sanity-check baseline.

    Null trailing observations are skipped. Stale/negative/zero values are retained;
    no maximum-age, solar geometry or physical correction policy is invented.
    """

    @property
    def model_id(self) -> str:
        return "last_value_persistence"

    def predict(self, context: ForecastContext) -> Forecast:
        """Return a constant forecast, failing if no finite eligible history exists."""
        history = context.history
        eligible = history.loc[history["pv_generation_mw"].notna()]
        if eligible.empty:
            raise ForecastDataError(
                f"{self.model_id}: no finite non-null PV observation in supplied history; "
                f"history_cutoff={context.history_cutoff}, issue_time={context.request.issue_time}"
            )
        # Context sorts history and rejects infinity before a model sees it.
        value = float(eligible["pv_generation_mw"].iloc[-1])
        return Forecast(
            context.request,
            pd.DataFrame({"target_time": context.request.target_times, "prediction_mw": value}),
            model_id=self.model_id,
        )


@dataclass(frozen=True, init=False)
class SeasonalPersistenceForecaster:
    """Exact target-minus-lag lookup in elapsed UTC time; no filling or fallback.

    Lag must be a positive whole number of quarter-hours. '24h' and '7D' match the
    existing feature definitions; neither means a previous local calendar date.
    """

    lag: pd.Timedelta

    def __init__(self, lag: str | timedelta = "24h") -> None:
        if not isinstance(lag, (str, timedelta)):
            raise ValueError("lag must be an elapsed duration string or timedelta")
        try:
            duration = pd.Timedelta(lag)
            if (
                pd.isna(duration)
                or duration <= pd.Timedelta(0)
                or duration.value % pd.Timedelta(minutes=15).value != 0
            ):
                raise ValueError("lag must be positive and aligned to 15 minutes")
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(
                "lag must be a finite positive whole number of quarter-hours"
            ) from error
        object.__setattr__(self, "lag", duration)

    @property
    def model_id(self) -> str:
        """Normalize equivalent lag spellings to the same stable identity."""
        minutes = self.lag.value // pd.Timedelta(minutes=1).value
        if minutes % (7 * 24 * 60) == 0:
            suffix = f"{minutes // (24 * 60)}d"
        elif minutes % 60 == 0:
            suffix = f"{minutes // 60}h"
        else:
            suffix = f"{minutes}m"
        return f"seasonal_persistence_{suffix}"

    def predict(self, context: ForecastContext) -> Forecast:
        """Return all exact historical matches or identify unavailable source times."""
        history = context.history.set_index("target_time")["pv_generation_mw"]
        targets = context.request.target_times
        required = targets - self.lag
        values = history.reindex(required)
        missing = values.isna().to_numpy()
        if missing.any():
            unavailable = required[missing]
            preview = ", ".join(stamp.isoformat() for stamp in unavailable[:5])
            more = "; additional timestamps omitted" if len(unavailable) > 5 else ""
            raise ForecastDataError(
                f"{self.model_id}: {len(unavailable)} required historical timestamp(s) "
                f"absent or null: [{preview}]{more}; history_cutoff={context.history_cutoff}, "
                f"issue_time={context.request.issue_time}. No fallback applied."
            )
        return Forecast(
            context.request,
            pd.DataFrame({"target_time": targets, "prediction_mw": values.to_numpy(dtype=float)}),
            model_id=self.model_id,
        )
