"""Whole-request, whole-vintage selection before any model call."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

from .spatial import SpatialWeights
from .transforms import weather_features
from .types import ForecastRun, WeatherCovariates, WeatherError, targets_utc, utc


@dataclass(frozen=True)
class NWPAvailability:
    """A lag approximation is never provider-observed publication evidence."""

    mode: Literal["assumed", "observed"] = "assumed"
    delay: pd.Timedelta = pd.Timedelta(hours=6)

    def __post_init__(self) -> None:
        if (
            self.mode not in ("assumed", "observed")
            or not isinstance(self.delay, pd.Timedelta)
            or pd.isna(self.delay)
            or self.delay < pd.Timedelta(0)
        ):
            raise WeatherError("invalid NWP availability policy")

    @property
    def name(self) -> str:
        if self.mode == "observed":
            return "gfs_observed_bundle_v1"
        if self.delay == pd.Timedelta(hours=6):
            return "gfs_init_plus_6h_v1"
        return f"gfs_init_plus_{self.delay.value}ns_v1"

    def available_at(self, run: ForecastRun) -> pd.Timestamp | None:
        available = (
            run.reference_time + self.delay if self.mode == "assumed" else run.observed_available_at
        )
        # Never backdate a known corrected copy to its original initialization.
        if run.revision:
            if run.version_available_at is None or available is None:
                return None
            available = max(available, run.version_available_at)
        return available


def select_nwp_vintage(
    runs: Sequence[ForecastRun],
    issue_time: str | datetime,
    targets: pd.DatetimeIndex,
    *,
    weights: SpatialWeights,
    policy: NWPAvailability = NWPAvailability(),
    radiation_policy: str = "strict",
) -> WeatherCovariates:
    """Newest eligible complete run, with no cross-run variable/time substitution.

    Multiple eligible versions at the same initialization are ambiguous and fail;
    callers must explicitly resolve archive revisions rather than depend on order.
    """
    issue, times = utc(issue_time, "issue_time"), targets_utc(targets)
    if (times <= issue).any():
        raise WeatherError("weather targets must follow issue")
    eligible = [(run, policy.available_at(run)) for run in runs]
    eligible = [
        (run, available)
        for run, available in eligible
        if available is not None and available <= issue
    ]
    references = [run.reference_time for run, _ in eligible]
    if len(references) != len(set(references)):
        raise WeatherError("ambiguous eligible versions of the same GFS cycle")
    failures = []
    for run, available in sorted(eligible, key=lambda item: item[0].reference_time, reverse=True):
        try:
            features = weather_features(run, times, weights, radiation_policy=radiation_policy)
        except WeatherError as error:
            failures.append(f"{run.reference_time}: {error}")
            continue
        assert available is not None
        return WeatherCovariates(
            features,
            issue_time=issue,
            reference_time=run.reference_time,
            available_at=available,
            availability_basis=policy.mode,
            policy_name=policy.name,
            run_version=run.version,
            weights_id=weights.identity,
            selection_notes=tuple(failures),
            radiation_policy=radiation_policy,
            radiation_adjusted_cell_hours=features.attrs["radiation_adjusted_cell_hours"],
        )
    raise WeatherError("no eligible complete NWP vintage; " + "; ".join(failures))


@dataclass(frozen=True)
class WeatherStore:
    """Local run inventory; engine invokes this selector, models never fetch archives."""

    runs: tuple[ForecastRun, ...]
    weights: SpatialWeights
    policy: NWPAvailability = NWPAvailability()
    radiation_policy: str = "strict"

    def for_issue(self, issue_time: pd.Timestamp, targets: pd.DatetimeIndex) -> WeatherCovariates:
        return select_nwp_vintage(
            self.runs,
            issue_time,
            targets,
            weights=self.weights,
            policy=self.policy,
            radiation_policy=self.radiation_policy,
        )
