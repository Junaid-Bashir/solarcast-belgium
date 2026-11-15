"""Offline example using synthetic normalized PV data; no forecast or download."""

import pandas as pd

from solarcast.features import (
    calendar_features,
    pv_lag_features,
    pv_rolling_features,
    solar_features,
)


def main() -> None:
    """Compose future-known and cutoff-limited historical features."""
    observed_times = pd.date_range("2026-06-01T00:00Z", periods=8 * 96, freq="15min")
    # Synthetic illustrative values, not observations or a forecasting model.
    pv = pd.DataFrame(
        {
            "target_time": observed_times,
            "pv_generation_mw": pd.array([100.0] * len(observed_times), dtype="Float64"),
            "monitored_capacity_mw": 1000.0,
        }
    )
    targets = pd.date_range("2026-06-09T12:00Z", periods=4, freq="15min")
    cutoff = "2026-06-08T11:00Z"
    features = calendar_features(targets)
    for group in (
        solar_features(targets),
        pv_lag_features(pv, targets, history_cutoff=cutoff),
        pv_rolling_features(pv, targets, history_cutoff=cutoff),
    ):
        features = features.merge(group, on="target_time", validate="one_to_one")
    print(features.to_string(index=False))
    print("Lag values after the cutoff stay missing; capacity is not a feature.")


if __name__ == "__main__":
    main()
