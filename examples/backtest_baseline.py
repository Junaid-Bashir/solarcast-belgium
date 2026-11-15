"""Offline normal/spring/autumn rolling baseline smoke; no metrics or downloads."""

import pandas as pd

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest
from solarcast.models import SeasonalPersistenceForecaster


def main() -> None:
    """Run three consecutive origins for each representative target-day case."""
    for target_day, expected in (("2026-02-01", 96), ("2026-03-29", 92), ("2026-10-25", 100)):
        local = pd.Timestamp(target_day, tz="Europe/Brussels")
        first_issue = (local - pd.DateOffset(days=1)).replace(hour=11)
        issues = pd.date_range(first_issue, periods=3, freq="D")
        times = pd.date_range(
            issues[0].tz_convert("UTC") - pd.Timedelta(days=8),
            issues[-1].tz_convert("UTC") + pd.Timedelta(days=3),
            freq="15min",
        )
        source = pd.DataFrame(
            {"target_time": times, "pv_generation_mw": pd.array(range(len(times)), dtype="Float64")}
        )
        result = backtest(
            model=SeasonalPersistenceForecaster("7D"),
            data=source,
            spec=BacktestSpec(
                issues, availability_policy=EventTimeCutoff("30min"), context_window="14D"
            ),
        )
        assert result.origins.target_count.iloc[0] == expected
        assert result.predictions.is_scorable.all()
        print(
            f"{target_day}: first origin {expected} targets; "
            f"{len(result.origins)} successful origins"
        )
        print(result.predictions.head(2).to_string(index=False))


if __name__ == "__main__":
    main()
