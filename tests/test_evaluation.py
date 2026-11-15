"""Manual metric checks, common-support fairness and offline baseline integration."""

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest, day_ahead_local_day
from solarcast.evaluation import bias, compare_models, evaluate_forecast, mae, rmse, rmse_skill
from solarcast.features import solar_features
from solarcast.models import LastValuePersistenceForecaster, SeasonalPersistenceForecaster


def rows(
    model: str = "a",
    predictions: list[float | None] | None = None,
    actuals: list[float | None] | None = None,
    targets: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    predictions = [1, 2, 3, 4] if predictions is None else predictions
    actuals = [1.0] * len(predictions) if actuals is None else actuals
    if targets is None:
        targets = pd.date_range("2026-06-10T12:00Z", periods=len(predictions), freq="15min")
    issue = pd.Timestamp("2026-06-10T09:00Z")
    return pd.DataFrame(
        {
            "issue_time": issue,
            "target_time": targets,
            "model_id": model,
            "prediction_mw": pd.array(predictions, dtype="Float64"),
            "actual_mw": pd.array(actuals, dtype="Float64"),
            "horizon": targets - issue,
            "status": "ok",
            "is_scorable": True,
        }
    )


def test_perfect_forecast() -> None:
    # The separate typed calls also verify the public numerical API.
    assert mae([10, 20, 30], [10, 20, 30]) == 0
    assert rmse([10, 20, 30], [10, 20, 30]) == 0
    assert bias([10, 20, 30], [10, 20, 30]) == 0


def test_manual_errors_and_bias_direction() -> None:
    actual, prediction = [10, 20, 30], [8, 24, 30]
    assert mae(actual, prediction) == pytest.approx(2)
    assert rmse(actual, prediction) == pytest.approx(math.sqrt(20 / 3))
    assert bias(actual, prediction) == pytest.approx(2 / 3)
    assert bias([0, 0], [-2, -4]) == -3
    assert bias([0, 0], [2, 4]) == 3
    assert rmse([0, 0], [-2, -4]) == pytest.approx(math.sqrt(10))


@pytest.mark.parametrize("invalid", [[], [float("nan")], [float("inf")], [True], ["1"], [[1]]])
def test_low_level_metrics_reject_incomplete_or_invalid_vectors(invalid: object) -> None:
    # Evaluation handles masking; numerical functions never silently drop observations.
    for function in (mae, rmse, bias):
        with pytest.raises(ValueError):
            function(np.asarray(invalid), [1])
    with pytest.raises(ValueError, match="identical lengths"):
        rmse([1], [1, 2])


def test_rmse_scaling_avoids_overflow_for_representable_errors() -> None:
    assert rmse([0, 0], [1e200, -1e200]) == pytest.approx(1e200)


@pytest.mark.parametrize("prediction,expected", [([1, 1], 0.5), ([2, 2], 0), ([4, 4], -1)])
def test_skill_better_equal_worse(prediction: list[float], expected: float) -> None:
    assert rmse_skill([0, 0], prediction, [2, 2]) == pytest.approx(expected)


def test_perfect_reference_skill_is_explicitly_undefined() -> None:
    assert math.isnan(rmse_skill([0, 0], [1, 1], [0, 0]))
    assert math.isnan(rmse_skill([0, 0], [0, 0], [0, 0]))
    benchmark = compare_models([rows("a", [1, 1]), rows("ref", [1, 1])], reference_model="ref")
    assert benchmark.summary.rmse_skill.isna().all()
    assert set(benchmark.summary.skill_status) == {"zero_reference_error"}


def test_independent_evaluation_counts_missing_invalid_and_unclipped_output() -> None:
    table = rows(
        predictions=[-2, None, float("inf"), 4, 5, 6], actuals=[0, 0, 0, None, float("inf"), 6]
    )
    table["is_scorable"] = False  # Recompute validity instead of trusting stale flags.
    summary = evaluate_forecast(table).summary.iloc[0]
    assert summary.n_total == summary.n_expected == 6
    assert summary.n_scored == 2
    assert summary.n_actual_missing == summary.n_actual_nonfinite == 1
    assert summary.n_prediction_missing == summary.n_prediction_invalid == 1
    assert summary.mae_mw == 1
    assert summary.bias_mw == -1
    assert summary.rmse_mw == pytest.approx(math.sqrt(2))
    assert summary.forecast_availability == pytest.approx(4 / 6)
    assert summary.subset == "all_intervals"


def test_common_support_uses_2_3_4_for_both_and_reports_union_denominator() -> None:
    targets = pd.date_range("2026-06-10T12:00Z", periods=5, freq="15min")
    a = rows("a", [100, 2, 3, 4], targets=targets[:4])
    b = rows("b", [2, 3, 4, 100], targets=targets[1:])
    original = a.copy(deep=True)
    result = compare_models([a, b], reference_model="b")
    assert list(result.summary.n_scored) == [3, 3]
    assert list(result.summary.n_expected) == [5, 5]
    assert list(result.summary.n_missing_rows) == [1, 1]
    assert list(result.summary.n_excluded_alignment) == [1, 1]
    assert list(result.summary.forecast_availability) == [0.8, 0.8]
    assert list(result.summary.rmse_mw) == pytest.approx([math.sqrt(14 / 3)] * 2)
    assert (result.summary.rmse_skill == 0).all()
    assert set(result.aligned_predictions.target_time) == set(targets[1:4])
    assert_frame_equal(a, original)


def test_alignment_removes_every_models_missing_or_invalid_predictions() -> None:
    a = rows("a", [1, 2, 3, 4])
    b = rows("b", [1, None, float("inf"), 4])
    benchmark = compare_models([a, b])
    assert list(benchmark.summary.n_scored) == [2, 2]
    assert list(benchmark.summary.n_valid_pairs) == [4, 2]
    assert list(benchmark.summary.n_excluded_alignment) == [2, 0]


def test_issue_plus_target_is_alignment_key_not_target_alone() -> None:
    a, b = rows("a"), rows("b")
    b["issue_time"] -= pd.Timedelta(hours=1)
    b["horizon"] = b.target_time - b.issue_time
    result = compare_models([a, b])
    assert result.summary.n_scored.eq(0).all()
    assert result.summary.n_expected.eq(8).all()
    assert result.leaderboard()["rank"].isna().all()


@pytest.mark.parametrize("actual", [2.0, None])
def test_conflicting_actual_values_or_availability_rejected(actual: float | None) -> None:
    a, b = rows("a"), rows("b")
    b.loc[0, "actual_mw"] = actual
    with pytest.raises(ValueError, match="Inconsistent actual"):
        compare_models([a, b])


def test_leaderboard_ranking_ties_reference_and_period_metadata() -> None:
    benchmark = compare_models(
        [rows("z", [2, 2]), rows("a", [2, 2]), rows("ref", [3, 3])], reference_model="ref"
    )
    board = benchmark.leaderboard()
    assert list(board.model_id) == ["a", "z", "ref"]
    assert list(board["rank"]) == [1, 1, 3]
    assert set(board.ranking_metric) == {"rmse"}
    assert list(board.rmse_skill) == [0.5, 0.5, 0]
    own = benchmark.summary.iloc[0]
    assert own.n_origins == 1
    assert own.first_target == pd.Timestamp("2026-06-10T12:00Z")
    assert own.last_scored_target == pd.Timestamp("2026-06-10T12:15Z")
    reverse = compare_models(
        [rows("ref", [3, 3]), rows("a", [2, 2]), rows("z", [2, 2])], reference_model="ref"
    )
    assert_frame_equal(board, reverse.leaderboard())


def test_primary_metric_can_change_ranking_explicitly() -> None:
    a, b = rows("a", [1, 5], [1, 1]), rows("b", [3.5, 3.5], [1, 1])
    result = compare_models([a, b])
    assert result.leaderboard().model_id.iloc[0] == "b"
    assert result.leaderboard(primary_metric="mae").model_id.iloc[0] == "a"


def test_failed_origins_stay_in_forecast_availability_and_common_support() -> None:
    a, b = rows("a"), rows("b")
    b["status"] = "forecast_failed"
    b["prediction_mw"] = pd.array([None] * 4, dtype="Float64")
    result = compare_models([a, b], reference_model="b")
    assert result.summary.n_scored.eq(0).all()
    assert result.summary.rmse_mw.isna().all()
    assert result.leaderboard()["rank"].isna().all()
    b_summary = result.summary.set_index("model_id").loc["b"]
    assert b_summary.n_failed_origins == 1
    assert b_summary.forecast_availability == 0


def test_grouped_horizons_local_hour_month_and_season() -> None:
    targets = pd.DatetimeIndex(["2026-06-10T22:00Z", "2026-06-10T22:15Z"])
    benchmark = compare_models(
        [rows("a", [3, 5], targets=targets), rows("b", [2, 2], targets=targets)]
    )
    assert len(benchmark.grouped("horizon")) == 4
    hours = benchmark.grouped("local_hour")
    assert set(hours.local_hour) == {0}  # Brussels is UTC+2 here, not 22.
    assert set(benchmark.grouped("month").month) == {"2026-06"}
    assert set(benchmark.grouped("season").season) == {"JJA"}
    assert benchmark.grouped("horizon").n_scored.eq(1).all()


def test_season_definitions_and_year_month_boundaries() -> None:
    tables: list[pd.DataFrame] = []
    for month in (1, 4, 7, 10):
        target = pd.DatetimeIndex([f"2026-{month:02d}-15T12:00Z"])
        table = rows("a", [2], targets=target)
        table["issue_time"] = target[0] - pd.Timedelta(hours=1)
        table["horizon"] = pd.Timedelta(hours=1)
        tables.append(table)
    result = compare_models([pd.concat(tables)])
    assert set(result.grouped("season").season) == {"DJF", "MAM", "JJA", "SON"}
    assert len(result.grouped("month")) == 4


def daylight(keys: pd.DataFrame) -> pd.Series:
    solar = solar_features(pd.DatetimeIndex(keys.target_time.unique())).set_index("target_time")
    return keys.target_time.map(solar.is_daylight).astype(bool)


def test_nighttime_included_by_default_and_explicit_daylight_subset_is_labelled() -> None:
    targets = pd.DatetimeIndex(["2026-06-21T00:00Z", "2026-06-21T12:00Z"])
    table = rows(predictions=[1, 3], actuals=[1, 1], targets=targets)
    assert evaluate_forecast(table).summary.n_scored.iloc[0] == 2
    subset = compare_models([table], subset=daylight, subset_name="reference_point_daylight")
    assert subset.summary.n_scored.iloc[0] == 1
    assert subset.summary.mae_mw.iloc[0] == 2
    assert subset.subset_name == "reference_point_daylight"
    assert subset.summary.subset.iloc[0] == "reference_point_daylight"


def test_empty_subset_has_no_fake_scores_or_ranks() -> None:
    result = compare_models(
        [rows()], subset=lambda keys: keys.target_time.isna(), subset_name="empty_test_subset"
    )
    assert result.summary.n_expected.iloc[0] == 0
    assert result.summary.rmse_mw.isna().all()
    assert result.leaderboard()["rank"].isna().all()
    assert result.grouped("horizon").empty


@pytest.mark.parametrize(
    "issue,count",
    [("2026-01-31T10:00Z", 96), ("2026-03-28T10:00Z", 92), ("2026-10-24T09:00Z", 100)],
)
def test_backtest_evaluation_dst_and_model_agnostic_integration(issue: str, count: int) -> None:
    stamp = pd.Timestamp(issue)
    targets = day_ahead_local_day(stamp)
    times = pd.date_range(targets[0] - pd.Timedelta(days=8), targets[-1], freq="15min")
    source = pd.DataFrame(
        {"target_time": times, "pv_generation_mw": pd.array(range(len(times)), dtype="Float64")}
    )
    spec = BacktestSpec(pd.DatetimeIndex([stamp]), availability_policy=EventTimeCutoff("30min"))
    results = [
        backtest(model=LastValuePersistenceForecaster(), data=source, spec=spec),
        backtest(model=SeasonalPersistenceForecaster("7D"), data=source, spec=spec),
    ]
    benchmark = compare_models(results, reference_model="seasonal_persistence_7d")
    assert benchmark.summary.n_scored.eq(count).all()
    assert list(benchmark.summary.n_expected) == [count, count]
    assert len(benchmark.backtest_specs) == 2
    if count == 100:
        local = benchmark.grouped("local_hour")
        assert list(local.loc[local.local_hour == 2, "n_scored"]) == [8, 8]


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "naive", "bad_horizon", "string_prediction", "bad_status", "mixed_models"],
)
def test_invalid_evaluation_input_fails_clearly(fault: str) -> None:
    table = rows()
    if fault == "duplicate":
        table = pd.concat([table, table])
    elif fault == "naive":
        table["issue_time"] = table.issue_time.dt.tz_localize(None)
    elif fault == "bad_horizon":
        table["horizon"] += pd.Timedelta(minutes=15)
    elif fault == "string_prediction":
        table["prediction_mw"] = "1"
    elif fault == "bad_status":
        table["status"] = "unknown"
    else:
        table.loc[0, "model_id"] = "other"
    with pytest.raises(ValueError):
        compare_models([table])


def test_missing_reference_duplicate_models_and_unlabelled_subsets_fail() -> None:
    with pytest.raises(ValueError, match="Reference model"):
        compare_models([rows()], reference_model="absent")
    with pytest.raises(ValueError, match="Duplicate model"):
        compare_models([rows(), rows()])
    with pytest.raises(ValueError, match="subset_name"):
        compare_models([rows()], subset=lambda keys: keys.target_time.notna())
