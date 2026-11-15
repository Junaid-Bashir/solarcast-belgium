"""Common-support model comparison over the existing backtest row schema."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from solarcast.backtesting import BacktestResult, BacktestSpec
from solarcast.evaluation.metrics import bias, mae, rmse, rmse_skill

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from solarcast.api import BenchmarkConfig
PrimaryMetric = Literal["rmse", "mae"]
Grouping = Literal["horizon", "local_hour", "month", "season"]
EvaluationInput = BacktestResult | pd.DataFrame
Subset = Callable[[pd.DataFrame], pd.Series]
KEYS = ["issue_time", "target_time"]


def _keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[KEYS])


def _validate(source: EvaluationInput) -> tuple[pd.DataFrame, str]:
    raw = source.predictions if isinstance(source, BacktestResult) else source
    required = {*KEYS, "model_id", "prediction_mw", "actual_mw", "horizon", "status"}
    if not raw.columns.is_unique or not required <= set(raw.columns) or raw.empty:
        raise ValueError("Evaluation requires nonempty backtest rows with unique required columns")
    frame = raw.copy(deep=True).reset_index(drop=True)
    for name in KEYS:
        if not isinstance(frame[name].dtype, pd.DatetimeTZDtype) or frame[name].isna().any():
            raise ValueError(f"{name} must contain non-null timezone-aware datetimes")
        frame[name] = frame[name].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    if (frame.target_time != frame.target_time.dt.floor("15min")).any():
        raise ValueError("target_time must be aligned to quarter-hour starts")
    if (frame.target_time <= frame.issue_time).any() or frame.duplicated(KEYS).any():
        raise ValueError("Targets must be future and issue/target pairs unique per model")
    if (
        not pd.api.types.is_timedelta64_dtype(frame.horizon.dtype)
        or frame.horizon.isna().any()
        or not (frame.horizon == frame.target_time - frame.issue_time).all()
    ):
        raise ValueError("horizon must equal target_time minus issue_time")
    identifiers = frame.model_id.drop_duplicates().tolist()
    if len(identifiers) != 1 or not isinstance(identifiers[0], str) or not identifiers[0].strip():
        raise ValueError("Each input must identify exactly one model/run")
    model_id = identifiers[0]
    if isinstance(source, BacktestResult) and source.model_id != model_id:
        raise ValueError("BacktestResult model_id does not match its rows")
    if "model_version" in frame and frame.model_version.dropna().nunique() > 1:
        raise ValueError("Mixed model versions need separate model IDs/runs")
    if frame.status.isna().any() or not frame.status.isin(["ok", "forecast_failed"]).all():
        raise ValueError("status must be ok or forecast_failed")
    for column in ("prediction_mw", "actual_mw"):
        values = frame[column]
        if (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
        ):
            raise ValueError(f"{column} must be real numeric and nullable")
        frame[column] = values.astype("Float64")
    frame["_actual_valid"] = np.isfinite(frame.actual_mw.to_numpy(dtype=float, na_value=np.nan))
    frame["_prediction_valid"] = np.isfinite(
        frame.prediction_mw.to_numpy(dtype=float, na_value=np.nan)
    ) & frame.status.eq("ok").to_numpy(dtype=bool)
    frame["_valid"] = frame._actual_valid & frame._prediction_valid
    frame["is_scorable"] = frame["_valid"]
    return frame.sort_values(KEYS, ignore_index=True), model_id


def _check_truth(frames: Sequence[pd.DataFrame]) -> None:
    combined = pd.concat([frame[KEYS + ["actual_mw", "_actual_valid"]] for frame in frames])
    # Compare finite values AND finite/missing availability on every shared key,
    # before excluding rows. Models cannot choose different scoring observations.
    if (combined.groupby(KEYS)["_actual_valid"].nunique() > 1).any():
        raise ValueError("Inconsistent actual availability on a shared issue/target key")
    finite = combined.loc[combined._actual_valid]
    if (finite.groupby(KEYS)["actual_mw"].nunique() > 1).any():
        raise ValueError("Inconsistent actual values on a shared issue/target key")


def _summaries(
    frame: pd.DataFrame,
    models: list[str],
    expected: pd.MultiIndex,
    support: pd.MultiIndex,
    reference_model: str | None,
    subset_name: str,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    reference = None
    if reference_model is not None:
        reference = frame.loc[frame.model_id == reference_model].set_index(KEYS)
    for model in models:
        own = frame.loc[frame.model_id == model]
        scored = own.loc[_keys(own).isin(support)].sort_values(KEYS)
        n_scored = len(scored)
        actual = scored.actual_mw.to_numpy(dtype=float)
        prediction = scored.prediction_mw.to_numpy(dtype=float)
        skill = float("nan")
        skill_status = "not_requested" if reference is None else "no_common_observations"
        if reference is not None and n_scored:
            baseline = reference.reindex(_keys(scored)).prediction_mw.to_numpy(dtype=float)
            skill = rmse_skill(actual, prediction, baseline)
            skill_status = "zero_reference_error" if np.isnan(skill) else "ok"
        summaries.append(
            {
                "model_id": model,
                "subset": subset_name,
                "comparison_policy": "common_support",
                "n_total": len(own),
                "n_expected": len(expected),
                "n_missing_rows": len(expected) - len(own),
                "n_actual_missing": int(own.actual_mw.isna().sum()),
                "n_actual_nonfinite": int((~own._actual_valid & own.actual_mw.notna()).sum()),
                "n_prediction_missing": int(own.prediction_mw.isna().sum()),
                "n_prediction_invalid": int(
                    (~own._prediction_valid & own.prediction_mw.notna()).sum()
                ),
                "n_valid_pairs": int(own._valid.sum()),
                "n_excluded_alignment": int(own._valid.sum()) - n_scored,
                "n_scored": n_scored,
                "n_origins": own.issue_time.nunique(),
                "n_scored_origins": scored.issue_time.nunique(),
                "n_failed_origins": own.loc[
                    own.status == "forecast_failed", "issue_time"
                ].nunique(),
                "forecast_availability": (
                    float(own._prediction_valid.sum()) / len(expected)
                    if len(expected)
                    else float("nan")
                ),
                "first_target": own.target_time.min(),
                "last_target": own.target_time.max(),
                "first_scored_target": scored.target_time.min(),
                "last_scored_target": scored.target_time.max(),
                "mae_mw": mae(actual, prediction) if n_scored else float("nan"),
                "rmse_mw": rmse(actual, prediction) if n_scored else float("nan"),
                "bias_mw": bias(actual, prediction) if n_scored else float("nan"),
                "reference_model": reference_model,
                "rmse_skill": skill,
                "skill_status": skill_status,
            }
        )
    return pd.DataFrame(summaries)


@dataclass(frozen=True)
class BenchmarkResult:
    """Numerical summaries plus the selected population and exact common support.

    Tables are owned output copies. Backtest specifications are retained when
    supplied; aligning observations cannot certify equal information access.
    Public benchmark() runs additionally populate config, metadata, complete
    predictions and origins; evaluator-only research calls retain their existing
    summary/aligned_predictions behavior and leave those new fields empty.
    """

    summary: pd.DataFrame
    aligned_predictions: pd.DataFrame
    primary_metric: PrimaryMetric
    reference_model: str | None
    subset_name: str
    backtest_specs: dict[str, BacktestSpec]
    _population: pd.DataFrame
    _expected: pd.MultiIndex
    _support: pd.MultiIndex
    config: "BenchmarkConfig | None" = None
    metadata: dict[str, object] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    origins: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def metrics(self) -> pd.DataFrame:
        """Owned common-support summary copy; independent populations are not mixed."""
        return self.summary.copy(deep=True)

    def leaderboard(self, primary_metric: PrimaryMetric | None = None) -> pd.DataFrame:
        """Lower MAE/RMSE wins; exact ties share rank, then sort by model ID."""
        metric = self.primary_metric if primary_metric is None else primary_metric
        if metric not in ("rmse", "mae"):
            raise ValueError("primary_metric must be rmse or mae")
        table = self.summary.copy(deep=True)
        table["rank"] = table[f"{metric}_mw"].rank(method="min").astype("Int64")
        table["ranking_metric"] = metric
        return table.sort_values([f"{metric}_mw", "model_id"], na_position="last").reset_index(
            drop=True
        )[
            [
                "rank",
                "model_id",
                "ranking_metric",
                "subset",
                "reference_model",
                "mae_mw",
                "rmse_mw",
                "bias_mw",
                "rmse_skill",
                "n_scored",
                "n_expected",
                "forecast_availability",
                "n_failed_origins",
            ]
        ]

    def grouped(self, by: Grouping) -> pd.DataFrame:
        """Same common-support scoring by horizon, Brussels hour, year-month or DJF/MAM/JJA/SON."""
        frame = self._population.copy(deep=True)
        local = frame.target_time.dt.tz_convert("Europe/Brussels")
        if by == "horizon":
            frame["_group"] = frame.horizon
        elif by == "local_hour":
            frame["_group"] = local.dt.hour
        elif by == "month":
            frame["_group"] = local.dt.strftime("%Y-%m")
        elif by == "season":
            frame["_group"] = np.array(["DJF", "MAM", "JJA", "SON"])[(local.dt.month % 12) // 3]
        else:
            raise ValueError("by must be horizon, local_hour, month or season")
        models = sorted(self.summary.model_id.tolist())
        groups: list[pd.DataFrame] = []
        for label, group in frame.groupby("_group", sort=True):
            expected = _keys(group).unique()
            support = expected.intersection(self._support)
            summary = _summaries(
                group, models, expected, support, self.reference_model, self.subset_name
            )
            summary.insert(0, by, label)
            groups.append(summary)
        if not groups:
            return pd.DataFrame(columns=[by, *self.summary.columns])
        return pd.concat(groups, ignore_index=True)


def compare_models(
    results: Sequence[EvaluationInput],
    *,
    primary_metric: PrimaryMetric = "rmse",
    reference_model: str | None = None,
    subset: Subset | None = None,
    subset_name: str | None = None,
) -> BenchmarkResult:
    """Compare raw backtest forecasts on common finite issue/target pairs only.

    The expected population is the union of supplied requested pairs, optionally
    filtered once by a model-independent subset. No postprocessing is applied.
    """
    if not results or primary_metric not in ("rmse", "mae"):
        raise ValueError("Provide results and primary_metric rmse or mae")
    frames: list[pd.DataFrame] = []
    models: list[str] = []
    specs: dict[str, BacktestSpec] = {}
    for source in results:
        frame, model = _validate(source)
        if model in models:
            raise ValueError("Duplicate model IDs: use one unambiguous run per ID")
        frames.append(frame)
        models.append(model)
        if isinstance(source, BacktestResult):
            specs[model] = source.spec
    if reference_model is not None and reference_model not in models:
        raise ValueError(f"Reference model {reference_model!r} is absent")
    _check_truth(frames)
    population = pd.concat(frames, ignore_index=True).sort_values(
        ["model_id", *KEYS], ignore_index=True
    )
    expected_frame = (
        population[KEYS + ["horizon"]]
        .drop_duplicates(KEYS)
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    name = "all_intervals"
    if subset is not None:
        if not isinstance(subset_name, str) or not subset_name.strip():
            raise ValueError("An explicit subset_name is required with subset")
        mask = subset(expected_frame.copy(deep=True))
        if (
            not isinstance(mask, pd.Series)
            or not mask.index.equals(expected_frame.index)
            or mask.isna().any()
            or not pd.api.types.is_bool_dtype(mask.dtype)
        ):
            raise ValueError("subset must return a non-null boolean Series with matching index")
        expected_frame = expected_frame.loc[mask]
        name = subset_name
    elif subset_name is not None:
        raise ValueError("subset_name requires a subset callable")
    expected = _keys(expected_frame)
    population = population.loc[_keys(population).isin(expected)].copy()
    support = expected
    for model in sorted(models):
        valid = population.loc[(population.model_id == model) & population._valid]
        support = support.intersection(_keys(valid))
    if not len(support):
        logger.warning("No common scorable observations; metrics and ranks are undefined")
    summary = _summaries(population, sorted(models), expected, support, reference_model, name)
    aligned = (
        population.loc[_keys(population).isin(support)]
        .drop(columns=["_valid", "_actual_valid", "_prediction_valid"])
        .reset_index(drop=True)
    )
    return BenchmarkResult(
        summary,
        aligned,
        primary_metric,
        reference_model,
        name,
        specs,
        population,
        expected,
        support,
    )


def evaluate_forecast(result: EvaluationInput) -> BenchmarkResult:
    """Evaluate a single existing backtest result; no cross-model ranking claim."""
    return compare_models([result])
