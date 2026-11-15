"""Offline public workflow, optional boundaries and fair comparison regression tests."""

import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pandas as pd
import pytest

from solarcast import BenchmarkConfig, BenchmarkResult, api, available_models, benchmark
from solarcast.backtesting import BacktestOriginError, backtest
from solarcast.evaluation import compare_models
from solarcast.models import Forecast, ForecastContext, LastValuePersistenceForecaster


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any accidental acquisition in a public benchmark test is a failure."""

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Public API tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_time": pd.date_range("2025-01-01", "2025-01-15", freq="15min", tz="UTC"),
            "pv_generation_mw": 100.0,
        }
    )


def config() -> BenchmarkConfig:
    return BenchmarkConfig(
        pd.date_range("2025-01-09 11:00", periods=2, freq="D", tz="Europe/Brussels"),
        observation_safety_lag="30min",
    )


def test_public_smoke_matches_research_and_does_not_mutate_input() -> None:
    source = data()
    original = source.copy(deep=True)
    cfg = replace(config(), reference_model="seasonal_persistence_48h")
    result = benchmark(
        data=source, config=cfg, models=["seasonal_persistence_48h", "last_value_persistence"]
    )
    assert isinstance(result, BenchmarkResult)
    table = result.leaderboard()
    assert table.n_scored.tolist() == [192, 192]
    assert table["rank"].tolist() == [1, 1]
    assert table.rmse_mw.eq(0).all()
    assert table.rmse_skill.isna().all()  # Zero-reference error is undefined.
    assert result.metrics.comparison_policy.eq("common_support").all()
    assert len(result.predictions) == 384 and len(result.origins) == 4
    assert result.config is not cfg and result.config is not None
    assert result.config.observation_safety_lag == "30min"
    assert result.metadata["target"] == "belgium_pv"
    assert result.metadata["effective_config"]["context_window"] == "14D"  # type: ignore[index]
    research = compare_models(
        [backtest(model=LastValuePersistenceForecaster(), data=source, spec=cfg.spec())]
    )
    single = benchmark(data=source, config=config(), models=["last_value_persistence"])
    pd.testing.assert_frame_equal(single.leaderboard(), research.leaderboard())
    pd.testing.assert_frame_equal(source, original)
    assert result.metadata["data_sha256"] == single.metadata["data_sha256"]


def test_discovery_and_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    original_version = version

    def missing(name: str) -> str:
        if name in {"tfc-t0", "torch"}:
            raise PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(api, "version", missing)
    discovered = {record.model_id: record for record in available_models()}
    assert discovered["seasonal_persistence_48h"].status == "built_in"
    assert discovered["t0"].status == "unavailable"
    assert {"tfc-t0", "torch"} <= set(discovered["t0"].missing_dependencies)
    with pytest.raises(ImportError, match=r"\.\[t0\]"):
        benchmark(data=data(), config=config(), models=["t0"])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"target": "wind"}, "Supported targets: belgium_pv"),
        ({"models": ["best"]}, "Unknown SolarCast model 'best'"),
        ({"models": []}, "nonempty"),
        ({"models": "t0"}, "sequence"),
        ({"models": ["t0", "t0"]}, "unique"),
        ({"models": ["lightgbm"]}, "training_issue_times"),
        ({"models": ["lightgbm_history_radiation_cloud"]}, "WeatherStore"),
    ],
)
def test_actionable_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        benchmark(data=data(), config=config(), **kwargs)


@pytest.mark.parametrize("day,count", [("2025-03-29", 92), ("2025-10-25", 100)])
def test_public_dst(day: str, count: int) -> None:
    issue = pd.Timestamp(day, tz="Europe/Brussels") + pd.Timedelta(hours=11)
    source = pd.DataFrame(
        {
            "target_time": pd.date_range(
                issue - pd.Timedelta("10D"), issue + pd.Timedelta("3D"), freq="15min"
            ),
            "pv_generation_mw": 100.0,
        }
    )
    result = benchmark(
        data=source,
        config=BenchmarkConfig(pd.DatetimeIndex([issue]), observation_safety_lag="30min"),
    )
    assert len(result.predictions) == count
    assert result.leaderboard().n_scored.iloc[0] == count


def test_missing_forecast_shrinks_every_model_support_and_retains_failures() -> None:
    source = data()
    # Exact seasonal lookup for the first target day requires this Jan 8 quarter.
    source.loc[source.target_time == pd.Timestamp("2025-01-08T12:00Z"), "pv_generation_mw"] = float(
        "nan"
    )
    result = benchmark(
        data=source, config=config(), models=["seasonal_persistence_48h", "last_value_persistence"]
    )
    table = result.leaderboard().set_index("model_id")
    assert table.n_scored.eq(96).all()
    assert table.loc["seasonal_persistence_48h", "forecast_availability"] == 0.5
    assert table.loc["last_value_persistence", "forecast_availability"] == 1
    assert table.loc["seasonal_persistence_48h", "n_failed_origins"] == 1
    assert len(result.predictions) == 384
    assert result.origins.error_type.eq("ForecastDataError").sum() == 1
    with pytest.raises(BacktestOriginError) as failure:
        benchmark(data=source, config=replace(config(), on_error="strict"))
    assert isinstance(failure.value.__cause__, ValueError)


def test_daylight_is_explicit_and_does_not_drop_raw_rows() -> None:
    result = benchmark(data=data(), config=replace(config(), subset="reference_point_daylight"))
    assert 0 < result.leaderboard().n_scored.iloc[0] < 192
    assert len(result.predictions) == 192
    assert result.metrics.subset.eq("reference_point_daylight").all()


def test_invalid_training_splits_and_fit_label_boundary() -> None:
    train = pd.date_range("2025-01-07T10:00Z", periods=2, freq="D")
    with pytest.raises(ValueError, match="both"):
        replace(config(), training_issue_times=train)
    with pytest.raises(ValueError, match="training issues"):
        replace(config(), training_issue_times=train, fit_time="2025-01-10T00:00Z")
    if next(m for m in available_models() if m.model_id == "lightgbm").status == "installed":
        with pytest.raises(ValueError, match="label cutoff"):
            benchmark(
                data=data(),
                models=["lightgbm"],
                config=replace(config(), training_issue_times=train, fit_time="2025-01-09T08:00Z"),
            )


def test_tsfm_orchestration_fake_runtime_and_future_pv_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[float] = []

    @dataclass
    class Metadata:
        revision: str = "fake offline runtime"

    class Fake:
        model_id = "t0"
        metadata = Metadata()

        def __init__(self, settings: Any) -> None:
            assert settings.local_files_only is True

        def predict(self, context: ForecastContext) -> Forecast:
            seen.append(float(context.history.pv_generation_mw.max()))
            return Forecast(
                context.request,
                pd.DataFrame(
                    {
                        "target_time": context.request.target_times,
                        "prediction_mw": -1.0,
                    }
                ),
                model_id=self.model_id,
            )

    monkeypatch.setattr(api, "T0Forecaster", Fake)
    monkeypatch.setattr(
        api, "available_models", lambda: (api.ModelInfo("t0", "installed", "t0", (), False, False),)
    )
    source = data()
    source.loc[source.target_time >= config().issue_times[0], "pv_generation_mw"] = 999999.0
    result = benchmark(data=source, config=config(), models=["t0"])
    assert seen[0] == 100.0 and seen[1] == 999999.0
    assert result.predictions.prediction_mw.eq(-1).all()
    assert result.predictions.actual_mw.eq(999999).all()


def test_minimal_import_and_discovery_do_not_import_optional_runtimes() -> None:
    program = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'pyarrow':
            raise ModuleNotFoundError(fullname)  # pandas probes its optional acceleration.
        if fullname.split('.')[0] in {'torch','t0','chronos','transformers','lightgbm',
                                      'eccodes','shapely','pyproj'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from solarcast import benchmark, available_models
available_models()
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_lightgbm_public_fit_once_before_evaluation() -> None:
    pytest.importorskip("lightgbm")
    result = benchmark(
        data=data(),
        models=["lightgbm"],
        config=replace(
            config(),
            training_issue_times=pd.date_range("2025-01-03T10:00Z", periods=3, freq="D"),
            fit_time="2025-01-07T08:00Z",
        ),
    )
    assert result.leaderboard().n_scored.iloc[0] == 192
    assert result.origins.status.eq("ok").all()
    info = result.metadata["model_metadata"]["lightgbm"]["training"]  # type: ignore[index]
    assert info["n_rows"] == 288
    assert info["last_training_target"] < info["fit_time"]
