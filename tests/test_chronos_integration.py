"""Explicit network/real-weight test: SOLARCAST_CHRONOS_INTEGRATION=1 enables it."""

import os

import numpy as np
import pandas as pd
import pytest

from solarcast.models import ForecastContext, ForecastRequest
from solarcast.models.chronos import ChronosConfig, ChronosForecaster

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SOLARCAST_CHRONOS_INTEGRATION") != "1",
        reason="Real Chronos downloads/inference require explicit environment opt-in",
    ),
]


def test_real_chronos_cpu_short_forecast() -> None:
    pytest.importorskip("chronos")
    cutoff = pd.Timestamp("2025-06-15T09:30Z")
    times = pd.date_range(cutoff - pd.Timedelta("2D"), cutoff, freq="15min", inclusive="left")
    hours = times.tz_convert("Europe/Brussels").hour.to_numpy()
    history = pd.DataFrame(
        {
            "target_time": times,
            "pv_generation_mw": np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) * 1000,
        }
    )
    history.loc[[10, len(history) - 1], "pv_generation_mw"] = np.nan
    request = ForecastRequest(
        "2025-06-15T10:00Z", pd.date_range("2025-06-15T10:15Z", periods=4, freq="15min")
    )
    model = ChronosForecaster(ChronosConfig(context_length=192))
    result = model.predict(ForecastContext(request, history, history_cutoff=cutoff))
    assert len(result.predictions) == 4
    assert np.isfinite(result.predictions.prediction_mw).all()
    assert model.last_inference is not None and model.last_inference.rollout_steps == 7
    assert model.last_inference.missing_slots == 2
