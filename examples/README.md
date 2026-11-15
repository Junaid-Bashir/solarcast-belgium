# Examples

Run commands from the repository root after installing SolarCast. Synthetic
examples illustrate integration, not Belgian performance. Base installation means
`python -m pip install -e .`; add only the indicated extras.

## Public API

| Command | Prerequisites | Network and output |
|---|---|---|
| `python examples/quickstart.py` | Base | Canonical offline synthetic benchmark; prints only |
| `python examples/advanced_benchmark.py path/to/pv.csv` | `lightgbm`; normalized local PV | Explicit January training/February evaluation; prints only |
| `python examples/optional_tsfm_benchmark.py path/to/pv.csv --cache-dir path/to/hub-cache` | `t0`; local PV and pinned checkpoint | Local-only weights; two issues; prints only |

Local CSV examples require explicit offsets in `target_time`, numeric
`pv_generation_mw`, warm-up history and coverage of their stated dates. Adapt the
explicit schedules for your data. See `--help` and the [public API guide](../docs/public-api.md).
`quickstart_benchmark.py` is a compatibility entry point for the same canonical
implementation, not a second quickstart recipe.

Weather is advanced: prepare historical operational GFS runs separately and pass
a local WeatherStore. The [weather API example](../docs/public-api.md#weather-advanced-prepared-local-gfs-only)
states the preparation and availability assumptions; it is not a beginner download.

## Offline research examples

| Command | Extra | Purpose |
|---|---|---|
| `python examples/forecast_contract.py` | Base | Fake forecaster contract |
| `python examples/generate_features.py` | Base | Calendar/solar and bounded history features |
| `python examples/persistence_baselines.py` | Base | Synthetic last-value and elapsed seasonal lags |
| `python examples/backtest_baseline.py` | Base | Normal/spring/autumn target grids |
| `python examples/evaluate_baselines.py` | Base | Synthetic common-support leaderboard |
| `python examples/benchmark_lightgbm.py` | `lightgbm` | Synthetic chronological training/evaluation |

These default commands print results without saving or downloading data. The
LightGBM research example can opt into a local CSV or explicit bounded Elia fetch;
read `--help` before changing input modes.

## Acquisition and real-weight smoke examples

| Command | Prerequisites | Side effects |
|---|---|---|
| `python examples/fetch_elia_history.py` | Base; internet | Explicit one-hour Elia request; four quarters; prints only |
| `python examples/fetch_weather_vintage.py` | Base for no-op/help | Exits without network unless `--live` |
| `python examples/fetch_weather_vintage.py --live` | `weather`; internet | Bounded two-hour target-window GFS subset; writes ignored data/artifacts; `--full-day` is separate opt-in |
| `python examples/forecast_t0.py` | `t0`; pinned cached weights | Synthetic PV, real heavyweight inference; local-only by default |
| `python examples/forecast_chronos.py` | `chronos`; pinned cached weights | Synthetic PV, real heavyweight inference; local-only by default |

For TSFM smoke commands, `--allow-download` permits pinned Hub acquisition/cache
writes. `--local-files-only` remains accepted. `--input-csv` replaces synthetic PV
with the caller's normalized snapshot. `--output-dir` explicitly saves to a new
directory and refuses to overwrite it. `_tsfm_smoke.py` is a shared helper, not an
independent entry point. These are not operational forecast services.

The optional [analysis notebook](../notebooks/README.md) explicitly requires
original local frozen artifacts and fails if they are absent. No such artifacts
are required by the canonical quickstart or the public API itself.
