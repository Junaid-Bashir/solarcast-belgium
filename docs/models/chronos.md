# Chronos-2 zero-shot forecaster

The first TSFM adapter uses **`amazon/chronos-2`**, the primary selection in the
[landscape review](tsfm-landscape.md). It forecasts univariate historical national
PV without Belgian-specific training, covariates or a custom scaler. The existing
ForecastContext, Forecast, backtester and evaluator are unchanged. Technical integration and the [controlled zero-shot comparison](../experiments/tsfm-zero-shot-v1.md)
are separate forms of evidence.

## Upstream verification and licensing

Rechecked on **14 November 2026** against the [official repository](https://github.com/amazon-science/chronos-forecasting),
[model card](https://huggingface.co/amazon/chronos-2), [package metadata](https://pypi.org/project/chronos-forecasting/)
and [native pipeline implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py).
The current package is 2.3.2 and declares Python >=3.10, Torch >=2.2,<3,
Transformers >=4.41,<6, NumPy <3 and pandas <4. These declared ranges overlap
SolarCast's Python 3.11–3.13/core ranges. Installation on every combination is not
implied by a successful Python 3.12 Windows smoke.

The selected model remains unchanged from the TSFM landscape review. Its [pinned config](https://huggingface.co/amazon/chronos-2/blob/29ec3766d36d6f73f0696f85560a422f50e8498c/config.json)
declares 8192 context steps, 1024 native output steps and 21 quantiles including
P10/P50/P90. Default revision:
`29ec3766d36d6f73f0696f85560a422f50e8498c`.

[Code is Apache-2.0](https://github.com/amazon-science/chronos-forecasting/blob/main/LICENSE).
The publisher's model card declares Apache-2.0 for the weights; no separate weight
LICENSE file was present at the checked Hub path. Keep source/weight attribution
and applicable notices. SolarCast does not bundle or redistribute the weights.
This is the same licensing finding as the TSFM landscape review.

## Installation and optional imports

From this repository, install:

```powershell
python -m pip install -e ".[chronos]"
```

For the released package, use `python -m pip install "solarcast-belgium[chronos]"`.
For source development tests use `".[dev,chronos]"`.
The extra adds **`chronos-forecasting>=2.3.2,<2.4`**, a reviewed minor-series range
allowing compatible patch releases. Its heavy framework requirements stay optional.
Use a separate virtual environment if preserving a working baseline environment.

For CPU wheels, follow the current [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
The tested isolated setup is recorded below; GPU wheel selection depends on the
platform and driver. Do not install CUDA merely to run this adapter.

Importing `solarcast`, data, baseline, LightGBM or the Chronos adapter **module**
does not import Torch/Chronos/Transformers. Constructing `ChronosForecaster` checks
the optional imports and raises an actionable extra-installation hint if absent.
It does not download weights. `model.load()` or the first `predict()` resolves the
snapshot and constructs the pipeline once; later origins reuse it.

## Usage

```python
from solarcast.backtesting import BacktestSpec, EventTimeCutoff, backtest
from solarcast.models.chronos import ChronosConfig, ChronosForecaster

model = ChronosForecaster(ChronosConfig(context_length=1344, device="cpu"))
model.load()  # Optional explicit warm-up; the first predict also loads.
forecast = model.predict(context)  # context is an existing ForecastContext.

spec = BacktestSpec(
    issues,
    availability_policy=EventTimeCutoff("30min"),
    context_window="14D",
    on_error="continue",
)
result = backtest(model=model, data=pv_data, spec=spec)
```

The existing evaluator accepts `result` alongside persistence/LightGBM results.
No model-specific branch or metric is added. The configured human-readable model
ID is `chronos`; Forecast model_version records the upstream library version.
The adapter's immutable `metadata` separately records checkpoint/revision,
context limit, 15-minute frequency, P50 policy, fp32 and device. Use distinct run
IDs/metadata when comparing multiple context configurations in a future experiment.

The UTC context-grid/bridge operations live in the small
`models/_tsfm.py` helper shared with t0-gamma. The example entry point now delegates
to `examples/_tsfm_smoke.py`; its CLI and two-origin behavior are preserved.

## Context and missing values

`ChronosConfig.context_length` is a maximum number of **elapsed UTC quarter-hour
grid slots**, an integer 1–8192, default 1344 (14 ×24 ×4). A nominal 7-day limit is
672. It is not a required count of observed values and not a Belgian calendar-day
length. The backtester first applies its own history availability/context policy.

At prediction, the adapter uses supplied observations with starts in
`[history_cutoff - context_length*15min, history_cutoff)`. The input grid starts
at the earliest retained supplied row and ends at cutoff exclusive. It contains
at most the configured number of slots. Short supplied history is not extended
backward using inaccessible observations. The history-only experiment's issue-anchored 14D window
and 30-minute safety lag supply **1342** slots, not 1344; the adapter does not fill
the remaining earlier slots from outside context.

Only copied `target_time`/`pv_generation_mw` data are used. Optional engineered
`context.features`, monitored capacity, Elia forecasts and weather are ignored.
The native input is a single 1-D **float32** NumPy array inside a one-series list.
No physical frequency token is needed by this API: adjacency represents exactly
15 elapsed minutes because SolarCast validates and reconstructs the UTC grid.
No hourly resampling or local-naive DST conversion is performed.

Interior absent rows, explicit null values and trailing gaps are represented as
**NaN**, preserving their timestamps. Native Chronos scaling/missingness handling
is used; SolarCast performs no interpolation, forward-fill or zero-fill. The
forecast anchor remains cutoff even when the latest grid slots are missing.
Empty history, no retained rows, all-NaN input or values overflowing fp32 fail
clearly rather than emitting artificial forecasts. One finite observed slot is
the adapter's minimum; accepting it is not a scientific sufficiency claim.

`last_inference` records retained grid/observed/missing counts after successful
calls and is reset before each attempt. Continue-mode failures remain in the
backtest's standard origins/predictions tables; a failure cannot reuse stale
successful diagnostics. Logs also expose successful-call diagnostics.

## Horizon, exact targets and DST

The upstream model predicts consecutive steps immediately after its context.
SolarCast's requested target day generally starts much later. The adapter must
predict the **entire cutoff-to-last-target bridge**, then index exact requested
timestamps. Asking only for `len(target_times)` would be wrong when the target
day starts after the first predicted interval.

For the existing 11:00 Brussels/30-minute-lag policy:

| Requested next local day | Returned target count | Upstream bridge length |
|---|---:|---:|
| Normal | 96 | 150 |
| Spring-forward | 92 | 146 |
| Autumn-backward | 100 | 154 |

Sparse valid requests also work: unused intervening predictions are discarded by
timestamp offset, not relabelled as requested timestamps. Issue time and ordered
UTC target keys are preserved exactly. There is no model-specific DST limitation
within these lengths.

This first adapter intentionally **rejects bridge lengths above 1024** even though
upstream supports heuristic long-horizon rollout. It also checks the loaded
checkpoint's actual context/output limits. It never silently truncates context,
duplicates target values or creates missing forecast rows. Load/inference errors
retain the upstream exception as their cause and include useful checkpoint or
issue/cutoff/context/horizon information.

## Point output, normalization and future probabilistic work

The adapter calls `predict_quantiles` with `quantile_levels=[0.5]`, one series,
`batch_size=1`, `cross_learning=False` and strict prediction-length checking.
It selects the returned tensor's P50 explicitly and validates shape
`(1, full_bridge_length, 1)` and finiteness over the whole bridge. The convenience
point/“mean” return is unused. Upstream's current implementation itself notes that
its convenience “mean” is actually the median; SolarCast does not label P50 a
calibrated expectation.

No fit, gradient updates, Belgian checkpoint adaptation or extra custom scaling
is performed. The pretrained model runs in eval mode and the native predictor
uses no-grad inference. Raw negative predictions remain negative; nighttime is
not forced to zero. Forecast units remain MW.

Upstream offers 21 marginal quantiles and covariates, leaving room for later
P10/P50/P90 integration. The adapter exposes neither a probabilistic Forecast contract
nor weather inputs, trajectory sampling or calibration claims.

## Loading, cache, device and timing

The official Hugging Face `snapshot_download` mechanism resolves only
`config.json` and `model.safetensors` for the selected checkpoint and commit.
The default Hub cache is used unless `cache_dir=Path(...)` is supplied. First use
may download about **478 MB** of weights. Cached weights are reused; repository
Git ignores model/cache output locations. Keep any manually chosen cache directory
outside tracked source.

`local_files_only=True` disables network access for snapshot resolution and local
pipeline loading; a missing cache fails clearly. Revision must be an explicit
40-character commit hash, not moving `main`. The adapter's config is read-only
after construction so metadata cannot silently describe another load policy.

CPU/fp32 is the explicit default. `cuda` and `cuda:N` are supported configuration
values; unavailable CUDA raises rather than silently falling back. No auto-device,
MPS, quantization, compile backend or thread-count mutation is introduced.
GPU execution is documented upstream but is not claimed tested by a CPU smoke.

`cache_seconds` measures Hub cache resolution/download. `load_seconds` separately
measures local model construction and device transfer. `last_inference` measures
the synchronized upstream prediction and tensor-to-CPU conversion, excluding
loading, download and SolarCast grid construction. CUDA synchronization brackets
the measurement. The first prediction may include framework warm-up; report it
separately from later calls. Heavy optional imports are not included in these
three timings. Peak memory profiling is deferred rather than introducing a new
cross-platform profiler.

## Offline and manual tests

Normal CI uses fake upstream tensors/pipelines and does not download weights,
even when Torch happens to be installed. Tests cover bridge and P50 conversion,
latest-N context, missing/absent/trailing slots, ignored features, future-value
canaries, output validation, negatives, one-time loading, optional imports,
failure propagation/availability and DST/sparse targets.

Run offline checks:

```powershell
python -m pytest
python -m pytest tests/test_chronos.py -q
python -m pytest tests/test_chronos.py -k "canary or dependency or imports" -q
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

The real integration test is explicitly opt-in. It allows download/cache access,
uses a tiny representative profile and verifies a four-target finite CPU forecast:

```powershell
$env:SOLARCAST_CHRONOS_INTEGRATION = "1"
python -m pytest tests/test_chronos_integration.py -m integration -q
Remove-Item Env:SOLARCAST_CHRONOS_INTEGRATION
```

An example makes one short forecast and a **two-origin** backtest through the
existing engine/evaluator. It reports technical shape/availability/timing only:

```powershell
python examples/forecast_chronos.py --allow-download
python examples/forecast_chronos.py --input-csv data/elia_benchmark_v1/20250601T0000Z_20250701T0000Z.csv --output-dir artifacts/smoke/chronos_cpu
python examples/forecast_chronos.py --input-csv data/elia_benchmark_v1/20250601T0000Z_20250701T0000Z.csv --local-files-only
```

The default example profile is **synthetic**, not measured PV. An explicitly
provided CSV uses that local snapshot, without Elia network retrieval or fallback.
Existing output directories are protected from overwrite. Successful-origin
diagnostics include missing-slot counts; failures are reported separately, not
counted as missingness-free successes. Do not infer accuracy from these tiny runs.

## Actual Windows CPU verification — 2026-11-14

Tested on Windows 11 build 26200, Intel64 Family 6 Model 142, four PyTorch CPU
threads, Python 3.12.4, `chronos-forecasting` 2.3.2, `torch` 2.15.0+cpu,
`transformers` 5.23.0, NumPy 2.5.4 and pandas 3.1.1. This is one tested
combination, not validation of every supported Python version or GPU device.
An isolated environment left the base development environment free of Torch:

```powershell
.\.venv\Scripts\python.exe -m venv .venv/chronos-smoke
.\.venv\chronos-smoke\Scripts\python.exe -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\chronos-smoke\Scripts\python.exe -m pip install -e ".[dev,chronos]"
```

The opt-in native test passed, including an interior NaN and a trailing NaN
(one test, 9.01 seconds with cached weights). Initial weight retrieval and native
inference also succeeded without a compiler or WSL. Hugging Face warned about
anonymous requests and Windows symlink cache fallback; neither prevented execution.
Transformers reported the upstream-compatible `torch_dtype` keyword as deprecated;
loading float32 still succeeded. No access token or administrator change was needed.

The reproducible local-data smoke command was:

```powershell
.\.venv\chronos-smoke\Scripts\python.exe examples/forecast_chronos.py --input-csv data/elia_benchmark_v1/20250601T0000Z_20250701T0000Z.csv --issue 2025-06-16T11:00:00+02:00 --local-files-only --output-dir artifacts/smoke/chronos_cpu_full_context
```

The input SHA-256 was
`41d2bc8cd8eb9ca2366bdc94fca02536a1efe5be1e23809e76cd11f19e189a03`.
The ignored output directory contains `smoke.json`, `short_forecast.csv`,
`predictions.csv`, `origins.csv` and `evaluation.csv`. Both origins succeeded:
192/192 finite, scorable target predictions; zero failed origins; both histories
had 1,342 observed quarter-hours and zero missing slots. Each next-day forecast
used a 150-step rollout to select its 96 requested targets. The separate short
forecast selected four targets from seven rollout steps.

Cached snapshot resolution took 0.000715 seconds and local model loading took
0.681 seconds. Short inference took 0.205 seconds; the two day-ahead calls took
0.186 and 0.156 seconds. These are single-run observations, exclude interpreter
startup/imports and initial download, and are not a performance benchmark.
No accuracy conclusion follows from this smoke run.

Offline verification passed: 302 tests, one opt-in integration test skipped;
all 28 adapter tests also passed with the optional runtime installed. Ruff,
format checking, strict base-environment mypy, dependency checks and wheel/sdist
build passed. Offline tests verify 92/96/100 target DST days; native real-model
DST runs and memory profiling were not performed.

## Known limitations

Historical Elia publication delay/revisions and TSFM pretraining contamination
remain unresolved. A safety lag does not reconstruct original information vintages.
The model has no observed weather, capacity evolution or deterministic solar
covariates in this track. Its default context cap is not evidence that short
context is sufficient for national PV. Median forecasts are not optimized means
for RMSE. Native gap handling does not guarantee useful forecasts with sparse
history. Reusing an instance assumes sequential calls; concurrent use/timing
mutations are not supported. Broader hardware, runtime-version, context-length
and controlled accuracy benchmarking remain later work.

## Public smoke acquisition policy

The standalone example is local-only by default; `--allow-download` deliberately
permits pinned checkpoint acquisition. Install `.[chronos]` first. This example
change does not alter adapter defaults or frozen inference. See the
[example catalog](../../examples/README.md).
