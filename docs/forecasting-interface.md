# Minimal forecasting interface

SolarCast uses a structural `Forecaster` protocol and three small typed objects:
`ForecastRequest`, `ForecastContext`, and `Forecast`. Stateless
[persistence baselines](baselines.md) now use this contract. Training algorithms,
registry and serialization are not implemented yet. Deterministic
[evaluation](evaluation.md) consumes the resulting numerical tables. The
[rolling backtester](backtesting.md) now constructs these contexts at each origin.

## Inference capability

```python
class Forecaster(Protocol):
    @property
    def model_id(self) -> str: ...

    def predict(self, context: ForecastContext) -> Forecast: ...
```

Adapters need not inherit from anything. A string attribute also satisfies the
read-only `model_id` property. Use a stable human-readable ID; no global registry
is required. Return a Forecast bound to `context.request` with that same ID.

The context carries its request, so there are not two independently supplied
request/context arguments to disagree. This is a deliberate simplification of
the conceptual `predict(request, context)` API. A future runner can call every
adapter through `model.predict(context)` without model-name branches. Adapters
must not fetch hidden observations during prediction.

## Request: issue, targets and horizons

```python
request = ForecastRequest(
    issue_time="2026-06-10T09:07Z",
    target_times=pd.date_range("2026-06-10T22:00Z", periods=4, freq="15min"),
)
```

`issue_time` is the simulated generation time. Accept an aware datetime/Timestamp
or offset-aware string; reject date-only, naive and missing values. Issue time
does not have to be aligned to a quarter hour or use a prescribed local issue hour.

`target_times` must be an aware, nonempty, unique and strictly increasing pandas
DatetimeIndex of quarter-hour interval starts, strictly **after** the issue.
Normalize to UTC without sorting or deduplicating invalid requests. Targets may
have gaps; the contract does not assume a fixed-length or consecutive forecast.
`request.horizons` derives elapsed `target_time - issue_time` durations.

Belgian local target days become explicit UTC grids containing 92, 96 or 100
quarter-hours. Repeated autumn local clocks remain distinct UTC instants.
Generating those local-day grids is the caller's concern; no day-ahead schedule
or 96-step assumption is encoded in the model contract.

## Context: supplied model inputs

```python
context = ForecastContext(
    request,
    pv_history,  # normalized history, explicitly filtered by the caller
    history_cutoff="2026-06-10T09:00Z",
    features=calendar_features(request.target_times),  # optional
)
```

The context contains:

- The request, including issue and target times.
- An explicit aware, quarter-hour `history_cutoff <= issue_time`.
- Historical `target_time` and nullable numeric `pv_generation_mw`, sorted in UTC.
- An optional already-computed feature DataFrame matching request targets exactly,
  in order. Null historical features are allowed; models decide how to handle them.

All history intervals must finish by the cutoff:
`history.target_time + 15 minutes <= history_cutoff`. A violation raises instead
of silently filtering. Duplicate, naive or misaligned history and infinite PV
values raise. Null and negative observations are retained. Empty typed history is
valid for adapters that do not need it.

The model-facing history deliberately projects only target and PV columns from
normalized ingestion data. Capacity and source/retrieval provenance stay with the
caller's original table and future run metadata. Explicit feature tables reject
raw `pv_generation_mw` labels and `monitored_capacity_mw`. No features are computed
automatically; feature derivation and information access remain caller obligations.
The contract cannot detect leakage hidden in an arbitrary derived column.

Optional `weather: WeatherCovariates | None` now carries seven complete numerical
weather features and typed selected-run availability provenance. It must match the
request issue and target times exactly. The backtester selects a complete eligible
vintage before model invocation; see the [weather contract](data/weather.md#22-historical-gfs-ingestion).
History-only models continue with `weather=None`. Observed future labels, scoring
data, raw weather/Elia responses, GPU settings and untyped metadata bags are excluded.

### Publication availability is still unknown

Interval completion is necessary, not proof of publication. Historical Elia latency,
revisions and capacity vintages remain unresolved. No `available_at` is fabricated
and retrieval time is not an availability policy. A future backtester must select
eligible versions using a defensible policy **before** constructing context and
computing historical features. The context enforces only its explicit temporal
bound and alignment, not an operational-realism claim.

## Result: numerical table plus metadata

```python
result = Forecast(
    request,
    pd.DataFrame({"target_time": request.target_times, "prediction_mw": 1.0}),
    model_id="fake_example",
    model_version="demo-v1",
)
```

This constant output only illustrates the contract. `Forecast.predictions` holds
exactly `target_time` (`datetime64[ns, UTC]`) and `prediction_mw` (`float64`). The
target grid must match the request exactly, in order; no missing/extra/duplicate
targets, implicit sorting, or silent alignment is allowed. Offset-aware result
times are normalized to UTC; naive times are rejected.

Predictions must be real numeric, finite and **non-missing**. Null, NaN, infinity,
booleans and numeric strings raise; no filling occurs. Negative forecasts are
preserved for later evaluation/postprocessing. A model unable to predict must
fail explicitly; a future backtester should record that failure and coverage.

Constant metadata lives once on the object: request/issue time, model ID, optional
model version, target variable (default `pv_generation`) and unit `MW`. Belgium is
not hard-coded into the result. Unit metadata is intentionally restricted to MW
because the prediction column is named `prediction_mw`; there is no unit conversion.

`result.to_frame()` returns a flat copy with issue time, elapsed horizon, model ID,
model version, target variable and unit repeated for export/comparison convenience.
Core metadata objects are frozen; input tables and public table/index accessors
are defensively copied. Feature tables should contain ordinary scalar values;
pandas deep copies do not recursively clone arbitrary nested Python objects.

## Stateless, pretrained and trainable models

Only inference is common today. Persistence and zero-shot TSFMs need no dummy
`fit()`. Trained and fine-tuned adapters satisfy exactly the same Forecaster protocol
after their model-specific preparation. Predicting before required preparation
should raise a clear model-specific error.

A generic training-data contract is **deliberately deferred**, so no
`TrainableForecaster` protocol or misleading generic `fit(DataFrame)` promise is
exported. Supervised labels, issue-specific features and fine-tuning examples need
scientific decisions that require explicit study. Training methods may exist on
concrete adapters without becoming part of the inference interface. Future
retraining orchestration must be designed explicitly; it is not solved here.
The [LightGBM adapter](models/lightgbm.md) has a small model-specific supervised
dataset builder and fit method; the common inference protocol is unchanged.

## Fake adapters and typing

```python
class StatelessFake:
    model_id = "stateless_fake"

    def predict(self, context: ForecastContext) -> Forecast:
        table = pd.DataFrame(
            {
                "target_time": context.request.target_times,
                "prediction_mw": 1.0,
            }
        )
        return Forecast(context.request, table, model_id=self.model_id)


model: Forecaster = StatelessFake()  # checked structurally by mypy
result = model.predict(context)
```

Tests also include a TrainableFake with a model-specific `fit` state gate. It is
assigned to `Forecaster` and invoked by the same typed function as StatelessFake.
Production baselines are separate from these fake examples. No training algorithm
is implemented. Protocol conformance is
checked statically; runtime isinstance checks are not a signature validator and
are not used. Run `python examples/forecast_contract.py` for the offline smoke.

## Future adapters and probabilistic output

A zero-shot TSFM adapter translates the supplied history, explicit target grid and
optional features to its own frequency/context constraints. It can reject
unsupported context lengths, irregular grids or missing values clearly. Loading
pretrained weights and recording upstream identity/training cutoffs belong to
that adapter and future experiment metadata, not to the shared inference signature.

Probabilistic output is not implemented. The typed result can later add an optional
separate long quantile table (`target_time`, `quantile`, `prediction_mw`) without
changing request/context or deterministic prediction columns. Quantile validation
and genuinely quantile-only model support must be designed in that future phase;
do not fabricate a point forecast from samples or assume it equals P50 now.
Ensembles will likewise need justified component-forecast inputs later. No unused
probability or ensemble machinery is added in anticipation.
