# Forecasting-oriented features

This layer produces deterministic tables, not forecasts. Four composable functions
separate future-known information from historical observations. No weather, Elia
forecasts, capacity-derived features, training or backtesting is included.

## Interface and output

```python
import pandas as pd
from solarcast.features import (
    calendar_features,
    solar_features,
    pv_lag_features,
    pv_rolling_features,
)

# pv is a normalized historical table from EliaPVClient or an equivalent fixture.
targets = pd.date_range("2026-06-09T00:00Z", periods=96, freq="15min")
cutoff = "2026-06-08T11:00Z"  # Explicit user-selected historical bound.

features = calendar_features(targets)
for group in (
    solar_features(targets),
    pv_lag_features(pv, targets, history_cutoff=cutoff),
    pv_rolling_features(pv, targets, history_cutoff=cutoff),
):
    features = features.merge(group, on="target_time", validate="one_to_one")
```

Input target times must be a timezone-aware, unique, non-null `DatetimeIndex`
aligned to 15-minute interval starts. Offset-aware indexes are converted to UTC;
naive input is rejected. Output preserves caller target order with a fresh
RangeIndex and `target_time` as `datetime64[ns, UTC]`. Inputs are not mutated.
Empty inputs return typed empty tables. Keep targets/labels separately; output
does not copy PV targets, capacity, source metadata, or unrequested input columns.

Historical functions require numeric nullable `pv_generation_mw` and aware unique
`target_time` columns. History may be unordered or have gaps; it is copied and
sorted for lookup. Invalid timestamps, duplicate keys, non-numeric PV and infinity
raise `ValueError`. Nulls remain missing; negatives are preserved. Capacity and
other columns stay in caller-owned input and do not influence features.

## Information availability

**Future-known:** calendar and solar features depend only on target timestamps and
documented fixed location/model settings. They work for arbitrary future target
grids without a history table.

**Historical-observed:** lag and rolling features require PV history and an explicit
`history_cutoff`. It accepts an aware datetime/Timestamp or offset-aware timestamp
string, aligned to a quarter hour. Date-only and naive values are rejected.

Eligibility requires `observation.target_time + 15 minutes <= history_cutoff`.
An interval average cannot be complete at its start. This is only a temporal
boundary, **not an inferred publication time**: Elia latency and revisions remain
unknown. The caller must also select an actually available snapshot or supply only
observations justified by a future availability policy. Never treat `retrieved_at`
as historical availability. Different simulated issue times require separate
cutoff-specific calls; this is not a backtesting implementation.

## Calendar features — future-known

Derived in `Europe/Brussels`, leaving the canonical UTC key unchanged:

| Columns | Convention / dtype |
|---|---|
| `local_hour`, `local_minute` | Clock fields, integer `int64` |
| `quarter_hour_in_hour` | 0, 1, 2, 3; not an index within a day |
| `day_of_week` | Monday=0 through Sunday=6 |
| `day_of_year`, `month` | 1-based local calendar fields |
| `utc_offset_minutes` | Distinguishes repeated autumn clock times, typically 60 or 120 |
| `time_of_day_sin`, `time_of_day_cos` | `float64`, phase `2π × local_minutes_since_midnight / 1440` |
| `day_of_year_sin`, `day_of_year_cos` | `float64`, phase `2π × (day_of_year−1 + local_minutes/1440) / days_in_year` |

`days_in_year` is 365 or 366. The annual encoding follows progress through each
calendar year; leap years slightly change phase on a given month/day. No weekend
flag is added because day of week already captures it.

Clock phase represents civil time, not elapsed progress through a 24-hour day.
Spring local days retain 92 rows and skip 02:00–02:45; autumn days retain 100 rows
and repeat those clock labels with different UTC times and offsets. Cyclical clock
features may repeat or jump across DST by design. Rows never collapse.

## Solar features — future-known

`BELGIUM_REFERENCE_LOCATION` is a project-selected rounded point near Brussels:
**latitude 50.85° north, longitude 4.35° east**. It is a convenient representative
point, not an official geographic center or a PV-capacity-weighted location.
Default altitude **0 m** is a sea-level computational reference, not a claim about
actual Brussels elevation. The immutable configuration is replaceable:

```python
from solarcast.features import SolarLocation, solar_features

solar = solar_features(targets, location=SolarLocation(50.85, 4.35, altitude_m=50.0))
# 50 m above is an example assumption, not verified site metadata.
```

Use pvlib's [solar-position wrapper](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html)
with `nrel_numpy`. Refraction uses fixed 12°C reference temperature and pressure
derived from configured altitude; these are deterministic assumptions, not weather
inputs. The angles below use apparent elevation/zenith consistently.

| Column | Units / meaning |
|---|---|
| `solar_apparent_elevation_deg` | Degrees above the horizon, `float64` |
| `solar_apparent_zenith_deg` | Degrees from zenith, `float64` |
| `solar_azimuth_deg` | Degrees clockwise from north, `float64` |
| `is_daylight` | Boolean: apparent elevation **strictly greater than 0°** |
| `clearsky_ghi_w_m2` | Haurwitz global horizontal irradiance reference, W/m², `float64` |

Angles and daylight are evaluated at **interval start**, not averaged across the
quarter-hour. A single point's horizon is not national sunrise/sunset. Do not use
this flag to zero or remove observed PV values.

The [Haurwitz model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.clearsky.haurwitz.html)
uses apparent zenith alone, so it can be calculated offline without turbidity
datasets or observed weather. It is a simple clear-sky reference. It does not
represent clouds, actual irradiance, plane-of-array radiation, or Belgian PV power;
no national calibration or physical plant simulation is claimed. More detailed
clear-sky models and spatial aggregation are deferred.

## Historical PV lags

`pv_lag_24h_mw`, `pv_lag_48h_mw`, `pv_lag_7d_mw` are nullable `Float64` MW values
looked up at exactly `target_time − 24 hours`, `− 48 hours`, `− 168 hours`.
Missing timestamps, null source values, and intervals ineligible at the cutoff
produce `pd.NA`. No row shifts, filling or interpolation are used.

These are **elapsed UTC durations**, not previous local calendar-day matches.
Around DST, 24 hours ago can have a different Brussels clock hour. This explicit
convention avoids ambiguous/nonexistent civil timestamps and avoids imposing a
one-hour civil-clock jump on daily solar alignment. It does not establish which
lag will perform best; evidence must come from later benchmarks. Local-calendar
lags are deferred until an ambiguity policy and scientific use case justify them.

For a day-ahead issue at 11:00, some next-day targets' 24-hour lags are later than
the cutoff and must remain missing, even if a downloaded retrospective table
contains those observations. Longer lags may still be eligible. Neither retrieval
time nor a lag's positive duration alone proves operational availability.

## Rolling context features

Two nullable `Float64` MW features are included:
`pv_trailing_24h_mean_mw` and `pv_trailing_24h_max_mw`.

For each target `t`, define `anchor = min(t, history_cutoff)`. Aggregate observation
interval starts in **`[anchor − 24 hours, anchor)`**, after cutoff filtering.
The left endpoint is included; the right endpoint and the current-target interval
are excluded. This is equivalent to pandas time-based rolling with `closed='left'`.

Require **96 present, non-null elapsed quarter-hours**. This is a fixed 24-hour
window, not a Belgian calendar day. Gaps, nulls or insufficient history leave both
statistics missing. Negative source values are retained. No partial-window mean
or maximum is silently substituted.

For all targets after a shared cutoff, these statistics describe the same historical
context and therefore repeat. For earlier targets the window ends at the target
instead, never including its own or later observations. This deliberate rule keeps
future targets from sliding a rolling window into unavailable data.

## Dependencies, checks and limitations

pvlib is now used directly for geometry and Haurwitz; NumPy is declared directly
for vectorized cycles and numeric validation (it was already transitive). Both
belong to the core deterministic feature layer. No ML stack is added. pvlib brings
its own scientific dependencies; no solar/weather data is downloaded by these
functions. Its missing typing marker is exempted only at the external `pvlib`
import boundary; project code remains strictly checked.

Offline tests cover calendar conversion, leap years, both DST days, solar physical
sanity, nulls/gaps, exact lag matching, cutoff/incomplete-interval exclusion,
rolling-window completeness, immutability and deterministic results. Run
`python examples/generate_features.py` for a synthetic offline composition example.

Results depend on installed pvlib/NumPy/timezone versions and configuration; record
these in future experiments. One national representative point is approximate.
Publication timing and revised snapshots remain unresolved. Feature generation
does not resolve leakage from later revisions or choose a training/evaluation policy.
