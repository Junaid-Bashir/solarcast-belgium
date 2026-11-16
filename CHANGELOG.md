# Changelog

## 0.1.0 — 2026-11-16

- Add normalized historical Elia national PV ingestion, source provenance and
  non-destructive schema/time/DST validation.
- Add deterministic calendar/solar features and explicitly cutoff-limited PV lags
  and rolling features.
- Define typed forecasting requests, bounded contexts, results and a structural
  inference protocol shared by all implemented adapters.
- Implement last-value and exact elapsed seasonal persistence, sequential rolling
  backtests, failed-origin retention and explicit availability approximations.
- Add raw-output MAE/RMSE/bias, explicit-reference RMSE skill, common-support
  comparisons and grouped evaluation.
- Add optional LightGBM with origin-aware chronological training and fit-time guards.
- Add optional pinned Chronos-2 and t0-gamma adapters with raw native P50,
  cutoff-to-target rollout alignment, missing-slot preservation and offline tests.
- Add bounded historical operational GFS ingestion, immutable local caches,
  variable-aware alignment, area-weighted features and eligible-vintage selection.
- Add weather-informed LightGBM recipes, vintage-correct training, missing-weather
  accounting and an explicit audited GRIB-packing precision option with strict defaults.
- Document known-period Belgian PV development comparisons, immutable input/source
  provenance, scientific limitations and the need for fresh confirmation.
- Add the public benchmark/configuration/discovery API, inspectable result metadata,
  canonical offline quickstart and advanced examples with optional dependencies.
- Add a clean-source optional interactive analysis notebook for separately prepared
  local real-data bundles; keep executed output and research caches outside Git.
