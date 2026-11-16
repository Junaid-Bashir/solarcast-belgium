# Weather / NWP

PV depends on future radiation and clouds. Historical benchmarking must use a weather forecast vintage eligible before the PV issue time.

```text
historical NWP run -> availability policy -> latest complete eligible vintage
                                                      |
                                           future weather covariates
                                                      |
                                                 PV forecaster
```

Future observed weather and reanalysis cannot replace historical forecast weather.

The implemented source is operational NOAA/NCEP GFS 0.25° via NODD. It preserves run identity and radiation bounds, aligns surface DSWRF interval averages, entire-atmosphere TCDC and 2m temperature, and aggregates a 209-cell Belgian subset with pinned area weights.

**Init plus six hours is an availability approximation**, not certified historical evidence. Retrieval time is not original availability. Selection requires a complete eligible vintage. Models receive numerical covariates, never GRIB. Missing required NWP fails the whole origin with accounting.

Preparation needs optional tooling and local archives; the quickstart does not. See the [weather contract](../data/weather.md), [Weather-LightGBM](../models/weather-lightgbm.md) and [development comparison](../experiments/weather-lightgbm-v1.md).
