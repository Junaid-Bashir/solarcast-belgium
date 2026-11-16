# Weather-LightGBM

Use `lightgbm_history_radiation_cloud` for locally trained LightGBM receiving history plus eligible forecast radiation/cloud. This 19-feature recipe shares the history-only adapter and backtesting engine.

Install `solarcast-belgium[lightgbm,weather]` for training and preparation. Supply a prepared local `WeatherStore`, chronological training issues and fit time. The benchmark does not fetch an archive.

Training reconstructs each issue's eligible vintage before joining fit-time-eligible labels. Missing required NWP fails an origin. See [LightGBM](lightgbm.md) and the [controlled weather report](../experiments/weather-lightgbm-v1.md) for configuration, frozen choices and limitations. The richer information set prevents an equal-input architecture comparison with history-only TSFMs.
