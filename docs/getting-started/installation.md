# Installation

Python 3.11–3.13 is supported. Production PyPI provides SolarCast Belgium 0.1.0.

```bash
python -m pip install solarcast-belgium
```

The **distribution** is `solarcast-belgium`; the **import package** is `solarcast`.

```python
import solarcast

print(solarcast.__version__)
```

Use a virtual environment. The core install runs persistence benchmarks without ML runtimes, weights or weather archives.

## Optional dependencies

| Extra | Purpose |
| --- | --- |
| `lightgbm` | Train LightGBM adapters |
| `weather` | Historical GRIB/Parquet weather tooling |
| `t0` | t0-gamma inference runtime |
| `chronos` | Chronos-2 inference runtime |
| `experiments` | Existing LightGBM experiment scripts and plotting |
| `dev` | Contributor checks and package building |

```bash
python -m pip install "solarcast-belgium[lightgbm]"
python -m pip install "solarcast-belgium[lightgbm,weather]"
```

Extras do not prepare archives or guarantee device compatibility and cached weights. See [models](../guide/models.md). For notebooks, install JupyterLab and Plotly separately; open the [real-data quickstart](https://github.com/Junaid-Bashir/solarcast-belgium/blob/main/notebooks/00_real_data_quickstart.ipynb).

Next: [offline quickstart](quickstart.md).
