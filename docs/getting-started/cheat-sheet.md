# API cheat sheet

```python
from solarcast import (
    benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    available_models,
    ModelInfo,
    __version__,
)
```

| I want to… | Use | Purpose |
| --- | --- | --- |
| Compare models | `benchmark()` | Run a bounded historical comparison |
| Configure dates and assumptions | `BenchmarkConfig` | Evaluation issues, lag, scoring and training settings |
| Inspect results | `BenchmarkResult` | Returned predictions, origins and metrics |
| See rankings | `results.leaderboard()` | Comparable scores alongside availability |
| Discover supported/installed models | `available_models()` | Dependency status without downloads |
| Understand discovery records | `ModelInfo` | Immutable identifiers and requirements |
| Record package version | `__version__` | Installed package version string |
| Control forecasting directly | [Advanced forecasting interface](../forecasting-interface.md) | Explicit requests and bounded context |

See [quickstart](quickstart.md) and [generated reference](../reference/index.md). `benchmark()` accepts implemented string identifiers, not arbitrary custom objects.
