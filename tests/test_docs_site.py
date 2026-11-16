"""Offline executable tutorials and public-site exclusion boundaries."""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_beginner_and_custom_forecaster_examples() -> None:
    namespace: dict[str, object] = {}
    for name in (
        "getting-started/quickstart.md",
        "getting-started/first-benchmark.md",
        "contributing/forecaster.md",
    ):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        for code in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
            exec(compile(code, name, "exec"), namespace)
    results = namespace["results"]
    leaderboard = results.leaderboard()  # type: ignore[attr-defined]
    assert leaderboard.iloc[0]["n_scored"] == 192
    assert leaderboard.iloc[0]["rmse_mw"] == 0
    forecast = namespace["forecast"]
    assert len(forecast.predictions) == 4  # type: ignore[attr-defined]


def test_public_site_excludes_working_records_and_adapts_source_links() -> None:
    spec = importlib.util.spec_from_file_location("docs_hooks", ROOT / "scripts/docs_hooks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = [SimpleNamespace(src_uri=name) for name in module.EXCLUDED | {"index.md", "AGENTS.md"}]
    assert [file.src_uri for file in module.on_files(files, None)] == ["index.md"]
    source = ROOT / "docs/contributing/development.md"
    rendered = module.adapt_links("[Policy](../../CONTRIBUTING.md)", source)
    assert "github.com/Junaid-Bashir/solarcast-belgium/blob/main/CONTRIBUTING.md" in rendered
    assert (
        module.adapt_links("[Guide](../guide/models.md)", source) == "[Guide](../guide/models.md)"
    )
