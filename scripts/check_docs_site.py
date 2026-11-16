"""Check generated API signatures, search topics and publication boundaries."""

import ast
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Text(HTMLParser):
    """Extract rendered text without depending on browser tooling."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def main() -> None:
    site = ROOT / "site"
    for path in site.rglob("*"):
        assert not any(
            name in path.as_posix()
            for name in (
                "AGENTS",
                "public-api-implementation-report",
                "preparation-report",
                "release-candidate-audit",
                "github-v0.1-checklist",
                "public-candidate-files",
            )
        ), path
    for page in ("benchmark", "config", "result", "discovery", "model-info"):
        html = (site / "reference" / page / "index.html").read_text(encoding="utf-8")
        assert 'class="doc-signature highlight"' in html, page
    source = ast.parse((ROOT / "src/solarcast/api.py").read_text(encoding="utf-8"))
    function = next(
        item
        for item in source.body
        if isinstance(item, ast.FunctionDef) and item.name == "benchmark"
    )
    html = (site / "reference/benchmark/index.html").read_text(encoding="utf-8")
    signature = html.split('class="doc-signature highlight"', 1)[1].split("</pre>", 1)[0]
    parser = Text()
    parser.feed(signature)
    text = "".join(parser.parts)
    for argument in function.args.kwonlyargs:
        assert argument.arg in text, argument.arg
    index = json.loads((site / "search/search_index.json").read_text(encoding="utf-8"))
    text = " ".join(document["text"] for document in index["docs"])
    for topic in ("benchmark", "LightGBM", "Chronos", "weather", "RMSE", "ForecastContext"):
        assert topic in text, topic
    print("Verified five generated signatures, benchmark parameters, search and site boundaries")


if __name__ == "__main__":
    main()
