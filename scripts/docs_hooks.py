"""Adapt repository links and authenticated results without copying scientific docs."""

import csv
import re
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/Junaid-Bashir/solarcast-belgium/blob/main/"
EXCLUDED = {
    "README.md",
    "requirements.txt",
    "public-api-implementation-report.md",
    "release/preparation-report.md",
    "release/release-candidate-audit.md",
    "release/github-v0.1-checklist.md",
    "release/public-candidate-files.txt",
}


def on_files(files, config):
    """Exclude local working records even when present in an unclean checkout."""
    for file in list(files):
        if file.src_uri in EXCLUDED or Path(file.src_uri).name == "AGENTS.md":
            files.remove(file)
    return files


def adapt_links(markdown: str, source: Path) -> str:
    """Keep documentation links local; send repository source links to GitHub."""
    markdown = markdown.replace(
        "lightgbm.md#optional-history--nwp-configuration",
        "lightgbm.md#optional-history-nwp-configuration",
    )

    def replace(match: re.Match[str]) -> str:
        href = match.group(1)
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        target = (source.parent / parsed.path).resolve()
        if target == ROOT / "docs" / "README.md":
            import os

            replacement = Path(os.path.relpath(ROOT / "docs" / "index.md", source.parent))
            return (
                "]("
                + replacement.as_posix()
                + ("#" + parsed.fragment if parsed.fragment else "")
                + ")"
            )
        if target.is_relative_to(ROOT / "docs"):
            return match.group(0)
        if not target.is_relative_to(ROOT) or not target.exists():
            return match.group(0)  # MkDocs reports invalid links in strict mode.
        relative = target.relative_to(ROOT).as_posix()
        if relative.startswith(("artifacts/", "data/", ".venv/")) or relative == "AGENTS.md":
            raise ValueError("Non-public repository link in documentation")
        return (
            "]("
            + REPOSITORY
            + quote(relative)
            + ("#" + parsed.fragment if parsed.fragment else "")
            + ")"
        )

    return re.sub(r"\]\(([^)\s]+)\)", replace, markdown)


def on_page_markdown(markdown, page, config, files):
    """Render public aggregate CSV values and adapt links for the hosted site."""
    if "<!-- benchmark-table -->" in markdown:
        with (ROOT / "docs/benchmarks/weather-lightgbm-v1.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        table = "| Model | RMSE (MW) | MAE (MW) | Availability |\n| --- | ---: | ---: | ---: |\n"
        table += "\n".join(
            f"| `{row['model_id']}` | {float(row['rmse_mw']):.2f} | "
            f"{float(row['mae_mw']):.2f} | {float(row['forecast_availability']):.0%} |"
            for row in rows
        )
        markdown = markdown.replace("<!-- benchmark-table -->", table)
    return adapt_links(markdown, Path(page.file.abs_src_path))
