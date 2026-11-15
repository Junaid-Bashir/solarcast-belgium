"""Check local Markdown links and heading fragments, without external requests."""

import re
import tomllib
from fnmatch import fnmatchcase
from pathlib import Path
from urllib.parse import unquote, urlsplit


def anchors(path: Path) -> set[str]:
    counts: dict[str, int] = {}
    result: set[str] = set()
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if fenced:
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if match:
            slug = re.sub(r"[^\w\- ]", "", match[1].lower()).replace(" ", "-")
            occurrence = counts.get(slug, 0)
            counts[slug] = occurrence + 1
            result.add(slug if not occurrence else f"{slug}-{occurrence}")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        configuration = tomllib.load(stream)
    exclusions = configuration["tool"]["hatch"]["build"]["targets"]["sdist"].get("exclude", [])

    def excluded(path: Path) -> bool:
        relative = "/" + path.resolve().relative_to(root).as_posix()
        return any(fnmatchcase(relative, pattern) for pattern in exclusions)

    paths = [
        *root.glob("*.md"),
        *root.glob("docs/**/*.md"),
        *root.glob("examples/*.md"),
        *root.glob("notebooks/*.md"),
    ]
    failures = []
    checked = 0
    paths = [path for path in paths if not excluded(path)]
    for path in paths:
        content = path.read_text(encoding="utf-8")
        links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
        links += re.findall(r"^\[[^\]]+\]:\s+(\S+)", content, flags=re.MULTILINE)
        for link in links:
            parts = urlsplit(link.strip().strip("<>"))
            if parts.scheme or parts.netloc:
                continue
            checked += 1
            target = path.parent / unquote(parts.path) if parts.path else path
            relative = target.resolve().relative_to(root)
            if excluded(target):
                failures.append(f"{path.relative_to(root)} -> {link}: excluded local-only document")
                continue
            if relative.parts and relative.parts[0] in {"artifacts", "data", "dist", ".venv"}:
                failures.append(f"{path.relative_to(root)} -> {link}: local-only artifact link")
                continue
            if not target.exists():
                failures.append(f"{path.relative_to(root)} -> {link}: missing file")
            elif parts.fragment and target.suffix == ".md":
                if unquote(parts.fragment) not in anchors(target):
                    failures.append(f"{path.relative_to(root)} -> {link}: missing heading")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Checked {checked} local links in {len(paths)} Markdown files; all resolve")


if __name__ == "__main__":
    main()
