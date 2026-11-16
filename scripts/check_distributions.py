"""Inspect built wheel/sdist contents without extracting or publishing anything."""

import argparse
import json
import re
import tarfile
import tomllib
import zipfile
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath


def inspect(entries: list[tuple[str, bytes]], *, sdist: bool) -> None:
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as stream:
        configuration = tomllib.load(stream)
    exclusions = configuration["tool"]["hatch"]["build"]["targets"]["sdist"].get("exclude", [])
    allowed_roots = {"src", "tests", "docs", "examples", "scripts", "configs", "notebooks"}
    root_files = {
        ".gitignore",
        ".gitattributes",
        "README.md",
        "CONTRIBUTING.md",
        "CITATION.cff",
        "CHANGELOG.md",
        "SECURITY.md",
        "THIRD_PARTY.md",
        "LICENSE",
        "pyproject.toml",
        "mkdocs.yml",
        "PKG-INFO",
    }
    fixtures = {f"tests/fixtures/weather/constant_{v}.grib2" for v in ("DSWRF", "TCDC", "TMP")}
    credential = re.compile(
        rb"ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|"
        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
    )
    for name, content in entries:
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts, name
        assert len(content) < 2_000_000, f"Unexpected large member: {name}"
        if sdist:
            assert not any(fnmatchcase("/" + name, pattern) for pattern in exclusions), name
            assert path.parts[0] in allowed_roots or name in root_files, name
        else:
            assert path.parts[0] == "solarcast" or path.parts[0].endswith(".dist-info"), name
        assert not set(path.parts) & {
            "artifacts",
            "data_cache",
            "model_weights",
            "checkpoints",
            ".venv",
            "__pycache__",
            ".git",
            ".ipynb_checkpoints",
            "node_modules",
        }, name
        assert path.suffix not in {
            ".safetensors",
            ".pt",
            ".pth",
            ".pkl",
            ".parquet",
            ".zip",
            ".log",
            ".pyc",
        }, name
        if path.suffix == ".ipynb":
            assert sdist and name in {
                "notebooks/00_real_data_quickstart.ipynb",
                "notebooks/01_real_belgian_forecast_analysis.ipynb",
            }, name
            notebook = json.loads(content)
            assert not any(cell.get("outputs") for cell in notebook["cells"]), name
            assert not any(cell.get("execution_count") is not None for cell in notebook["cells"]), (
                name
            )
        assert path.suffix != ".grib2" or name in fixtures, name
        assert not credential.search(content), f"Credential pattern in {name}"
        assert not re.search(rb"[A-Za-z]:[\\/]Users[\\/]|/(?:Users|home)/[A-Za-z]", content), name
    names = {name for name, _ in entries}
    if sdist:
        assert {
            "README.md",
            "SECURITY.md",
            "THIRD_PARTY.md",
            "examples/quickstart.py",
            "src/solarcast/api.py",
        } <= names
    else:
        assert {
            "solarcast/api.py",
            "solarcast/py.typed",
            "solarcast/data/weather/belgium.geojson",
        } <= names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheels = sorted(args.dist.glob("*.whl"))
    sources = sorted(args.dist.glob("*.tar.gz"))
    if not wheels or not sources:
        parser.error("Build both wheel and sdist first: python -m build")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            entries = [
                (name, archive.read(name)) for name in archive.namelist() if not name.endswith("/")
            ]
        inspect(entries, sdist=False)
        print(f"Inspected {wheel.name}: {len(entries)} intended members")
    for source in sources:
        with tarfile.open(source) as archive:
            entries = []
            for member in archive.getmembers():
                assert not member.issym() and not member.islnk(), member.name
                if member.isfile():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    name = str(PurePosixPath(*PurePosixPath(member.name).parts[1:]))
                    entries.append((name, stream.read()))
        inspect(entries, sdist=True)
        print(f"Inspected {source.name}: {len(entries)} intended members")


if __name__ == "__main__":
    main()
