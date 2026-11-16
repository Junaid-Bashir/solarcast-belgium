# Development setup

Clone → install development tools → test → change → check → open a pull request.

```bash
git clone https://github.com/Junaid-Bashir/solarcast-belgium.git
cd solarcast-belgium
python -m venv .venv
```

Activate `.venv` using your shell, then:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
python scripts/check_docs.py
```

Normal CI is offline; real-weight tests need explicit opt-in. Keep datasets, weights and generated runs outside Git. Read the [contribution policy](../../CONTRIBUTING.md).

## Preview documentation

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
python -m mkdocs build --strict
```

Open the URL printed by MkDocs. Existing Markdown stays in place; reference pages are generated from source. Main deploys through the separate Pages workflow; pull requests build only. Package publishing is a separate owner-authorized process.
