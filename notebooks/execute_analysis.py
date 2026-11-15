"""Execute the real-data notebook; rendered output must stay in artifacts/analysis."""

import argparse
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="solarcast-analysis")
    args = parser.parse_args()
    source = root / "notebooks/01_real_belgian_forecast_analysis.ipynb"
    output = root / "artifacts/analysis/01_real_belgian_forecast_analysis/executed.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)

    def progress(cell: Any, cell_index: int, **kwargs: Any) -> None:
        print(f"Executed cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(source.parent)}},
        on_cell_executed=progress,
    )
    client.execute()
    nbformat.write(notebook, output)
    print(f"Executed all {len(notebook.cells)} cells successfully: {output}")


if __name__ == "__main__":
    main()
