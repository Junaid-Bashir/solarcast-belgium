"""Check the installed package rather than injecting src into the import path."""

from importlib.metadata import distribution
from pathlib import Path

import solarcast


def test_installed_metadata_matches_public_version() -> None:
    """The public version comes from the installed distribution metadata."""
    metadata = distribution("solarcast-belgium")
    assert solarcast.__version__ == metadata.version
    assert metadata.metadata["Name"] == "solarcast-belgium"


def test_package_is_loaded_from_src_layout() -> None:
    """An editable installation exposes the intended source package."""
    assert solarcast.__file__ is not None
    repository = Path(__file__).resolve().parents[1]
    expected = repository / "src" / "solarcast" / "__init__.py"
    assert Path(solarcast.__file__).resolve() == expected.resolve()
