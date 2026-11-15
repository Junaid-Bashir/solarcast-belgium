"""Public documentation checks must not depend on local excluded documents."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("link_private", [False, True])
def test_documentation_respects_package_exclusions(tmp_path: Path, link_private: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "check_docs.py").write_bytes((root / "scripts/check_docs.py").read_bytes())
    (tmp_path / "docs").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.sdist]\nexclude = ["/docs/private-notes.md"]\n'
    )
    # The private file exists and has a broken link; it is never a public source.
    (tmp_path / "docs/private-notes.md").write_text("[private draft](missing.md)\n")
    (tmp_path / "README.md").write_text(
        "[notes](docs/private-notes.md)\n" if link_private else "# Public documentation\n"
    )
    result = subprocess.run(
        [sys.executable, str(scripts / "check_docs.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert (result.returncode != 0) is link_private
    if link_private:
        assert "excluded local-only document" in result.stderr
