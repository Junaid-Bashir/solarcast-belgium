"""Exercise the actual public README example, without making assertions about prose."""

import argparse
import re
import runpy
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from solarcast import BenchmarkResult


def test_readme_python_example_is_offline_and_returns_real_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("README example attempted network access")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    source = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    match = re.search(r"```python\n(.*?)```", source, flags=re.DOTALL)
    assert match is not None
    namespace: dict[str, Any] = {}
    exec(compile(match[1], "README.md", "exec"), namespace)
    result = namespace["results"]
    assert isinstance(result, BenchmarkResult)
    assert result.leaderboard().n_scored.eq(192).all()
    assert result.leaderboard().forecast_availability.eq(1).all()


@pytest.mark.parametrize(
    "flags,local_only", [([], True), (["--local-files-only"], True), (["--allow-download"], False)]
)
def test_real_weight_example_requires_download_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    local_only: bool,
) -> None:
    """Inspect actual parsed options before any model construction or acquisition."""
    helper = Path(__file__).resolve().parents[1] / "examples/_tsfm_smoke.py"
    main = runpy.run_path(str(helper))["main"]
    original = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        def __init__(self, settings: argparse.Namespace) -> None:
            self.settings = settings

    def capture(
        parser: argparse.ArgumentParser,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        raise Parsed(original(parser, args, namespace))

    monkeypatch.setattr(sys, "argv", ["forecast_t0.py", *flags])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(Parsed) as caught:
        main("t0")
    assert caught.value.settings.local_files_only is local_only
