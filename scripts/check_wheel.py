"""Install the local wheel in a temporary base-only venv and run the public quickstart.

Package installation may access PyPI for dependencies. The benchmark subprocess
blocks sockets, uses isolated Python, and runs outside the repository. No provider
data or model weights are acquired. The temporary environment is removed on exit.
"""

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    wheels = [args.wheel] if args.wheel else sorted((root / "dist").glob("*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file():
        parser.error("Provide one built wheel with --wheel, or build exactly one wheel in dist/")
    with tempfile.TemporaryDirectory(prefix="solarcast-wheel-") as temporary:
        work = Path(temporary)
        environment = work / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(executable),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(wheels[0].resolve()),
            ],
            check=True,
            cwd=work,
        )
        subprocess.run([str(executable), "-m", "pip", "check"], check=True, cwd=work)
        program = f"""
import importlib.metadata as md
from pathlib import Path
import runpy, socket, sys
import solarcast
assert Path(solarcast.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert solarcast.__version__ == md.version('solarcast-belgium') == '0.1.0'
assert set(solarcast.__all__) == {{
    '__version__', 'benchmark', 'available_models', 'BenchmarkConfig',
    'BenchmarkResult', 'ModelInfo',
}}
assert callable(solarcast.benchmark) and callable(solarcast.available_models)
for name in ('lightgbm','tfc-t0','chronos-forecasting','eccodes','pyarrow','shapely','pyproj'):
    try: md.version(name)
    except md.PackageNotFoundError: pass
    else: raise AssertionError('Unexpected optional distribution: '+name)
def blocked(*args, **kwargs):
    raise AssertionError('Public quickstart must be offline')
socket.socket.connect = blocked
socket.create_connection = blocked
runpy.run_path({str(root / "examples/quickstart.py")!r}, run_name='__main__')
assert not {{'torch','t0','chronos','lightgbm','eccodes'}} & set(sys.modules)
print('Clean base-only wheel quickstart passed; sockets blocked; site-packages verified')
"""
        subprocess.run([str(executable), "-I", "-c", program], check=True, cwd=work)


if __name__ == "__main__":
    main()
