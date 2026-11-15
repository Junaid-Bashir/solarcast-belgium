"""Checksummed immutable cropped-native Parquet bundles and JSON provenance."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .types import ForecastRun, WeatherError


def digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class WeatherCache:
    """One directory per retrieval identity; never overwrite a source version.

    Interrupted/altered entries are explicit errors. Write in a sibling temporary
    directory, then rename atomically. A racing writer cannot replace an entry.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, identity: dict[str, Any]) -> ForecastRun | None:
        entry = self.root / digest(identity)
        if not entry.exists():
            return None
        try:
            manifest = json.loads((entry / "manifest.json").read_text(encoding="utf8"))
            raw = (entry / "native.parquet").read_bytes()
            if (
                manifest["identity"] != identity
                or manifest["sha256"] != hashlib.sha256(raw).hexdigest()
            ):
                raise WeatherError("weather cache checksum/identity mismatch")
            frame = pd.read_parquet(entry / "native.parquet", engine="pyarrow")
            run = ForecastRun(
                identity["reference_time"],
                frame,
                version=manifest["version"],
                metadata=manifest["metadata"],
            )
            if run.metadata["identity"] != identity or run.version != digest(
                {"identity": identity, "sources": run.metadata["sources"]}
            ):
                raise WeatherError("weather cache source/version mismatch")
            return run
        except ImportError as error:
            raise ImportError("Install solarcast-belgium[weather] for Parquet cache") from error
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise WeatherError(f"invalid weather cache entry {entry}: {error}") from error

    def save(self, identity: dict[str, Any], run: ForecastRun) -> Path:
        if (
            run.metadata.get("identity") != identity
            or identity["reference_time"] != run.reference_time.isoformat()
            or run.observed_available_at is not None
            or run.revision
            or run.version != digest({"identity": identity, "sources": run.metadata.get("sources")})
        ):
            raise WeatherError(
                "cache supports original archive bundles, not availability assertions"
            )
        existing = self.load(identity)
        entry = self.root / digest(identity)
        if existing is not None:
            if existing.version != run.version:
                raise WeatherError("immutable cache already contains a different source version")
            return entry
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".weather-", dir=self.root) as temporary:
            stage = Path(temporary) / "entry"
            stage.mkdir()
            payload = stage / "native.parquet"
            try:
                run.data.to_parquet(payload, index=False, engine="pyarrow")
            except ImportError as error:
                raise ImportError("Install solarcast-belgium[weather] for Parquet cache") from error
            manifest = {
                "identity": identity,
                "version": run.version,
                "metadata": run.metadata,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "observed_available_at": None,
                "availability_note": (
                    "retrieval/Last-Modified are not historical first-publication evidence"
                ),
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8"
            )
            try:
                os.rename(stage, entry)
            except OSError:
                loaded = self.load(identity)
                if loaded is None or loaded.version != run.version:
                    raise WeatherError("conflicting concurrent weather cache writer") from None
        return entry
