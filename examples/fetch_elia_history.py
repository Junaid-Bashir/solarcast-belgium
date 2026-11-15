"""Fetch one hour only when explicitly run: python examples/fetch_elia_history.py."""

import logging

from solarcast.data import EliaPVClient


def main() -> None:
    """Inspect a small historical period; do not write or cache data."""
    logging.basicConfig(level=logging.INFO)
    with EliaPVClient() as client:
        pv = client.fetch_history(start="2026-11-12T12:00:00Z", end="2026-11-12T13:00:00Z")
    print(pv.to_string(index=False))
    print(pv.dtypes)


if __name__ == "__main__":
    main()
