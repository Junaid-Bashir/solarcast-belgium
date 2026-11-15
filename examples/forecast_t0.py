"""Real t0 inference: install .[t0]; cached weights by default.

Synthetic PV by default; --allow-download opts into pinned weight acquisition.
"""

from _tsfm_smoke import main

if __name__ == "__main__":
    main("t0")
