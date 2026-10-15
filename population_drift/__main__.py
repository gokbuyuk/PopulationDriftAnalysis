"""Command line: ``python -m population_drift``."""

from __future__ import annotations

import sys

from population_drift.analysis import main

if __name__ == "__main__":
    sys.exit(main())
