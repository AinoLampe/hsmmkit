"""Regenerate `docs/generated/estimator_parity.md` by running the parity suite (E13-T07).

Usage: `uv run python scripts/gen_parity_table.py [--slow]`. The exit code is the one of pytest, so the
nightly `parity` job fails when any row regresses. Without `--slow` the sampler rows (NUTS / Gibbs on the
two-regime fixture, ~15 minutes) are left out of the table: use `--slow` for the full table.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Run `tests/parity` with the table writer enabled; returns pytest's exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow", action="store_true", help="include the slow sampler rows")
    args = parser.parse_args(argv)
    os.environ["HSMMKIT_WRITE_PARITY"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("HSMMKIT_TYPECHECK", "1")
    marker = ["-m", "slow or not slow"] if args.slow else []
    return int(pytest.main([str(ROOT / "tests" / "parity"), "-q", "-p", "no:cacheprovider", *marker]))


if __name__ == "__main__":
    sys.exit(main())
