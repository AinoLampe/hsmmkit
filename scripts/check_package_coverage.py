"""Fail if any tracked regimekit package's line coverage is below its required threshold.

Reads `coverage.json` (produced by `pytest --cov --cov-report=json`) and aggregates covered /
statement counts per top-level package under `src/regimekit`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_PACKAGE_COVERAGE = 85.0
TRACKED_PACKAGES = ("model", "engine", "estimation", "inference", "data", "evaluation")
COVERAGE_JSON = Path("coverage.json")


def package_of(file_path: str) -> str | None:
    """Return the top-level regimekit package name for a coverage.json file path, if any."""
    normalized = file_path.replace("\\", "/")
    marker = "src/regimekit/"
    idx = normalized.find(marker)
    if idx == -1:
        return None
    rest = normalized[idx + len(marker) :]
    parts = rest.split("/")
    return parts[0] if parts else None


def main() -> int:
    """Check per-package coverage and print a summary; exit non-zero on any failure."""
    if not COVERAGE_JSON.exists():
        print(f"error: {COVERAGE_JSON} not found; run `pytest --cov --cov-report=json` first.")
        return 1

    data = json.loads(COVERAGE_JSON.read_text())
    totals: dict[str, dict[str, int]] = {}
    for file_path, file_data in data.get("files", {}).items():
        pkg = package_of(file_path)
        if pkg is None:
            continue
        summary = file_data["summary"]
        bucket = totals.setdefault(pkg, {"covered": 0, "statements": 0})
        bucket["covered"] += summary["covered_lines"]
        bucket["statements"] += summary["num_statements"]

    failed = False
    for pkg in TRACKED_PACKAGES:
        bucket = totals.get(pkg, {"covered": 0, "statements": 0})
        statements = bucket["statements"]
        percent = 100.0 if statements == 0 else 100.0 * bucket["covered"] / statements
        status = "ok"
        if percent < REQUIRED_PACKAGE_COVERAGE:
            failed = True
            status = "FAIL"
        print(f"{pkg:<12} {percent:6.2f}%  (required >= {REQUIRED_PACKAGE_COVERAGE:.0f}%)  {status}")

    if failed:
        print("error: per-package coverage gate failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
