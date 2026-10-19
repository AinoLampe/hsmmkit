"""Build `benchmarks/baseline.json` from several `pytest-benchmark` JSON files (CI runs).

Usage:
    uv run python scripts/make_benchmark_baseline.py bench1.json bench2.json bench3.json

The baseline entry of a benchmark is the mean over the files of its mean time. Use files produced by the
CI runner (the nightly `benchmark` job uploads `bench.json`): `compare_benchmarks.py` gates a >30 %
slowdown, which only means something when baseline and current run share hardware.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "benchmarks" / "baseline.json"


def build(paths: list[Path]) -> list[dict[str, object]]:
    """`[{name, mean}]` sorted by name, averaged over `paths`; a benchmark must appear in every file."""
    per_file = [
        {b["name"]: b["stats"]["mean"] for b in json.loads(p.read_text())["benchmarks"]} for p in paths
    ]
    names = sorted(set.intersection(*(set(m) for m in per_file)))
    return [{"name": n, "mean": round(statistics.fmean(m[n] for m in per_file), 6)} for n in names]


def main(argv: list[str]) -> int:
    """Write the baseline from the given files."""
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    TARGET.write_text(json.dumps(build([Path(a) for a in argv]), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
