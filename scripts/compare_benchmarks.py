"""Compare a pytest-benchmark run against the recorded baseline.

Fails (non-zero exit) if any benchmark present in both the current run and the baseline is more
than 30% slower (by mean time) than its baseline. Benchmarks with no baseline entry (including the
case where the baseline is still the empty list from E0-T03, filled in by E4-T07) are reported but
never fail the comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REGRESSION_THRESHOLD = 0.30


def load_baseline(path: Path) -> dict[str, float]:
    """Return {benchmark name: baseline mean seconds} from the simplified baseline format."""
    entries = json.loads(path.read_text())
    return {entry["name"]: entry["mean"] for entry in entries}


def load_current(path: Path) -> dict[str, float]:
    """Return {benchmark name: current mean seconds} from a pytest-benchmark `--benchmark-json` file."""
    data = json.loads(path.read_text())
    return {b["name"]: b["stats"]["mean"] for b in data.get("benchmarks", [])}


def compare(baseline: dict[str, float], current: dict[str, float]) -> tuple[bool, list[str]]:
    """Return (ok, report_lines); ok is False if any shared benchmark regressed > 30%."""
    ok = True
    lines = []
    for name, current_mean in current.items():
        baseline_mean = baseline.get(name)
        if baseline_mean is None:
            lines.append(f"{name}: no baseline, skipped")
            continue
        regression = (current_mean - baseline_mean) / baseline_mean
        status = "FAIL" if regression > REGRESSION_THRESHOLD else "ok"
        if status == "FAIL":
            ok = False
        lines.append(
            f"{name}: baseline={baseline_mean:.6g}s current={current_mean:.6g}s "
            f"regression={regression * 100:.1f}% {status}"
        )
    return ok, lines


def main(argv: list[str]) -> int:
    """Compare `argv[0]` (current bench.json) against `argv[1]` (baseline.json)."""
    if len(argv) != 2:
        print("usage: compare_benchmarks.py <bench.json> <baseline.json>")
        return 2
    current_path, baseline_path = Path(argv[0]), Path(argv[1])
    baseline = load_baseline(baseline_path)
    current = load_current(current_path)
    ok, lines = compare(baseline, current)
    print("\n".join(lines))
    if not ok:
        print(f"error: at least one benchmark regressed more than {REGRESSION_THRESHOLD * 100:.0f}%.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
