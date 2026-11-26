"""E10-T11: `import hsmmkit` pulls no extra package, and its import time stays near the recorded baseline."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks" / "import_time_baseline.json"
FORBIDDEN = {
    "matplotlib",
    "sklearn",
    "numpyro",
    "arviz",
    "xarray",
    "scipy",
}  # scipy: ADR-0020 vendors the Hungarian algorithm
ROW = re.compile(r"^import time:\s+(\d+) \|\s+(\d+) \| (\s*)(\S+)$")


def _import_hsmmkit() -> tuple[set[str], int]:
    """Top-level module names imported by `import hsmmkit`, and its cumulative import time in us."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("HSMMKIT_TYPECHECK", None)
    out = subprocess.run(
        [sys.executable, "-I", "-X", "importtime", "-c", "import hsmmkit"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stderr
    tops: set[str] = set()
    total = 0
    for line in out.splitlines():
        m = ROW.match(line)
        if m:
            tops.add(m.group(4).split(".")[0])
            if m.group(4) == "hsmmkit":
                total = int(m.group(2))
    return tops, total


def test_e10_t11_import_pulls_no_extra_package() -> None:
    """AC1: none of the extras' packages (nor scipy) is imported by `import hsmmkit`."""
    tops, total = _import_hsmmkit()
    assert total > 0
    assert not tops & FORBIDDEN, sorted(tops & FORBIDDEN)


@pytest.mark.skipif(
    "PYTEST_XDIST_WORKER" in os.environ,
    reason="timing is meaningless while other workers compete for the CPU",
)
def test_e10_t11_import_time_within_baseline() -> None:
    """The best of three imports stays below the committed baseline x 1.3 (refresh needs a justification)."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    best = min(_import_hsmmkit()[1] for _ in range(3))
    assert best <= baseline["cumulative_us"] * baseline["factor"], (best, baseline)
