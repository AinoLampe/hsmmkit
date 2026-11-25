"""E5-T05a: the parameter-recovery suite (`-m slow`; the weekly CI job).

Six scenarios, 5 seeds each, `T=20000` by default (`HSMMKIT_RECOVERY_T` env var overrides; `order2` runs at
2.5x, CLAUDE.md §9 / ticket AC3). Each scenario must pass every gating tolerance category in at least 4/5
seeds against concept Appendix D's tolerances (`hsmmkit.estimation.recovery`, never loosened per
CLAUDE.md §6); `explicit_gamma`'s mean-sojourn category is report-only. AC2: a failing assertion prints every
failing seed's truth/estimate/SE lines (`SuiteResult.failures()`). Each scenario's full outcome (including
report-only categories) is written to `runs/recovery/<scenario>.json` for the CI artifact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hsmmkit.estimation.recovery import SCENARIOS, run_recovery_suite

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_e5_t05a_recovery(name: str) -> None:
    """AC1: each scenario passes its gating categories in >= 4/5 seeds. AC2: failures list the checks."""
    suite = run_recovery_suite(name)
    out_dir = Path(os.environ.get("HSMMKIT_RECOVERY_REPORT_DIR", "runs/recovery"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(suite.summary(), indent=2), encoding="utf-8")
    assert suite.passed, f"{name}: category outcomes {suite.categories}\n{suite.failures()}"
