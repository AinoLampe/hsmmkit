"""E5-T05a: the quick nightly smoke run at the old size, `T=5000` (`-m slow`, one scenario).

At `T=5000` the ticket's truth values give only ~64 regime visits per path, so even an oracle that reads
the true regime path misses the sojourn / jump tolerances in about half of the seeds: those tolerances are
not asserted here (the weekly full suite does, at `T=20000`). The smoke run checks that the pipeline works
end to end on a realistic size -- simulate, multi-start MAP fit, checks -- and that the one category that
is attainable at this size, `sigma` (<= 10 % relative), holds in at least 4 of 5 seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimekit.estimation.recovery import run_recovery_suite

pytestmark = pytest.mark.slow


def test_e5_t05a_recovery_smoke_t5000_pipeline_and_sigma() -> None:
    """Finite fits for all 5 seeds and the `sigma` category in >= 4/5 seeds, on `gbm_negbin` at T=5000."""
    suite = run_recovery_suite("gbm_negbin", t_len=5000)
    assert len(suite.reports) == 5
    assert all(np.isfinite(r.fit.loglik) for r in suite.reports)
    assert all(r.t_len == 5000 for r in suite.reports)
    ok, total = suite.categories["sigma"]
    assert ok >= min(4, total), f"sigma: {ok}/{total} seeds\n{suite.failures()}"
