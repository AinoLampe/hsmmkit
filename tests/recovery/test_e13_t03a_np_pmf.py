"""E13-T03a AC2: nonparametric pmf recovery by GEM at T = 20 000 (total variation <= 0.05 per state)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation.gem import GEMEstimator
from hsmmkit.model.params.manager import ParamManager
from tests.helpers import gem_models

D_MAX = 60


def _truth() -> np.ndarray:
    """Compact supports (<= 10 bins each): with >1000 segments per regime the sampling TV stays below 0.05."""
    pmf = np.zeros((3, D_MAX))
    pmf[0, 1:6] = [0.1, 0.3, 0.3, 0.2, 0.1]  # d = 2..6
    pmf[1, 5:10] = [0.1, 0.2, 0.4, 0.2, 0.1]  # d = 6..10
    pmf[2, :8] = 0.7 ** np.arange(8)  # d = 1..8, decaying
    return pmf / pmf.sum(axis=1, keepdims=True)


@pytest.mark.slow
def test_e13_t03a_nonparametric_pmf_recovery(tmp_path: Path) -> None:
    """Three regimes with distinct pmfs, well separated emissions: TV <= 0.05 per regime (T = 20 000)."""
    pmf = _truth()
    pi = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    y, z = gem_models.simulate(20_000, pmf, pi, np.array([-3.0, 0.0, 3.0]), np.array([0.8, 0.8, 0.8]), 7)
    data = gem_models.dataset(y)
    spec = gem_models.build(
        tmp_path, 3, D_MAX, mean_sojourn=10.0, means=[-2.5, 0.2, 2.5], scales=[1.0, 1.0, 1.0]
    )
    cfg = EstimationSection(method="gem", restarts=0, max_iter=400, tol_f=1e-10, use_priors=False)
    res = GEMEstimator().fit(spec, data, cfg, 0, warm_start=ParamManager(spec.param_specs).init_u())
    est = np.asarray(res.params["duration.logits"])
    order = np.argsort(np.asarray(res.params["emission.mean"]).ravel())
    tv = 0.5 * np.abs(est[order] - pmf).sum(axis=1)
    assert np.all(tv <= 0.05), tv
    # the sampling error of the true segments themselves (the oracle) is far below the tolerance
    starts = np.flatnonzero(np.r_[True, z[1:] != z[:-1]])
    lengths = np.diff(np.r_[starts, z.size])
    for j in range(3):
        emp = np.bincount(lengths[z[starts] == j], minlength=D_MAX + 1)[1 : D_MAX + 1].astype(float)
        assert 0.5 * np.abs(emp / emp.sum() - pmf[j]).sum() <= 0.05
