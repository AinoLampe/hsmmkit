"""E13-T02a AC1: the EM log-likelihood never decreases (3 synthetic datasets per family)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.estimation.em import run_em
from hsmmkit.statespace.builders import build_expanded
from tests.helpers import em_models

FAMILIES = ["gaussian", "gaussian_full", "poisson", "categorical"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e13_t02a_loglik_is_non_decreasing(tmp_path: Path, family: str, seed: int) -> None:
    """Violations stay below 1e-10 (absolute, plus 1e-13 relative) over 60 iterations, geometric durations."""
    spec, theta0 = em_models.build(tmp_path, family)
    data = em_models.dataset(family, 500, seed)
    pb = SequenceBatch.of(data).padded()
    cfg = EstimationSection(max_iter=60, tol_f=1e-14, use_priors=False)
    _, trace, _, n_iter = run_em(
        theta0, spec, build_expanded(spec), pb, cfg, debug=True
    )  # debug raises on a decrease
    ll = np.asarray(trace)
    assert n_iter >= 5
    assert ll[-1] > ll[0]
    assert np.all(np.diff(ll) >= -(1e-10 + 1e-13 * np.abs(ll[:-1])))


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_e13_t02a_negbin_durations_are_monotone_too(tmp_path: Path, family: str) -> None:
    """Negative-binomial durations (phase pooling per regime)."""
    spec, theta0 = em_models.build(tmp_path, family, duration="negbin")
    data = em_models.dataset(family, 500, 3)
    pb = SequenceBatch.of(data).padded()
    cfg = EstimationSection(max_iter=60, tol_f=1e-14, use_priors=False)
    _, trace, _, _ = run_em(theta0, spec, build_expanded(spec), pb, cfg, debug=True)
    ll = np.asarray(trace)
    assert np.all(np.diff(ll) >= -(1e-10 + 1e-13 * np.abs(ll[:-1])))
