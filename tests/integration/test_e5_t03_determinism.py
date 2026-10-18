"""E5-T03 AC4 (concept §6.4.6): `fit` is bitwise deterministic given the same config/data/seed."""

from __future__ import annotations

import warnings

import numpy as np

from regimekit.config.schema import EstimationSection
from regimekit.estimation import fit
from tests.helpers.objective_fixtures import make_case


def test_e5_t03_fit_twice_gives_bitwise_identical_uparams() -> None:
    """Running `fit` twice with the same config/data/seed on CPU gives bitwise identical `uparams`."""
    case = make_case("gbm", "negbin", 1, t_len=150, seed=9)
    cfg = EstimationSection(  # type: ignore[call-arg]
        optimizer="lbfgs", max_iter=30, restarts=3, init=["random", "rollvol_kmeans"]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r1 = fit(case.spec, case.data, cfg, seed=42)
        r2 = fit(case.spec, case.data, cfg, seed=42)

    assert np.array_equal(np.asarray(r1.uparams), np.asarray(r2.uparams))
    assert r1.objective == r2.objective
    assert r1.loglik == r2.loglik
    assert r1.restarts["objective"].tolist() == r2.restarts["objective"].tolist()
