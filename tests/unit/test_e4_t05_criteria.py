"""E4-T05: AIC/BIC/ICL."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hsmmkit.engine.criteria import information_criteria, posterior_entropy
from hsmmkit.model.params import ParamManager, ParamSpec
from hsmmkit.utils.errors import NumericalError


def test_e4_t05_aic_bic_hand_checked() -> None:
    """AC4: p=3, T=4, loglik=-10 -> AIC=26, BIC=3 ln 4 + 20; one-hot posterior has zero entropy."""
    gamma = np.eye(2)[[0, 1, 1, 0]]
    ic = information_criteria(-10.0, 3, gamma)
    assert ic.aic == pytest.approx(26.0, abs=1e-12)
    assert ic.bic == pytest.approx(3 * math.log(4) + 20.0, abs=1e-12)
    assert ic.entropy == 0.0
    assert ic.icl == ic.bic
    assert (ic.n_params, ic.n_obs) == (3, 4)


def test_e4_t05_entropy_hand_checked_and_icl_adds_twice_entropy() -> None:
    """AC4: uniform 2-state rows have entropy T ln 2; ICL = BIC + 2 H."""
    gamma = np.full((5, 2), 0.5)
    assert posterior_entropy(gamma) == pytest.approx(5 * math.log(2), abs=1e-12)
    ic = information_criteria(-3.0, 2, gamma)
    assert ic.icl == pytest.approx(ic.bic + 10 * math.log(2), abs=1e-12)


def test_e4_t05_icl_ge_bic_for_random_posteriors() -> None:
    """AC4: entropy >= 0 so ICL >= BIC always (structural zeros in gamma are fine)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = rng.dirichlet(np.ones(4) * 0.3, size=30)
        g[:, 3] = 0.0
        g /= g.sum(axis=1, keepdims=True)
        ic = information_criteria(float(rng.normal(-50, 10)), int(rng.integers(1, 20)), g)
        assert ic.icl >= ic.bic


def test_e4_t05_masked_steps_are_excluded_from_t_and_entropy() -> None:
    """`T` counts unmasked steps only; masked rows do not add entropy."""
    gamma = np.full((4, 2), 0.5)
    mask = np.array([[True], [False], [True], [False]])
    ic = information_criteria(-1.0, 2, gamma, mask)
    assert ic.n_obs == 2
    assert ic.bic == pytest.approx(2 * math.log(2) + 2.0, abs=1e-12)
    assert ic.entropy == pytest.approx(2 * math.log(2), abs=1e-12)


def test_e4_t05_p_matches_appendix_b_n_free() -> None:
    """AC4: `p = ParamManager.n_free` (Appendix B: N=4, GBM + NB, order 1, full jump matrix -> 20)."""
    n = 4
    specs = (
        ParamSpec(
            name="transition.jump",
            shape=(n, n - 1),
            domain="simplex_ref",
            mode="free",
            init=np.full((n, n - 1), 1.0 / (n - 1)),
        ),
        ParamSpec(name="duration.rate", shape=(n,), domain="positive", mode="free", init=np.ones(n)),
        ParamSpec(name="emission.mean", shape=(n,), domain="real", mode="free", init=np.ones(n)),
        ParamSpec(name="emission.scale", shape=(n,), domain="positive", mode="free", init=np.ones(n)),
    )
    p = ParamManager(specs).n_free
    assert p == 20
    ic = information_criteria(-5.0, p, np.full((7, 3), 1.0 / 3.0))
    assert ic.aic == pytest.approx(2 * 20 + 10.0, abs=1e-12)
    assert ic.bic == pytest.approx(20 * math.log(7) + 10.0, abs=1e-12)


def test_e4_t05_non_finite_loglik_and_empty_mask_raise() -> None:
    """Invariant 5: bad inputs raise `NumericalError`."""
    with pytest.raises(NumericalError):
        information_criteria(float("nan"), 2, np.ones((3, 2)))
    with pytest.raises(NumericalError):
        information_criteria(-1.0, 2, np.ones((3, 2)), np.zeros((3, 1), dtype=bool))
