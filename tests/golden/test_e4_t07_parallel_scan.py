"""E4-T07 golden: `associative_scan` smoothing equals the sequential scan; it is opt-in."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from regimekit.engine import parallel
from regimekit.engine.base import Posterior
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from tests.helpers.engine_fixtures import make_scenario


def assert_same_posterior(par: Posterior, seq: Posterior) -> None:
    """Loglik rel <= 1e-10; every table max-abs <= 1e-10 on the same `-inf` pattern."""
    ll_p, ll_s = float(par.loglik), float(seq.loglik)
    assert abs(ll_p - ll_s) <= 1e-10 * abs(ll_s)
    for name in ("log_alpha", "log_beta"):
        a, b = np.asarray(getattr(par, name)), np.asarray(getattr(seq, name))
        finite = np.isfinite(b)
        assert np.array_equal(np.isfinite(a), finite), name
        np.testing.assert_allclose(a[finite], b[finite], rtol=0, atol=1e-10, err_msg=name)
    np.testing.assert_allclose(
        np.asarray(par.smoothed),
        np.asarray(seq.smoothed),
        rtol=0,
        atol=1e-10,
    )


@given(
    n_states=st.integers(2, 4),
    order=st.integers(1, 2),
    t_len=st.integers(5, 200),
    seed=st.integers(0, 10_000),
    data=st.data(),
)
def test_e4_t07_parallel_scan_equals_sequential(
    n_states: int, order: int, t_len: int, seed: int, data: st.DataObject
) -> None:
    """AC3: parallel == sequential to 1e-10 on random N/T/r_i/order/covariates and irregular `dt`."""
    r = data.draw(st.lists(st.integers(1, 3), min_size=n_states, max_size=n_states))
    n_cov = data.draw(st.integers(0, 2))
    sc = make_scenario(
        n_states, r, order, t_len, n_cov_pi=n_cov, n_cov_d=n_cov, seed=seed, irregular=bool(seed % 2)
    )
    args = (sc.theta, sc.spec, sc.space, sc.data)
    seq = ExpandedHMMEngine().forward_backward(*args)
    par = ExpandedHMMEngine(parallel_scan=True).forward_backward(*args)
    assert_same_posterior(par, seq)


def test_e4_t07_parallel_scan_respects_masked_steps_and_free_initial() -> None:
    """Masked observations and `initial="free"` go through the same emission/initial code."""
    sc = make_scenario(3, [2, 1, 3], 1, 80, initial="free", seed=6)
    mask = np.ones((80, 1), dtype=bool)
    mask[20:27] = False
    fields = {f: getattr(sc.data, f) for f in sc.data.__dataclass_fields__}
    data = sc.data.__class__(**{**fields, "mask": mask})
    args = (sc.theta, sc.spec, sc.space, data)
    assert_same_posterior(
        ExpandedHMMEngine(parallel_scan=True).forward_backward(*args),
        ExpandedHMMEngine().forward_backward(*args),
    )


def test_e4_t07_parallel_scan_is_off_by_default_and_used_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: the parallel path runs only when the flag is set."""
    sc = make_scenario(3, [2, 2, 2], 1, 30, seed=1)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    calls: list[int] = []
    real = parallel.parallel_forward_backward

    def spy(*a: object, **k: object) -> object:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(parallel, "parallel_forward_backward", spy)
    assert ExpandedHMMEngine().parallel_scan is False
    ExpandedHMMEngine().forward_backward(*args)
    ExpandedHMMEngine().forward(*args)
    ExpandedHMMEngine().loglik(*args)
    assert calls == []
    ExpandedHMMEngine(parallel_scan=True).forward_backward(*args)
    assert calls == [1]


def test_e4_t07_parallel_scan_falls_back_above_the_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """`T * S^2 > 5e7` warns and uses the sequential scan (same result)."""
    sc = make_scenario(3, [2, 2, 2], 1, 30, seed=1)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    seq = ExpandedHMMEngine().forward_backward(*args)
    monkeypatch.setattr(parallel, "MAX_CELLS", 10.0)
    with pytest.warns(UserWarning, match="sequential scan"):
        par = ExpandedHMMEngine(parallel_scan=True).forward_backward(*args)
    assert_same_posterior(par, seq)


def test_e4_t07_parallel_scan_long_sharp_sequence_agrees_on_everything_that_matters() -> None:
    """T=2000 with well-separated scales and outliers (entries hundreds of nats apart).

    Loglik, smoothed and filtered probabilities still agree to 1e-10 (the joint log tables to rel
    1e-10). A `-inf` in the parallel `log_alpha`
    or `log_beta` where the sequential scan is finite is allowed only for entries more than 700 nats
    below the largest entry of their row (their probability is < 1e-300): the documented underflow.
    """
    import jax.numpy as jnp

    from tests.helpers.engine_fixtures import DT, make_dataset

    sc = make_scenario(3, [2, 3, 2], 1, 2000, seed=0)
    sigma = np.array([0.05, 0.2, 0.8])
    theta = {**sc.theta, "emission.sigma": jnp.asarray(sigma)}
    rng = np.random.default_rng(0)
    regime = np.repeat(rng.integers(0, 3, size=41), 50)[:2000]
    y = rng.normal(0.0, sigma[regime] * np.sqrt(DT))
    y[rng.integers(0, 2000, size=20)] *= 6.0
    data = make_dataset(y[:, None], np.full(2000, DT))
    args = (theta, sc.spec, sc.space, data)
    seq = ExpandedHMMEngine().forward_backward(*args)
    par = ExpandedHMMEngine(parallel_scan=True).forward_backward(*args)

    assert abs(float(par.loglik) - float(seq.loglik)) <= 1e-10 * abs(float(seq.loglik))
    np.testing.assert_allclose(np.asarray(par.smoothed), np.asarray(seq.smoothed), rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(par.regime_filtered()), np.asarray(seq.regime_filtered()), rtol=0, atol=1e-10
    )
    for name in ("log_alpha", "log_beta"):
        a, b = np.asarray(getattr(par, name)), np.asarray(getattr(seq, name))
        both = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(a[both], b[both], rtol=1e-10, atol=1e-9, err_msg=name)
        lost = np.isfinite(b) & ~np.isfinite(a)
        assert lost.any() or name == "log_beta"  # the scenario really exercises the underflow branch
        row_max = np.max(np.where(np.isfinite(b), b, -np.inf), axis=1, keepdims=True)
        assert np.all((b - row_max)[lost] < -700.0), name
