"""E11-T05: MS-AR(p): conditional likelihood, own-lag covariates, Hamilton's mean-adjusted time-step space."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.contracts import ObservationSet, lag_names, own_lag_columns
from hsmmkit.data.prepare import Preparer
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.model.build import build_model_spec, required_own_lags
from hsmmkit.model.emission.msar import MSARFamily
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.statespace.builders import build_expanded
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw


def _raw(tmp_path: Path, emission: dict[str, Any], n_states: int = 2) -> dict[str, Any]:
    raw = core_raw(tmp_path)
    raw["model"]["n_states"] = n_states
    raw["model"]["emission"] = emission
    raw["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [20.0] * n_states
    return raw


def _emission(form: str, p: int, n_states: int = 2) -> dict[str, Any]:
    loc = "intercept" if form == "coefficients" else "mean"
    return {
        "family": "msar",
        "ar": {"p": p, "form": form},
        "params": {
            loc: {"mode": "free", "init": [0.1 * i for i in range(n_states)]},
            "scale": {"mode": "free", "init": [0.5 + 0.5 * i for i in range(n_states)]},
        },
    }


def _observations(y: np.ndarray) -> ObservationSet:
    times = pd.bdate_range("2012-01-02", periods=y.shape[0]).to_numpy().astype("datetime64[ns]")
    return ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "synthetic")


def _prepared(cfg: Any, y: np.ndarray) -> Any:  # noqa: ANN401
    obs = _observations(y)
    prep = Preparer(cfg.data, own_lags=required_own_lags(cfg)).fit(obs, [], train_end=obs.require_times()[-1])
    return prep.transform(obs, [])


def _ar_series(rng: np.random.Generator, n: int, c: float, phi: tuple[float, ...], sd: float) -> np.ndarray:
    y = np.zeros(n)
    for t in range(n):
        y[t] = c + sum(f * y[t - 1 - i] for i, f in enumerate(phi) if t - 1 - i >= 0) + sd * rng.normal()
    return y


# --- AC1: one-state AR(p) against statsmodels' conditional likelihood ------------------------------


@pytest.mark.parametrize("p", [1, 2, 3])
def test_e11_t05_one_state_ar_matches_statsmodels_conditional_llf(p: int) -> None:
    """AC1: the conditional log-likelihood of a one-regime AR(p) equals `AutoReg(...).fit().llf` (<= 1e-6)."""
    from statsmodels.tsa.ar_model import AutoReg

    rng = np.random.default_rng(p)
    y = _ar_series(rng, 400, 0.3, (0.5, -0.2, 0.1)[:p], 0.8)
    res = AutoReg(y, lags=p, trend="c").fit()
    fam = MSARFamily(p=p, n_series=1)
    lags, ok = own_lag_columns(y.reshape(-1, 1), np.ones((y.shape[0], 1), bool), p)
    theta = {
        "emission.intercept": jnp.asarray([[res.params[0]]]),
        "emission.ar": jnp.asarray(res.params[1:]).reshape(1, p, 1, 1),
        "emission.scale": jnp.asarray([[np.sqrt(res.sigma2)]]),
    }
    lp = np.asarray(
        fam.log_prob(jnp.asarray(y.reshape(-1, 1)), jnp.ones(y.shape[0]), theta, jnp.asarray(lags))
    )
    assert abs(lp[ok, 0].sum() - res.llf) <= 1e-6
    # the weighted least squares M-step with unit weights is the least-squares / ML estimate
    gamma = jnp.asarray(ok.astype(float)).reshape(-1, 1)
    stats_ = fam.sufficient_stats(
        jnp.asarray(y.reshape(-1, 1)), jnp.ones(y.shape[0]), gamma, jnp.asarray(lags)
    )
    hat = fam.m_step(stats_)
    assert hat is not None
    np.testing.assert_allclose(np.asarray(hat["emission.intercept"])[0, 0], res.params[0], rtol=1e-8)
    np.testing.assert_allclose(np.asarray(hat["emission.ar"]).ravel(), res.params[1:], rtol=1e-8)
    np.testing.assert_allclose(np.asarray(hat["emission.scale"])[0, 0] ** 2, res.sigma2, rtol=1e-8)


def test_e11_t05_m_step_is_weighted_least_squares_for_a_bivariate_series() -> None:
    """The WLS M-step matches a direct weighted `lstsq` per regime for `M = 2`, `p = 2` (full covariance)."""
    rng = np.random.default_rng(3)
    t, m, p = 600, 2, 2
    y = rng.normal(size=(t, m)).cumsum(axis=0) * 0.1 + rng.normal(size=(t, m))
    lags, ok = own_lag_columns(y, np.ones((t, m), bool), p)
    gamma = rng.dirichlet(np.ones(2), size=t) * ok[:, None]
    fam = MSARFamily(p=p, n_series=m, covariance="full")
    hat = fam.m_step(fam.sufficient_stats(jnp.asarray(y), jnp.ones(t), jnp.asarray(gamma), jnp.asarray(lags)))
    assert hat is not None
    x = np.concatenate([np.ones((t, 1)), lags], axis=1)
    for i in range(2):
        w = np.sqrt(gamma[:, i])[:, None]
        b, *_ = np.linalg.lstsq(x * w, y * w, rcond=None)
        np.testing.assert_allclose(np.asarray(hat["emission.intercept"])[i], b[0], rtol=1e-8, atol=1e-10)
        phi = np.asarray(hat["emission.ar"])[i]  # (p, M, M)
        for lag in range(p):
            np.testing.assert_allclose(phi[lag], b[1 + lag * m : 1 + (lag + 1) * m].T, rtol=1e-8, atol=1e-10)
        resid = y - x @ b
        sigma = (resid * gamma[:, [i]]).T @ resid / gamma[:, i].sum()
        factor = np.zeros((m, m))
        factor[np.tril_indices(m)] = np.asarray(hat["emission.chol"])[i]
        np.testing.assert_allclose(factor @ factor.T, sigma, rtol=1e-7, atol=1e-9)


def test_e11_t05_conjugate_prior_posterior_is_the_wls_estimate_for_a_flat_prior() -> None:
    """With `kappa0 -> 0` the matrix-Normal-Wishart posterior mean of `B` is the least-squares estimate."""
    rng = np.random.default_rng(4)
    t, p = 300, 1
    y = _ar_series(rng, t, 0.2, (0.6,), 1.0).reshape(-1, 1)
    lags, ok = own_lag_columns(y, np.ones((t, 1), bool), p)
    fam = MSARFamily(p=p, n_series=1, covariance="full")
    st_ = fam.sufficient_stats(
        jnp.asarray(y), jnp.ones(t), jnp.asarray(ok.astype(float)).reshape(-1, 1), jnp.asarray(lags)
    )
    prior = fam.conjugate_prior()
    assert prior is not None
    post = type(prior)(kappa0=1e-12).update(st_)
    hat = fam.m_step(st_)
    assert hat is not None
    np.testing.assert_allclose(
        np.asarray(post["B"])[0, 0, 0], np.asarray(hat["emission.intercept"])[0, 0], rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(post["B"])[0, 1, 0], np.asarray(hat["emission.ar"]).ravel()[0], rtol=1e-6
    )


# --- AC4: own lags never look ahead ------------------------------------------------------------------


def test_e11_t05_preparer_appends_masked_lag_columns(tmp_path: Path) -> None:
    """The preparer adds `__lag{l}_x` columns, masks the first `p` steps and records the names."""
    cfg = parse_project_config(_raw(tmp_path, _emission("coefficients", 2)))
    y = np.arange(1.0, 21.0)
    data = _prepared(cfg, y)
    assert data.cov_names["emission"] == lag_names(("x",), 2) == ("__lag1_x", "__lag2_x")
    assert not bool(jnp.any(data.mask[:2]))
    assert bool(jnp.all(data.mask[2:]))
    z = np.asarray(data.cov_emission)
    np.testing.assert_array_equal(z[2:, 0], y[1:-1])  # y_{t-1}
    np.testing.assert_array_equal(z[2:, 1], y[:-2])  # y_{t-2}


@given(
    st.integers(min_value=1, max_value=3),
    st.integers(min_value=4, max_value=40),
    st.data(),
)
def test_e11_t05_spike_at_t_changes_nothing_before_t_plus_1(p: int, n: int, data: st.DataObject) -> None:
    """AC4: a spike injected at `t` leaves the lag block and mask of rows `<= t` bit-identical."""
    t = data.draw(st.integers(min_value=0, max_value=n - 1))
    rng = np.random.default_rng(n * 7 + p)
    y = rng.normal(size=(n, 2))
    mask = rng.random((n, 2)) > 0.1
    spiked = y.copy()
    spiked[t] += 1e6
    a, ok_a = own_lag_columns(y, mask, p)
    b, ok_b = own_lag_columns(spiked, mask, p)
    np.testing.assert_array_equal(a[: t + 1], b[: t + 1])
    np.testing.assert_array_equal(ok_a[: t + 1], ok_b[: t + 1])
    assert not np.any(ok_a[:p])  # the first p steps are always masked


def test_e11_t05_rows_after_a_missing_value_are_masked_until_the_lags_are_back() -> None:
    """A step whose lag is unobserved is masked (the lag value is not invented)."""
    mask = np.ones((10, 1), bool)
    mask[4] = False
    _, ok = own_lag_columns(np.arange(10.0).reshape(-1, 1), mask, 2)
    assert ok.tolist() == [False, False, True, True, True, False, False, True, True, True]


# --- AC3: Hamilton's mean-adjusted form on the time-step space ---------------------------------------


def test_e11_t05_mean_adjusted_with_p3_is_a_config_error(tmp_path: Path) -> None:
    """AC3: `p = 3` mean-adjusted is rejected (family, config build)."""
    with pytest.raises(ConfigError, match="p <= 2"):
        MSARFamily(p=3, form="mean_adjusted")
    cfg = parse_project_config(_raw(tmp_path, _emission("mean_adjusted", 3)))
    with pytest.raises(ConfigError, match="p <= 2"):
        _ = build_model_spec(
            cfg, _prepared(parse_project_config(_raw(tmp_path, _emission("coefficients", 3))), np.zeros(30))
        )


def test_e11_t05_mean_adjusted_requires_geometric_durations_and_the_expanded_engine(tmp_path: Path) -> None:
    """`memory_mode: steps` needs geometric durations (and the expanded engine)."""
    base = parse_project_config(_raw(tmp_path, _emission("coefficients", 1)))
    data = _prepared(base, np.arange(30.0))
    raw = _raw(tmp_path, _emission("mean_adjusted", 1))
    raw["model"]["duration"] = {
        "family": "negbin",
        "r": [2, 2],
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [10.0, 10.0]}},
    }
    with pytest.raises(ConfigError) as err:
        build_model_spec(parse_project_config(raw), data)
    assert err.value.path == "model.duration.family"
    raw = _raw(tmp_path, _emission("mean_adjusted", 1))
    raw["model"]["engine"] = "explicit"
    with pytest.raises(ConfigError) as err2:
        build_model_spec(parse_project_config(raw), data)
    assert err2.value.path == "model.engine"


def _brute_force_loglik(
    y: np.ndarray, p: int, mu: np.ndarray, phi: np.ndarray, sd: np.ndarray, adv: np.ndarray, pi: np.ndarray
) -> float:
    """`log p(y_{p+1:T})` by enumerating every regime path `(s_{-p}, ..., s_{T-1})` (uniform start tuple)."""
    n, t = mu.shape[0], y.shape[0]
    total = -np.inf
    for path in itertools.product(range(n), repeat=t + p):  # path[k] = s_{k-p}
        logp = -(p + 1) * np.log(n) + 0.0  # uniform over the first tuple (s_{-p}, ..., s_0)
        for k in range(p, t + p - 1):  # transition s_{k-p} -> s_{k+1-p}
            a, b = path[k], path[k + 1]
            logp += np.log(1 - adv[a]) if a == b else np.log(adv[a] * pi[a, b])
        for step in range(p, t):  # density of y_step given regimes at step-p..step
            s = [path[step + p - lag] for lag in range(p + 1)]  # s_t, s_{t-1}, ...
            mean = mu[s[0]] + sum(phi[lag - 1] * (y[step - lag] - mu[s[lag]]) for lag in range(1, p + 1))
            logp += stats.norm.logpdf(y[step], mean, sd[s[0]])
        total = np.logaddexp(total, logp)
    return float(total)


@pytest.mark.parametrize("p", [1, 2])
def test_e11_t05_mean_adjusted_matches_brute_force_enumeration(p: int, tmp_path: Path) -> None:
    """AC3: `N^(p+1)` states; the likelihood equals the enumeration over all regime paths (<= 1e-10)."""
    rng = np.random.default_rng(10 + p)
    t = 8
    y = rng.normal(size=t)
    cfg = parse_project_config(_raw(tmp_path, _emission("mean_adjusted", p)))
    data = _prepared(cfg, y)
    spec = build_model_spec(cfg, data)
    space = build_expanded(spec)
    assert spec.memory_mode == "steps"
    assert space.n_expanded == 2 ** (p + 1)
    mgr = ParamManager(spec.param_specs)
    theta = dict(mgr.unflatten(mgr.init_u()))
    theta["emission.mean"] = jnp.asarray([[-0.4], [0.9]])
    theta["emission.ar"] = jnp.asarray([0.5, -0.3][:p]).reshape(p, 1, 1)
    theta["emission.scale"] = jnp.asarray([[0.7], [1.3]])
    ll = float(ExpandedHMMEngine().loglik(theta, spec, space, data))
    adv = np.asarray(spec.duration.advance_prob_t(theta, jnp.asarray(1.0), None))
    brute = _brute_force_loglik(
        y.astype(float),
        p,
        np.asarray(theta["emission.mean"])[:, 0],
        np.asarray(theta["emission.ar"])[:, 0, 0],
        np.asarray(theta["emission.scale"])[:, 0],
        adv,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert abs(ll - brute) <= 1e-10 * max(1.0, abs(brute))


def test_e11_t05_steps_state_space_layout(tmp_path: Path) -> None:
    """Tuples `(s_{t-p}, ..., s_t)` with repeats: `N^(p+1)` states, rows of `A_t` sum to 1."""
    cfg = parse_project_config(_raw(tmp_path, _emission("mean_adjusted", 2, n_states=3), n_states=3))
    data = _prepared(cfg, np.arange(20.0))
    spec = build_model_spec(cfg, data)
    space = build_expanded(spec)
    assert space.n_expanded == 27
    idx = space.index
    assert idx.memory_of.shape == (27, 2)
    assert bool(jnp.all(idx.phase_of == 1))
    assert len({tuple(r) for r in np.c_[np.asarray(idx.memory_of), np.asarray(idx.regime_of)]}) == 27
    mgr = ParamManager(spec.param_specs)
    a, _ = space.transition_fn(mgr.unflatten(mgr.init_u()), None, None, jnp.asarray(1.0))
    np.testing.assert_allclose(np.asarray(a).sum(axis=1), 1.0, atol=1e-12)
    assert int((np.asarray(a) > 0).sum(axis=1).max()) == 3  # N successors per tuple


def test_e11_t05_mean_adjusted_has_no_per_regime_density(tmp_path: Path) -> None:
    """The per-regime `log_prob` / `cdf` do not exist for the mean-adjusted form (regime history)."""
    fam = MSARFamily(p=1, form="mean_adjusted")
    theta = {
        "emission.mean": jnp.zeros((2, 1)),
        "emission.ar": jnp.zeros((1, 1, 1)),
        "emission.scale": jnp.ones((2, 1)),
    }
    with pytest.raises(ConfigError, match="previous steps"):
        fam.log_prob(jnp.zeros((3, 1)), jnp.ones(3), theta, jnp.zeros((3, 1)))
    with pytest.raises(ConfigError, match="previous steps"):
        fam.cdf(jnp.zeros((3, 1)), jnp.ones(3), theta, jnp.zeros((3, 1)))


# --- config and diagnostics --------------------------------------------------------------------------


def test_e11_t05_config_errors_name_their_path(tmp_path: Path) -> None:
    """A missing `ar` block, a foreign covariate and an unknown parameter are rejected with their path."""
    data = _prepared(parse_project_config(_raw(tmp_path, _emission("coefficients", 1))), np.arange(30.0))
    raw = _raw(tmp_path, _emission("coefficients", 1))
    del raw["model"]["emission"]["ar"]
    with pytest.raises(ConfigError) as err:
        build_model_spec(parse_project_config(raw), data)
    assert err.value.path == "model.emission.ar"
    raw = _raw(tmp_path, _emission("coefficients", 1))
    raw["model"]["emission"]["params"]["bogus"] = {"mode": "free"}
    with pytest.raises(ConfigError) as err2:
        build_model_spec(parse_project_config(raw), data)
    assert err2.value.path == "model.emission.params.bogus"
    # lag columns of a different order than the model's are refused
    wrong = _raw(tmp_path, _emission("coefficients", 2))
    with pytest.raises(ConfigError, match="own-lag columns"):
        build_model_spec(parse_project_config(wrong), data)


def test_e11_t05_spectral_radius_is_the_companion_eigenvalue() -> None:
    """AR(1) coefficient 0.9 and AR(2) `(0.5, 0.3)` have the textbook companion spectral radii."""
    fam1 = MSARFamily(p=1)
    r1 = fam1.spectral_radius({"emission.ar": jnp.asarray([0.9, -1.2]).reshape(2, 1, 1, 1)})
    np.testing.assert_allclose(np.asarray(r1), [0.9, 1.2])
    fam2 = MSARFamily(p=2)
    r2 = fam2.spectral_radius({"emission.ar": jnp.asarray([[0.5, 0.3]]).reshape(1, 2, 1, 1)})
    expected = max(abs(np.roots([1.0, -0.5, -0.3])))
    np.testing.assert_allclose(np.asarray(r2), [expected])


def test_e11_t05_sample_path_follows_the_autoregression() -> None:
    """A long simulated one-regime AR(1) has the stationary variance `sd^2 / (1 - phi^2)` (within 4 SE)."""
    fam = MSARFamily(p=1)
    theta = {
        "emission.intercept": jnp.asarray([[0.0]]),
        "emission.ar": jnp.asarray([0.8]).reshape(1, 1, 1, 1),
        "emission.scale": jnp.asarray([[1.0]]),
    }
    import jax

    n = 40_000
    y = np.asarray(fam.sample_path(jax.random.key(0), jnp.zeros(n, dtype=jnp.int32), jnp.ones(n), theta))
    acf1 = np.corrcoef(y[1:, 0], y[:-1, 0])[0, 1]
    assert abs(acf1 - 0.8) < 4 * np.sqrt((1 - 0.8**2) / n)
    assert abs(y.var() - 1 / (1 - 0.64)) < 0.15 * (1 / (1 - 0.64))


def test_e11_t05_steps_space_decodes_and_has_trivial_memory_marginals(tmp_path: Path) -> None:
    """Forward-backward and Viterbi run on the tuple space; `memory_marginals` is the trivial `(T, 1)`."""
    from hsmmkit.engine.augmented import memory_marginals

    rng = np.random.default_rng(5)
    y = np.r_[rng.normal(-1, 0.3, 40), rng.normal(2, 0.3, 40)]
    cfg = parse_project_config(_raw(tmp_path, _emission("mean_adjusted", 2)))
    data = _prepared(cfg, y)
    spec = build_model_spec(cfg, data)
    space = build_expanded(spec)
    mgr = ParamManager(spec.param_specs)
    theta = dict(mgr.unflatten(mgr.init_u()))
    theta["emission.mean"] = jnp.asarray([[-1.0], [2.0]])
    theta["emission.scale"] = jnp.asarray([[0.4], [0.4]])
    engine = ExpandedHMMEngine()
    post = engine.forward_backward(theta, spec, space, data)
    np.testing.assert_allclose(np.asarray(post.smoothed).sum(axis=1), 1.0, atol=1e-10)
    assert memory_marginals(post).shape == (80, 1)
    regimes = np.asarray(space.index.regime_of[engine.viterbi(theta, spec, space, data)])
    assert (regimes[5:40] == 0).mean() > 0.9
    assert (regimes[45:] == 1).mean() > 0.9
