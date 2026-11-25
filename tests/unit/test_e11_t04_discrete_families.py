"""E11-T04: Poisson, Bernoulli and Categorical emissions: closed-form M-steps, conjugate priors, config."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.contracts import ObservationSet
from hsmmkit.data.prepare import Preparer
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.conjugate import BetaConjugate, DirichletConjugate, GammaConjugate, posterior
from hsmmkit.model.emission.bernoulli import BernoulliFamily
from hsmmkit.model.emission.categorical import CategoricalFamily
from hsmmkit.model.emission.poisson import PoissonFamily
from hsmmkit.utils.errors import ConfigError, DataError
from tests.helpers.core_config import core_raw


def test_e11_t04_poisson_exposure_mle_is_sum_y_over_sum_dt() -> None:
    """AC3: with irregular `dt`, the one-state MLE of the rate is `sum(y) / sum(dt)`."""
    rng = np.random.default_rng(0)
    dt = jnp.asarray(rng.uniform(0.2, 3.0, size=300))
    y = jnp.asarray(rng.poisson(4.0 * np.asarray(dt)).astype(np.float64))[:, None]
    fam = PoissonFamily()
    theta = fam.m_step(fam.sufficient_stats(y, dt, jnp.ones((300, 1))))
    assert theta is not None
    np.testing.assert_allclose(
        float(jnp.exp(theta["emission.eta"][0, 0])), float(jnp.sum(y) / jnp.sum(dt)), rtol=1e-12
    )


def test_e11_t04_m_steps_are_the_weighted_frequencies() -> None:
    """Bernoulli `p = sum(gamma y) / sum(gamma)`; Categorical `p_k = sum(gamma 1[y = k]) / sum(gamma)`."""
    rng = np.random.default_rng(1)
    t = 500
    gamma = jnp.asarray(rng.dirichlet(np.ones(2), size=t))
    yb = jnp.asarray(rng.integers(0, 2, size=(t, 1)).astype(np.float64))
    fb = BernoulliFamily()
    eta = fb.m_step(fb.sufficient_stats(yb, jnp.ones(t), gamma))["emission.eta"]  # type: ignore[index]
    p = np.asarray(gamma.T @ yb) / np.asarray(jnp.sum(gamma, axis=0))[:, None]
    np.testing.assert_allclose(special.expit(np.asarray(eta)), p, rtol=1e-12)
    yc = jnp.asarray(rng.integers(0, 3, size=(t, 1)).astype(np.float64))
    fc = CategoricalFamily(n_categories=3)
    eta_c = np.asarray(fc.m_step(fc.sufficient_stats(yc, jnp.ones(t), gamma))["emission.eta"])  # type: ignore[index]
    counts = np.stack(
        [np.asarray(gamma.T @ (np.asarray(yc) == k).astype(float))[:, 0] for k in range(3)], axis=1
    )
    pk = counts / counts.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(eta_c, np.log(pk[:, 1:] / pk[:, :1]), rtol=1e-10)


def test_e11_t04_m_step_clips_degenerate_probabilities() -> None:
    """A regime with no successes gets `p = 1e-12` instead of `-inf`."""
    fam = BernoulliFamily()
    eta = fam.m_step({"w": jnp.asarray([5.0]), "wy": jnp.asarray([[0.0]])})["emission.eta"]  # type: ignore[index]
    assert bool(jnp.isfinite(eta).all())


def test_e11_t04_covariate_links_have_no_closed_form() -> None:
    """With covariate links `sufficient_stats` and `m_step` answer `None` (ECM)."""
    y, dt, gamma = jnp.ones((4, 1)), jnp.ones(4), jnp.full((4, 2), 0.5)
    for fam in (
        PoissonFamily(mean_idx=(0,)),
        BernoulliFamily(mean_idx=(0,)),
        CategoricalFamily(mean_idx=(0,)),
    ):
        assert fam.sufficient_stats(y, dt, gamma) is None
        assert fam.m_step(None) is None


def test_e11_t04_conjugate_posteriors_match_the_closed_forms() -> None:
    """AC4: zero statistics give the prior; data give the Gamma / Beta / Dirichlet updates."""
    g, b, d = GammaConjugate(2.0, 3.0), BetaConjugate(2.0, 5.0), DirichletConjugate(0.5)
    z = g.update(g.zero_stats(2, 3))
    np.testing.assert_allclose(np.asarray(z["a"]), 2.0)
    np.testing.assert_allclose(np.asarray(z["b"]), 3.0)
    zb = b.update(b.zero_stats(2, 3))
    np.testing.assert_allclose(np.asarray(zb["a"]), 2.0)
    np.testing.assert_allclose(np.asarray(zb["b"]), 5.0)
    zd = d.update(d.zero_stats(2, 4))
    np.testing.assert_allclose(np.asarray(zd["alpha"]), 0.5)
    stats_p = {"w_s": jnp.asarray([10.0]), "wy": jnp.asarray([[7.0]])}
    post = posterior(stats_p, g)
    assert float(post["a"][0, 0]) == 9.0
    assert float(post["b"][0, 0]) == 13.0
    assert float(g.expected_log_params(post)["rate"][0, 0]) == pytest.approx(9.0 / 13.0)
    post_b = posterior({"w": jnp.asarray([10.0]), "wy": jnp.asarray([[7.0]])}, b)
    assert (float(post_b["a"][0, 0]), float(post_b["b"][0, 0])) == (9.0, 8.0)
    el = b.expected_log_params(post_b)
    assert float(el["log_p"][0, 0]) == pytest.approx(float(special.digamma(9.0) - special.digamma(17.0)))
    post_d = posterior(
        {"w": jnp.asarray([6.0]), "counts": jnp.asarray([[1.0, 2.0, 3.0]])}, DirichletConjugate(1.0)
    )
    np.testing.assert_allclose(np.asarray(post_d["alpha"]), [[2.0, 3.0, 4.0]])


def _cfg(tmp_path: Path, family: str, params: dict, **extra: object):  # noqa: ANN202
    raw = core_raw(tmp_path)
    raw["model"]["emission"] = {"family": family, "params": params, **extra}
    return parse_project_config(raw)


def _build(cfg, n_series: int = 1):  # noqa: ANN001, ANN202
    return build_model_spec_from(
        cfg, BuildInputs(n_series=n_series, cov_names={}, clock="steps", nominal_dt=1.0)
    )


def test_e11_t04_config_converts_natural_units_to_eta(tmp_path: Path) -> None:
    """`rate` / `prob` / `probs` init values become `log`, `logit` and log-ratios."""
    spec = _build(_cfg(tmp_path, "poisson", {"rate": {"mode": "free", "init": [2.0, 5.0]}}))
    eta = next(p for p in spec.param_specs if p.name == "emission.eta")
    np.testing.assert_allclose(eta.init, np.log([[2.0], [5.0]]))
    assert spec.emission.scale_by_dt  # exposure by default
    spec = _build(_cfg(tmp_path, "bernoulli", {"prob": {"mode": "free", "init": [0.2, 0.8]}}))
    eta = next(p for p in spec.param_specs if p.name == "emission.eta")
    np.testing.assert_allclose(eta.init, special.logit([[0.2], [0.8]]))
    spec = _build(_cfg(tmp_path, "categorical", {"probs": {"mode": "free", "init": [0.5, 0.3, 0.2]}}))
    eta = next(p for p in spec.param_specs if p.name == "emission.eta")
    assert eta.shape == (2, 2)
    np.testing.assert_allclose(eta.init[0], np.log([0.3 / 0.5, 0.2 / 0.5]))


def test_e11_t04_config_errors(tmp_path: Path) -> None:
    """Missing / invalid parameters and unsupported modes are `ConfigError`s with a path."""
    with pytest.raises(ConfigError, match=r"params\.rate"):
        _build(_cfg(tmp_path, "poisson", {"prob": {"mode": "free"}}))
    with pytest.raises(ConfigError, match="positive"):
        _build(_cfg(tmp_path, "poisson", {"rate": {"mode": "free", "init": [0.0, 1.0]}}))
    with pytest.raises(ConfigError, match=r"\(0, 1\)"):
        _build(_cfg(tmp_path, "bernoulli", {"prob": {"mode": "free", "init": [1.0, 0.5]}}))
    with pytest.raises(ConfigError, match="sum to 1"):
        _build(_cfg(tmp_path, "categorical", {"probs": {"mode": "free", "init": [0.5, 0.3, 0.3]}}))
    with pytest.raises(ConfigError, match="one series"):
        _build(_cfg(tmp_path, "categorical", {"probs": {"mode": "free", "init": [0.5, 0.5]}}), n_series=2)
    with pytest.raises(ConfigError, match="mode"):
        _build(_cfg(tmp_path, "poisson", {"rate": {"mode": "fixed", "values": [1.0, 2.0]}}))


def test_e11_t04_multi_series_poisson_is_wrapped_per_series(tmp_path: Path) -> None:
    """`M > 1` builds the per-series product so that masks act per series."""
    from hsmmkit.model.emission.product import ProductEmission

    spec = _build(_cfg(tmp_path, "poisson", {"rate": {"mode": "free", "init": [2.0, 5.0]}}), n_series=2)
    assert isinstance(spec.emission, ProductEmission)
    eta = next(p for p in spec.param_specs if p.name == "emission.eta")
    assert eta.shape == (2, 2)


@pytest.mark.parametrize(
    ("support", "bad"), [("count", 1.5), ("count", -1.0), ("binary", 2.0), ("categorical", -1.0)]
)
def test_e11_t04_preparer_rejects_out_of_support_data(support: str, bad: float) -> None:
    """AC5: data violating the support are rejected with the preparer's message."""
    from hsmmkit.config.schema import DataSection

    obs = ObservationSet(("y",), None, np.array([[0.0], [1.0], [bad]]), "memory", "")
    section = DataSection.model_validate(
        {"source": {"kind": "numpy", "observation_columns": ["y"]}, "clock": "steps"}
    )
    with pytest.raises(DataError, match="support violated"):
        Preparer(section, support=support).fit(obs, [], 2).transform(obs, [])  # type: ignore[arg-type]
