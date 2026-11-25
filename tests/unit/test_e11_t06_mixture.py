"""E11-T06: per-state finite mixture: K = 1 identity, ordering, recovery, config, relabelling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize, special, stats

from hsmmkit.config.gates import used_gates
from hsmmkit.config.schema import EstimationSection, parse_project_config
from hsmmkit.data.contracts import ObservationSet
from hsmmkit.data.prepare import Preparer
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.estimation import fit
from hsmmkit.model.build import build_model_spec
from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.emission.mixture import MixtureFamily
from hsmmkit.model.emission.product import ProductEmission
from hsmmkit.model.lattice import post_sort_states
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.statespace.builders import build_expanded
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw


def _emission(base: str, k: int, params: dict[str, Any]) -> dict[str, Any]:
    return {"family": "mixture", "mixture": {"components": k, "base": base}, "params": params}


def _gauss_params(init_mean: list[float], init_scale: list[float], **scale_extra: str) -> dict[str, Any]:
    return {
        "mean": {"mode": "free", "init": init_mean},
        "scale": {"mode": "free", "init": init_scale, **scale_extra},
    }


def _cfg(tmp_path: Path, emission: dict[str, Any], n_states: int = 2) -> Any:  # noqa: ANN401
    raw = core_raw(tmp_path)
    raw["model"]["n_states"] = n_states
    raw["model"]["emission"] = emission
    raw["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [20.0] * n_states
    return parse_project_config(raw)


def _family(tmp_path: Path, emission: dict[str, Any], n_states: int = 2, n_series: int = 1) -> Any:  # noqa: ANN401
    cfg = _cfg(tmp_path, emission, max(n_states, 2))  # the config needs >= 2 regimes; a family may have one
    return MixtureFamily.from_config(cfg.model.emission, n_states, n_series, ())


# --- AC1: K = 1 is the base family, bitwise ------------------------------------------------------------


def test_e11_t06_k1_is_the_base_family_bitwise() -> None:
    """AC1: `K = 1` gives bitwise equal `log_prob` and `sample` (same key), and the base's parameter specs."""
    base = GaussianFamily()
    fam = MixtureFamily.of(base, 1)
    rng = np.random.default_rng(0)
    theta = {
        "emission.mean": jnp.asarray(rng.normal(size=(2, 1))),
        "emission.scale": jnp.asarray(rng.uniform(0.5, 2, (2, 1))),
    }
    y, dt = jnp.asarray(rng.normal(size=(30, 1))), jnp.ones(30)
    assert np.array_equal(
        np.asarray(fam.log_prob(y, dt, theta, None)), np.asarray(base.log_prob(y, dt, theta, None))
    )
    regime = jnp.asarray(rng.integers(0, 2, 30))
    key = jax.random.key(3)
    assert np.array_equal(
        np.asarray(fam.sample(key, regime, dt, theta, None)),
        np.asarray(base.sample(key, regime, dt, theta, None)),
    )
    assert [s.name for s in fam.param_specs(2, 1, ())] == [s.name for s in base.param_specs(2, 1, ())]


def test_e11_t06_k1_from_config_has_the_base_specs(tmp_path: Path) -> None:
    """`K = 1` from a config: no weights spec, the base's names and shapes."""
    fam, specs = _family(tmp_path, _emission("gaussian", 1, _gauss_params([0.0, 0.1], [1.0, 2.0])))
    assert fam.k == 1
    assert [(s.name, s.shape) for s in specs] == [("emission.mean", (2, 1)), ("emission.scale", (2, 1))]


# --- the mixture density ------------------------------------------------------------------------------


def test_e11_t06_two_component_log_prob_and_moments(tmp_path: Path) -> None:
    """`log f = logsumexp(log w + log g)`; mean and variance of the mixture are the textbook ones."""
    fam, specs = _family(
        tmp_path, _emission("gaussian", 2, _gauss_params([-1.0, 1.0, 0.0, 2.0], [0.5, 1.5, 1.0, 0.7]))
    )
    assert [s.name for s in specs] == ["emission.mean", "emission.scale", "emission.weights"]
    mgr = ParamManager(specs)
    theta = dict(mgr.unflatten(mgr.init_u()))
    theta["emission.weights"] = jnp.asarray([[0.3, 0.7], [0.6, 0.4]])
    y = jnp.linspace(-3, 3, 11).reshape(-1, 1)
    lp = np.asarray(fam.log_prob(y, jnp.ones(11), theta, None))
    mu, sd, w = (np.asarray(theta[k]) for k in ("emission.mean", "emission.scale", "emission.weights"))
    for i in range(2):
        ref = special.logsumexp(
            [
                np.log(w[i, c]) + stats.norm.logpdf(np.asarray(y)[:, 0], mu[2 * i + c, 0], sd[2 * i + c, 0])
                for c in range(2)
            ],
            axis=0,
        )
        np.testing.assert_allclose(lp[:, i], ref, atol=1e-12)
    mean, var = fam.moments(jnp.asarray([0, 1]), jnp.ones(2), theta, None)
    for i in range(2):
        m_ref = sum(w[i, c] * mu[2 * i + c, 0] for c in range(2))
        v_ref = sum(w[i, c] * (sd[2 * i + c, 0] ** 2 + mu[2 * i + c, 0] ** 2) for c in range(2)) - m_ref**2
        np.testing.assert_allclose([float(mean[i, 0]), float(var[i, 0])], [m_ref, v_ref], rtol=1e-12)


def test_e11_t06_product_base_masks_series_and_mixes_components(tmp_path: Path) -> None:
    """`M = 2` Poisson: components are products over the series; a masked series drops out of each."""
    params = {"rate": {"mode": "free", "init": [1.0, 5.0, 2.0, 8.0]}}
    fam, specs = _family(tmp_path, _emission("poisson", 2, params), n_series=2)
    assert isinstance(fam.base, ProductEmission)
    mgr = ParamManager(specs)
    theta = dict(mgr.unflatten(mgr.init_u()))
    y = jnp.asarray([[1.0, 4.0], [2.0, 0.0], [0.0, 7.0]])
    mask = jnp.asarray([[True, True], [True, False], [False, True]])
    dt = jnp.ones(3)
    got = np.asarray(fam.log_prob_masked(y, mask, dt, theta, None))
    lam = np.exp(np.asarray(theta["emission.eta"]))  # (4, 2)
    w = np.asarray(theta["emission.weights"])
    for t in range(3):
        for i in range(2):
            comp = [
                np.log(w[i, c])
                + sum(stats.poisson.logpmf(float(y[t, m]), lam[2 * i + c, m]) for m in range(2) if mask[t, m])
                for c in range(2)
            ]
            np.testing.assert_allclose(got[t, i], special.logsumexp(comp), atol=1e-12)


def test_e11_t06_sampling_follows_the_weights() -> None:
    """The component mix of `sample` has the configured weights (3 SE)."""
    fam = MixtureFamily.of(GaussianFamily(), 2)
    theta = {
        "emission.mean": jnp.asarray([[-10.0], [10.0]]),
        "emission.scale": jnp.asarray([[0.1], [0.1]]),
        "emission.weights": jnp.asarray([[0.25, 0.75]]),
    }
    n = 20_000
    s = np.asarray(fam.sample(jax.random.key(0), jnp.zeros(n, dtype=jnp.int32), jnp.ones(n), theta, None))
    p = float((s > 0).mean())
    assert abs(p - 0.75) < 3 * np.sqrt(0.75 * 0.25 / n)


# --- AC4: ordering within a state -----------------------------------------------------------------------


def test_e11_t06_ordered_rows_are_increasing_for_any_u(tmp_path: Path) -> None:
    """AC4 (property): each state's components are strictly increasing in the ordered parameter, any `u`."""
    params = _gauss_params(
        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], [0.5, 1.5, 0.5, 1.5, 0.5, 1.5], constraint="increasing"
    )
    fam, specs = _family(tmp_path, _emission("gaussian", 2, params), n_states=3)
    assert fam.ordered == ("emission.scale",)
    scale = next(s for s in specs if s.name == "emission.scale")
    assert scale.shape == (3, 2)
    assert scale.domain == "increasing_positive"
    mgr = ParamManager(specs)
    rng = np.random.default_rng(1)
    for _ in range(50):
        theta = mgr.unflatten(jnp.asarray(rng.normal(scale=3.0, size=mgr.n_free)))
        assert bool(jnp.all(jnp.diff(theta["emission.scale"], axis=1) > 0))
    theta0 = mgr.unflatten(mgr.init_u())
    np.testing.assert_allclose(np.asarray(mgr.flatten(theta0)), np.asarray(mgr.init_u()), atol=1e-12)


def test_e11_t06_unordered_init_for_an_ordered_parameter_is_rejected(tmp_path: Path) -> None:
    """The init of an increasing parameter must itself increase along the components."""
    params = _gauss_params([0.0, 0.0], [2.0, 1.0], constraint="increasing")
    with pytest.raises(ConfigError, match="increase"):
        _family(tmp_path, _emission("gaussian", 2, params), n_states=1)


# --- AC3: recovery of weights and scales -----------------------------------------------------------------


def test_e11_t06_gaussian_mixture_recovers_weights_and_scales(tmp_path: Path) -> None:
    """AC3: `T = 20000`, one state: weights within 0.05 and scales within 10 % (maximum likelihood)."""
    params = _gauss_params([-0.1, 0.4], [0.8, 1.2], constraint="increasing")
    fam, specs = _family(tmp_path, _emission("gaussian", 2, params), n_states=1)
    rng = np.random.default_rng(7)
    t = 20_000
    comp = rng.random(t) < 0.65
    y = np.where(comp, rng.normal(0.5, 1.6, t), rng.normal(0.0, 0.6, t)).reshape(-1, 1)
    mgr = ParamManager(specs)
    yj, dt = jnp.asarray(y), jnp.ones(t)

    def nll(u: jax.Array) -> jax.Array:
        return -jnp.sum(fam.log_prob(yj, dt, mgr.unflatten(u), None))

    vg = jax.jit(jax.value_and_grad(nll))
    res = optimize.minimize(
        lambda u: tuple(np.asarray(v, dtype=np.float64) for v in vg(jnp.asarray(u))),
        np.asarray(mgr.init_u()),
        jac=True,
        method="L-BFGS-B",
    )
    theta = mgr.unflatten(jnp.asarray(res.x))
    w, scale = np.asarray(theta["emission.weights"])[0], np.asarray(theta["emission.scale"])[0]
    assert np.all(np.abs(w - np.array([0.35, 0.65])) <= 0.05)
    assert np.all(np.abs(scale / np.array([0.6, 1.6]) - 1.0) <= 0.10)
    assert scale[0] < scale[1]  # the ordering prevents component label switching


# --- config -----------------------------------------------------------------------------------------------


def test_e11_t06_gate_removed_and_misuse_rejected(tmp_path: Path) -> None:
    """AC5: the key builds (no gate); `mixture` without `family: mixture` (and vice versa) is rejected."""
    cfg = _cfg(tmp_path, _emission("gaussian", 2, _gauss_params([0.0, 1.0, 0.0, 1.0], [1.0, 2.0, 1.0, 2.0])))
    assert all(path != "model.emission.mixture" for path, _ in used_gates(cfg))
    y = np.random.default_rng(0).normal(size=60)
    times = np.arange(60).astype("datetime64[D]").astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    spec = build_model_spec(cfg, data)
    assert spec.emission.k == 2
    assert [s.name for s in spec.param_specs if s.name.startswith("emission")] == [
        "emission.mean",
        "emission.scale",
        "emission.weights",
    ]
    raw = core_raw(tmp_path)
    raw["model"]["emission"]["mixture"] = {"components": 2, "base": "gaussian"}
    with pytest.raises(ConfigError) as err:
        build_model_spec(parse_project_config(raw), data)
    assert err.value.path == "model.emission.mixture"
    for bad in ("mixture", "nope"):
        raw = core_raw(tmp_path)
        raw["model"]["emission"] = _emission(bad, 2, {})
        with pytest.raises(ConfigError) as err2:
            build_model_spec(parse_project_config(raw), data)
        assert err2.value.path == "model.emission.mixture.base"


def test_e11_t06_hmm_with_mixture_emission_fits_and_post_sort_is_consistent(tmp_path: Path) -> None:
    """A 2-regime HMM with a 2-component mixture per regime: finite fit; relabelling permutes whole blocks."""
    rng = np.random.default_rng(2)
    y = np.concatenate([rng.normal(0, s, 150) * 1.0 for s in (0.5, 2.0, 0.5, 2.0)])
    times = np.arange(y.shape[0]).astype("datetime64[D]").astype("datetime64[ns]")
    cfg = _cfg(
        tmp_path,
        _emission(
            "gaussian",
            2,
            _gauss_params([-0.1, 0.1, -0.1, 0.1], [0.3, 0.7, 1.5, 2.5], constraint="increasing"),
        ),
    )
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    spec = build_model_spec(cfg, data)
    res = fit(
        spec, data, EstimationSection(restarts=1, max_iter=60, init=["rolling_kmeans"], device="cpu"), seed=0
    )  # type: ignore[call-arg]
    assert np.isfinite(res.loglik)
    theta = dict(res.params)
    flip = jnp.asarray([1, 0])
    flipped = {
        k: spec.emission.permute_state_axis(k, v, flip) for k, v in theta.items() if k.startswith("emission.")
    }
    swapped, perm = post_sort_states({**theta, **flipped}, spec, by="emission.scale")
    # states swapped on purpose (whole component blocks): sorting by the weighted scale restores the order
    assert perm.tolist() == [1, 0]
    for name in ("emission.mean", "emission.scale", "emission.weights"):
        np.testing.assert_allclose(np.asarray(swapped[name]), np.asarray(theta[name]), atol=1e-12)
    # `post_sort_states` leaves a model's likelihood unchanged for a pure relabelling
    space = build_expanded(spec)
    ll = float(ExpandedHMMEngine().loglik(theta, spec, space, data))
    assert np.isfinite(ll)
    assert abs(ll - res.loglik) <= 1e-9 * abs(ll)


def test_e11_t06_k1_keeps_the_base_state_ordering(tmp_path: Path) -> None:
    """`K = 1` with `constraint: increasing` is the base's own state-axis ordering (review finding)."""
    params = _gauss_params([0.0, 0.1], [1.0, 2.0], constraint="increasing")
    fam, specs = _family(tmp_path, _emission("gaussian", 1, params))
    assert fam.ordered == ()
    scale = next(s for s in specs if s.name == "emission.scale")
    assert scale.constraint == "increasing"
    assert scale.shape == (2,)


def test_e11_t06_inits_keep_the_mixture_shapes_and_the_ordering(tmp_path: Path) -> None:
    """`random` and `lattice_jitter` give every parameter its spec shape and keep rows increasing (review)."""
    from hsmmkit.estimation.init import lattice_jitter, random

    y = np.random.default_rng(0).normal(size=80)
    times = np.arange(80).astype("datetime64[D]").astype("datetime64[ns]")
    cfg = _cfg(
        tmp_path,
        _emission(
            "gaussian",
            2,
            _gauss_params([-0.1, 0.1, -0.1, 0.1], [0.5, 0.6, 1.5, 1.6], constraint="increasing"),
        ),
    )
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    spec = build_model_spec(cfg, data)
    mgr = ParamManager(spec.param_specs)
    for init in (random, lattice_jitter):
        for seed in range(10):
            key = jax.random.key(seed)
            theta = init(data, spec, key) if init is random else init(spec, key)
            for ps in spec.param_specs:
                assert theta[ps.name].shape == ps.shape, (init.__name__, ps.name)
            assert bool(jnp.all(jnp.isfinite(mgr.flatten(theta))))
            assert bool(jnp.all(jnp.diff(theta["emission.scale"], axis=1) > 0))
