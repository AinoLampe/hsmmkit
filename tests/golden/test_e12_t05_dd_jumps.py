"""E12-T05 golden: duration-dependent jumps and per-state durations in the explicit engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.contracts import ObservationSet, PreparedDataset
from hsmmkit.data.prepare import Preparer
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.model.build import build_model_spec
from hsmmkit.model.duration.per_state import PerStateDuration
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.model.spec import ModelSpec
from hsmmkit.statespace.builders import build_explicit
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw
from tests.reference import ref_hsmm_dd


def _raw(tmp_path: Path, n: int, d_max: int, *, dd: bool, engine: str = "explicit") -> dict[str, Any]:
    raw = core_raw(tmp_path)
    m = raw["model"]
    m.update({"n_states": n, "engine": engine, "d_max": d_max})
    m["emission"]["params"] = {
        "mean": {"mode": "free", "init": [0.4 * i for i in range(n)]},
        "scale": {"mode": "free", "init": [0.5 + 0.2 * i for i in range(n)]},
    }
    m["duration"] = {
        "family": "nonparametric",
        "params": {"logits": {"mode": "free"}},
        "duration_dependent_jumps": dd,
    }
    return raw


def _build(tmp_path: Path, n: int, t: int, d_max: int, seed: int, *, dd: bool) -> tuple[Any, Any, Any]:
    cfg = parse_project_config(_raw(tmp_path, n, d_max, dd=dd))
    y = np.random.default_rng(seed).normal(size=t)
    times = pd.bdate_range("2020-01-01", periods=t).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    return cfg, data, build_model_spec(cfg, data)


def _ref_inputs(
    spec: ModelSpec, theta: dict[str, Any], data: PreparedDataset, d_max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = build_explicit(spec).layout
    pi_d = np.asarray(
        spec.transition.jump_probs_dur(theta, None, layout, jnp.arange(1, d_max + 1, dtype=jnp.float64))
    )
    with np.errstate(divide="ignore"):
        log_trans_d = np.log(pi_d)
    log_pmf = np.asarray(spec.duration.log_pmf(theta, d_max, data.nominal_dt, None))
    y = np.asarray(data.observations)[:, 0]
    mean = np.asarray(theta["emission.mean"]).reshape(-1)
    scale = np.asarray(theta["emission.scale"]).reshape(-1)
    log_emis = -0.5 * ((y[:, None] - mean) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)
    return log_trans_d, log_pmf, log_emis


@pytest.mark.parametrize("seed", range(6))
def test_e12_t05_dd_engine_matches_enumeration(seed: int, tmp_path: Path) -> None:
    """AC3: loglik, smoothed marginals and the Viterbi path equal the enumeration reference (<= 1e-8)."""
    rng = np.random.default_rng(seed)
    n, t, d_max = int(rng.integers(3, 5)), int(rng.integers(5, 9)), int(rng.integers(2, 5))
    _, data, spec = _build(tmp_path, n, t, d_max, seed, dd=True)
    mgr = ParamManager(spec.param_specs)
    theta = dict(mgr.unflatten(jnp.asarray(rng.normal(size=mgr.n_free))))
    theta["emission.mean"] = jnp.asarray(rng.normal(size=n))
    theta["emission.scale"] = jnp.asarray(rng.uniform(0.4, 1.5, size=n))
    assert "transition.c" in theta
    space = build_explicit(spec)
    eng = ExplicitDurationEngine()
    post = eng.forward_backward(theta, spec, space, data)
    log_trans_d, log_pmf, log_emis = _ref_inputs(spec, theta, data, d_max)
    ll, smoothed, path = ref_hsmm_dd.brute_force_dd(
        np.full(n, -np.log(n)), log_trans_d, log_pmf, log_emis, d_max
    )
    assert abs(float(post.loglik) - ll) <= 1e-8 * max(1.0, abs(ll))
    assert abs(float(eng.loglik(theta, spec, space, data)) - ll) <= 1e-8 * max(1.0, abs(ll))
    np.testing.assert_allclose(np.asarray(post.regime_smoothed()), smoothed, atol=1e-8)
    got = np.asarray(space.index.regime_of[eng.viterbi(theta, spec, space, data)])
    # the best segmentation is unique for generic random parameters
    np.testing.assert_array_equal(got, path)


def test_e12_t05_zero_coefficients_equal_the_standard_engine(tmp_path: Path) -> None:
    """AC2: with `c = 0` the dd model gives the constant-`pi` log-likelihood (<= 1e-10: the sums reorder)."""
    n, t, d_max = 3, 60, 12
    _, data, spec_dd = _build(tmp_path, n, t, d_max, 1, dd=True)
    _, _, spec_std = _build(tmp_path, n, t, d_max, 1, dd=False)
    mgr_dd, mgr_std = ParamManager(spec_dd.param_specs), ParamManager(spec_std.param_specs)
    rng = np.random.default_rng(3)
    theta_std = dict(mgr_std.unflatten(jnp.asarray(rng.normal(size=mgr_std.n_free))))
    theta_std["emission.mean"] = jnp.asarray(rng.normal(size=n))
    theta_std["emission.scale"] = jnp.asarray(rng.uniform(0.4, 1.5, size=n))
    # the same jump matrix in the flat layout: reference target logit 0, the others log(pi_j / pi_ref)
    pi = np.asarray(theta_std["transition.pi_rows"])  # (N, N-1) over targets j != i
    a = []
    for i in range(n):
        targets = [j for j in range(n) if j != i]
        a.extend(np.log(pi[i, k] / pi[i, 0]) for k in range(1, len(targets)))
    theta_dd = {k: v for k, v in theta_std.items() if not k.startswith("transition.")}
    theta_dd.update(
        {k: v for k, v in dict(mgr_dd.unflatten(mgr_dd.init_u())).items() if k.startswith("transition.")}
    )
    theta_dd["transition.a"] = jnp.asarray(a)
    theta_dd["transition.c"] = jnp.zeros_like(theta_dd["transition.c"])
    a_ll = float(ExplicitDurationEngine().loglik(theta_std, spec_std, build_explicit(spec_std), data))
    b_ll = float(ExplicitDurationEngine().loglik(theta_dd, spec_dd, build_explicit(spec_dd), data))
    assert abs(a_ll - b_ll) <= 1e-10 * abs(a_ll)


def test_e12_t05_expanded_engine_rejects_dd_jumps(tmp_path: Path) -> None:
    """AC4: the documented `ConfigError`."""
    raw = _raw(tmp_path, 2, 20, dd=True, engine="expanded")
    raw["model"]["duration"] = {
        "family": "geometric",
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [5.0, 5.0]}},
        "duration_dependent_jumps": True,
    }
    with pytest.raises(
        ConfigError, match="only the explicit engine can express duration-dependent jumps"
    ) as err:
        parse_project_config(raw)
    assert err.value.path == "model.duration.duration_dependent_jumps"


def test_e12_t05_per_state_durations_build_and_stack(tmp_path: Path) -> None:
    """AC5 (duration block): a geometric / gamma / nonparametric mix builds with prefixed parameters."""
    raw = core_raw(tmp_path)
    m = raw["model"]
    m.update({"n_states": 3, "engine": "explicit", "d_max": 60})
    m["emission"]["params"] = {
        "mean": {"mode": "free", "init": [0.0, 0.2, 0.4]},
        "scale": {"mode": "free", "init": [0.5, 0.7, 0.9]},
    }
    m["duration"] = {
        "per_state": [
            {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 8.0}}},
            {
                "family": "gamma",
                "params": {
                    "shape": {"mode": "free", "init": 2.0},
                    "scale": {"mode": "free", "init": 5.0, "init_mean_sojourn_steps": None},
                },
            },
            {"family": "nonparametric", "params": {"logits": {"mode": "free"}}},
        ]
    }
    cfg = parse_project_config(raw)
    y = np.random.default_rng(0).normal(size=80)
    times = pd.bdate_range("2020-01-01", periods=80).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    spec = build_model_spec(cfg, data)
    assert isinstance(spec.duration, PerStateDuration)
    names = [s.name for s in spec.param_specs if s.name.startswith("duration.")]
    assert names == ["duration.s0.rate", "duration.s1.shape", "duration.s1.scale", "duration.s2.logits"]
    mgr = ParamManager(spec.param_specs)
    theta = mgr.unflatten(mgr.init_u())
    pmf = np.exp(np.asarray(spec.duration.log_pmf(theta, 60, data.nominal_dt, None)))
    assert pmf.shape == (3, 60)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-10)
    ll = ExplicitDurationEngine().loglik(theta, spec, build_explicit(spec), data)
    assert np.isfinite(float(ll))
    means = np.asarray(spec.duration.mean_sojourn(theta, data.nominal_dt))
    assert means.shape == (3,)


def test_e12_t05_review_guards(tmp_path: Path) -> None:
    """Review fixes: dd + absorbing rejected, no relabelling of per-state durations, size guard, forecast."""
    from hsmmkit.engine import explicit_dd
    from hsmmkit.inference.forecast import _reject_autoregressive
    from hsmmkit.model.lattice import has_identifiability_guard
    from hsmmkit.utils.errors import NumericalError

    raw = _raw(tmp_path, 3, 10, dd=True)
    raw["model"]["topology"] = "left_to_right"
    cfg_abs = parse_project_config(raw)
    y0 = np.random.default_rng(0).normal(size=30)
    t0 = pd.bdate_range("2020-01-01", periods=30).to_numpy().astype("datetime64[ns]")
    obs0 = ObservationSet(("x",), t0, y0.reshape(-1, 1), "memory", "s")
    data0 = Preparer(cfg_abs.data).fit(obs0, [], train_end=t0[-1]).transform(obs0, [])
    with pytest.raises(ConfigError) as err:
        build_model_spec(cfg_abs, data0)
    assert err.value.path == "model.topology"

    _, data, spec = _build(tmp_path, 3, 30, 10, 0, dd=True)
    with pytest.raises(ConfigError, match="forecasting with duration-dependent jumps"):
        _reject_autoregressive(spec)
    old = explicit_dd.MAX_STATES_SQUARED
    explicit_dd.MAX_STATES_SQUARED = 10.0
    try:
        with pytest.raises(NumericalError):
            ExplicitDurationEngine().loglik({}, spec, build_explicit(spec), data)
    finally:
        explicit_dd.MAX_STATES_SQUARED = old

    raw = core_raw(tmp_path)
    raw["model"].update({"n_states": 3, "engine": "explicit", "d_max": 20, "topology": "left_to_right"})
    raw["model"]["emission"]["params"] = {
        "mean": {"mode": "free", "init": [0.0, 0.2, 0.4]},
        "scale": {"mode": "free", "init": [0.5, 0.7, 0.9]},
    }
    raw["model"]["duration"] = {
        "per_state": [
            {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 5.0}}},
            {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 7.0}}},
            {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 9.0}}},
        ]
    }
    cfg = parse_project_config(raw)
    y = np.random.default_rng(0).normal(size=40)
    times = pd.bdate_range("2020-01-01", periods=40).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    spec_ps = build_model_spec(cfg, data)
    modes = {s.name: s.mode for s in spec_ps.param_specs if s.name.startswith("duration.")}
    assert modes == {"duration.s0.rate": "free", "duration.s1.rate": "free", "duration.s2.rate": "fixed"}
    assert has_identifiability_guard(spec_ps)
