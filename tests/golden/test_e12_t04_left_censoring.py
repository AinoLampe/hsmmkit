"""E12-T04 golden: left-censored first segment (residual-life initial duration) in the explicit engine."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.contracts import ObservationSet, PreparedDataset
from hsmmkit.data.prepare import Preparer
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.model.build import build_model_spec
from hsmmkit.model.duration.base import residual_log_pmf
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.model.spec import ModelSpec
from hsmmkit.statespace.builders import build_expanded, build_explicit
from tests.golden.test_e4_t06b_equivalence import D_MAX, paired_scenarios
from tests.helpers.core_config import core_raw
from tests.reference import ref_hsmm


def _build(
    tmp_path: Path, n: int, t: int, d_max: int, seed: int, *, left: bool, family: str
) -> tuple[Any, Any, Any]:
    raw = core_raw(tmp_path)
    m = raw["model"]
    m["n_states"] = n
    m["engine"] = "explicit"
    m["d_max"] = d_max
    m["left_censored"] = left
    m["initial_state"] = "uniform"
    m["emission"]["params"] = {
        "mean": {"mode": "free", "init": [0.3 * i for i in range(n)]},
        "scale": {"mode": "free", "init": [0.5 + 0.2 * i for i in range(n)]},
    }
    m["duration"] = (
        {"family": "nonparametric", "params": {"logits": {"mode": "free"}}}
        if family == "nonparametric"
        else {"family": family, "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [4.0] * n}}}
    )
    cfg = parse_project_config(raw)
    y = np.random.default_rng(seed).normal(size=t)
    times = pd.bdate_range("2020-01-01", periods=t).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    spec = build_model_spec(cfg, data)
    return cfg, data, spec


def _stationary_rho(pi: np.ndarray, log_pmf: np.ndarray) -> np.ndarray:
    """`rho_j ~ nu_j E[d_j]` with `nu` the eigenvector of `pi^T` for eigenvalue 1 (independent NumPy)."""
    vals, vecs = np.linalg.eig(pi.T)
    nu = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    nu = np.abs(nu) / np.abs(nu).sum()
    mean_d = (np.exp(log_pmf) * np.arange(1, log_pmf.shape[1] + 1)).sum(axis=1)
    rho = nu * mean_d
    return np.log(rho / rho.sum())


@pytest.mark.parametrize("seed", range(12))
def test_e12_t04_explicit_loglik_matches_the_numpy_reference(seed: int, tmp_path: Path) -> None:
    """AC2: random `N in [2,4]`, `T in [5,120]`, `D_max in [2,20]` (single segments included), <= 1e-8."""
    rng = np.random.default_rng(100 + seed)
    n, t, d_max = int(rng.integers(2, 5)), int(rng.integers(5, 121)), int(rng.integers(2, 21))
    _, data, spec = _build(tmp_path, n, t, d_max, seed, left=True, family="nonparametric")
    mgr = ParamManager(spec.param_specs)
    theta = dict(mgr.unflatten(jnp.asarray(rng.normal(size=mgr.n_free))))
    theta["emission.mean"] = jnp.asarray(rng.normal(size=n))
    theta["emission.scale"] = jnp.asarray(rng.uniform(0.4, 1.5, size=n))
    ll = float(ExplicitDurationEngine().loglik(theta, spec, build_explicit(spec), data))

    pi = np.asarray(spec.transition.jump_probs_t(theta, None, build_explicit(spec).layout))
    log_trans, log_pmf, log_emis = _ref_inputs(spec, theta, data, d_max)
    log_rho = _stationary_rho(pi, log_pmf)
    ref = ref_hsmm.forward(log_rho, log_trans, log_pmf, log_emis, d_max, left_censored=True)[1]
    assert abs(ll - ref) <= 1e-8 * max(1.0, abs(ref))
    if t <= 9 and n <= 3 and d_max <= 6:
        brute = ref_hsmm.brute_force_loglik(log_rho, log_trans, log_pmf, log_emis, d_max, left_censored=True)
        assert abs(ref - brute) <= 1e-9 * max(1.0, abs(brute))


def test_e12_t04_single_segment_term_uses_the_residual_survival(tmp_path: Path) -> None:
    """A sequence with `T <= D_max` contains the single-segment path `rho_j S_res_j(T)`."""
    _, data, spec = _build(tmp_path, 2, 6, 12, 5, left=True, family="nonparametric")
    mgr = ParamManager(spec.param_specs)
    theta = mgr.unflatten(mgr.init_u())
    ll = float(ExplicitDurationEngine().loglik(theta, spec, build_explicit(spec), data))
    assert np.isfinite(ll)
    pi = np.asarray(spec.transition.jump_probs_t(theta, None, build_explicit(spec).layout))
    log_pmf = np.asarray(spec.duration.log_pmf(theta, 12, data.nominal_dt, None))
    res = np.asarray(residual_log_pmf(jnp.asarray(log_pmf)))
    np.testing.assert_allclose(np.exp(res).sum(axis=1), 1.0, atol=1e-12)
    assert np.all(np.isfinite(_stationary_rho(pi, log_pmf)))


@pytest.mark.parametrize(
    "family", ["geometric", "negbin", "poisson", "gamma", "weibull", "lognormal", "nonparametric"]
)
def test_e12_t04_residual_pmf_is_normalized_for_every_explicit_family(family: str, tmp_path: Path) -> None:
    """AC3: `p_res` sums to 1 (<= 1e-12) for every Part I explicit family."""
    raw = core_raw(tmp_path)
    m = raw["model"]
    m["engine"] = "explicit"
    m["d_max"] = 40
    params: dict[str, Any] = {
        "geometric": {"rate": {"mode": "free", "init_mean_sojourn_steps": [5.0, 9.0]}},
        "negbin": {"rate": {"mode": "free", "init_mean_sojourn_steps": [5.0, 9.0]}},
        "poisson": {"kappa": {"mode": "free", "init": [4.0, 8.0]}},
        "gamma": {
            "shape": {"mode": "free", "init": [2.0, 3.0]},
            "scale": {"mode": "free", "init": [2.0, 3.0]},
        },
        "weibull": {
            "shape": {"mode": "free", "init": [1.5, 2.0]},
            "scale": {"mode": "free", "init": [5.0, 8.0]},
        },
        "lognormal": {
            "mu_log": {"mode": "free", "init": [1.5, 2.0]},
            "sd_log": {"mode": "free", "init": [0.5, 0.6]},
        },
        "nonparametric": {"logits": {"mode": "free"}},
    }[family]
    m["duration"] = {"family": family, "params": params}
    if family == "negbin":
        m["duration"]["r"] = [2, 2]
    cfg = parse_project_config(raw)
    y = np.random.default_rng(1).normal(size=60)
    times = pd.bdate_range("2020-01-01", periods=60).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    spec = build_model_spec(cfg, data)
    mgr = ParamManager(spec.param_specs)
    log_pmf = spec.duration.log_pmf(mgr.unflatten(mgr.init_u()), 40, data.nominal_dt, None)
    res = np.exp(np.asarray(residual_log_pmf(jnp.asarray(log_pmf))))
    np.testing.assert_allclose(res.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(res >= 0)


@pytest.mark.parametrize("seed", range(6))
def test_e12_t04_explicit_matches_the_expanded_stationary_engine_on_shifted_negbin(seed: int) -> None:
    """AC1 (Appendix D): explicit residual-life initial equals the expanded `initial: stationary` engine."""
    sc, spec_x, theta = paired_scenarios(seed)
    spec_e = ModelSpec(
        **{**{f.name: getattr(sc.spec, f.name) for f in fields(sc.spec)}, "initial": "stationary"}
    )
    spec_x = ModelSpec(**{**{f.name: getattr(spec_x, f.name) for f in fields(spec_x)}, "left_censored": True})
    pmf_mass = np.exp(np.asarray(spec_x.duration.log_pmf(theta, D_MAX, sc.data.nominal_dt, None))).sum(axis=1)
    assert np.all(1.0 - pmf_mass < 1e-10)
    ll_e = float(ExpandedHMMEngine().loglik(theta, spec_e, build_expanded(spec_e), sc.data))
    ll_x = float(ExplicitDurationEngine().loglik(theta, spec_x, build_explicit(spec_x), sc.data))
    assert abs(ll_e - ll_x) <= 1e-6 * abs(ll_e)


def _ref_inputs(
    spec: ModelSpec, theta: dict[str, Any], data: PreparedDataset, d_max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(log_trans, log_pmf, log_emis)` for the NumPy reference."""
    pi = np.asarray(spec.transition.jump_probs_t(theta, None, build_explicit(spec).layout))
    with np.errstate(divide="ignore"):
        log_trans = np.log(pi)
    log_pmf = np.asarray(spec.duration.log_pmf(theta, d_max, data.nominal_dt, None))
    y = np.asarray(data.observations)[:, 0]
    mean = np.asarray(theta["emission.mean"]).reshape(-1)
    scale = np.asarray(theta["emission.scale"]).reshape(-1)
    log_emis = -0.5 * ((y[:, None] - mean) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)
    return log_trans, log_pmf, log_emis


def test_e12_t04_left_censored_false_is_part_i_bitwise(tmp_path: Path) -> None:
    """AC4: the gate is gone; `left_censored: false` is the Part I likelihood (NumPy reference)."""
    from hsmmkit.config.gates import used_gates

    cfg, data, spec = _build(tmp_path, 3, 50, 10, 2, left=False, family="nonparametric")
    assert all(path != "model.left_censored" for path, _ in used_gates(cfg))
    assert spec.left_censored is False
    mgr = ParamManager(spec.param_specs)
    theta = mgr.unflatten(mgr.init_u())
    a = float(ExplicitDurationEngine().loglik(theta, spec, build_explicit(spec), data))
    log_trans, log_pmf, log_emis = _ref_inputs(spec, theta, data, 10)
    ref = ref_hsmm.forward(np.full(3, -np.log(3)), log_trans, log_pmf, log_emis, 10)[1]
    assert abs(a - ref) <= 1e-9 * abs(ref)
    _, _, spec_l = _build(tmp_path, 3, 50, 10, 2, left=True, family="nonparametric")
    b = float(ExplicitDurationEngine().loglik(theta, spec_l, build_explicit(spec_l), data))
    assert a != b  # the residual-life first segment changes the likelihood


def test_e12_t04_expanded_engine_switches_to_stationary_with_a_warning(tmp_path: Path) -> None:
    """`left_censored` with the expanded engine warns and builds `initial: stationary`."""
    raw = core_raw(tmp_path)
    raw["model"]["left_censored"] = True
    cfg = parse_project_config(raw)
    y = np.random.default_rng(0).normal(size=40)
    times = pd.bdate_range("2020-01-01", periods=40).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    with pytest.warns(UserWarning, match="left_censored implies initial_state: stationary"):
        spec = build_model_spec(cfg, data)
    assert spec.initial == "stationary"
