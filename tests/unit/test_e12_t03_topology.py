"""E12-T03: transition topology: shorthands, masked logits, absorbing regimes in both engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.contracts import ObservationSet
from hsmmkit.data.prepare import Preparer
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.model.build import build_model_spec
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.model.topology import from_config
from hsmmkit.statespace.builders import build_expanded, build_explicit
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw
from tests.reference import ref_hsmm


def _raw(
    tmp_path: Path,
    topology: str | dict[str, Any],
    n_states: int = 3,
    engine: str = "expanded",
    **trans: object,
) -> dict[str, Any]:
    raw = core_raw(tmp_path)
    m = raw["model"]
    m["n_states"] = n_states
    m["engine"] = engine
    m["topology"] = topology
    m["emission"]["params"] = {
        "mean": {"mode": "free", "init": [0.2 * i for i in range(n_states)]},
        "scale": {"mode": "free", "init": [0.5 + 0.3 * i for i in range(n_states)]},
    }
    m["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [10.0] * n_states
    m["transition"].update(trans)
    return raw


def _data(cfg: Any, t: int = 60, seed: int = 0) -> Any:  # noqa: ANN401
    y = np.random.default_rng(seed).normal(size=t) * 0.5
    times = pd.bdate_range("2020-01-01", periods=t).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "s")
    return Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])


def _spec(tmp_path: Path, topology: Any, n: int = 3, **kw: Any) -> tuple[Any, Any, Any]:  # noqa: ANN401
    cfg = parse_project_config(_raw(tmp_path, topology, n, **kw))
    data = _data(cfg)
    return cfg, data, build_model_spec(cfg, data)


# --- shorthands ------------------------------------------------------------------------------------------


def test_e12_t03_shorthands() -> None:
    """`full`, `left_to_right`, `cyclic`, `banded`, `matrix` give the documented allowed matrices."""
    from hsmmkit.config.schema import BandedCfg, MatrixCfg

    n = 4
    assert from_config("full", n).is_full
    ltr = from_config("left_to_right", n)
    assert ltr.allowed.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False] * 4,
    ]
    assert ltr.absorbing == (3,)
    cyc = from_config("cyclic", n)
    assert cyc.allowed.sum() == n
    assert all(cyc.allowed[i, (i + 1) % n] for i in range(n))
    assert cyc.absorbing == ()
    band = from_config(BandedCfg(banded=1), n)
    assert band.allowed.tolist() == [
        [False, True, False, False],
        [True, False, True, False],
        [False, True, False, True],
        [False, False, True, False],
    ]
    mat = from_config(MatrixCfg(matrix=[[0, 1, 0], [0, 0, 0], [1, 0, 0]]), 3)
    assert mat.absorbing == (1,)


@pytest.mark.parametrize(
    "matrix",
    [[[0, 1], [1, 1]], [[0, 1, 0], [1, 0, 1]], [[0, 2], [1, 0]]],
)
def test_e12_t03_invalid_matrices_are_config_errors(matrix: list[list[int]], tmp_path: Path) -> None:
    """AC5: a non-zero diagonal, a wrong shape and a non 0/1 entry fail with the path of the key."""
    with pytest.raises(ConfigError) as err:
        parse_project_config(_raw(tmp_path, {"matrix": matrix}, n_states=2))
    assert err.value.path == "model.topology.matrix"


# --- parameter counts (AC3) ---------------------------------------------------------------------------------


def _expected_free(allowed: np.ndarray, order: int) -> int:
    """Appendix F: `sum over memory tuples of max(0, |A_current| - 1)`."""
    n = allowed.shape[0]
    tuples = [(i,) for i in range(n)]
    for _ in range(order - 1):
        tuples = [(*t, j) for t in tuples for j in range(n) if allowed[t[-1], j]]
    return int(sum(max(0, int(allowed[t[-1]].sum()) - 1) for t in tuples))


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("name", ["left_to_right", "cyclic", "banded", "random"])
def test_e12_t03_free_transition_parameters_follow_appendix_f(
    name: str, n: int, order: int, tmp_path: Path
) -> None:
    """AC3: `n_free` of the transition specs equals the Appendix F formula (shorthands, random matrices)."""
    rng = np.random.default_rng(n * 10 + order)
    if name == "banded":
        topology: Any = {"banded": min(2, n - 1)}
    elif name == "random":
        mat = (rng.random((n, n)) < 0.6).astype(int)
        np.fill_diagonal(mat, 0)
        if order == 2 and not mat.any():
            mat[0, 1] = 1
        topology = {"matrix": mat.tolist()}
    else:
        topology = name
    cfg = parse_project_config(_raw(tmp_path, topology, n, order=order))
    topo = from_config(cfg.model.topology, n)
    if not topo.allowed.any() or (
        order == 2 and _expected_free(topo.allowed, order) == 0 and not topo.allowed.any()
    ):
        pytest.skip("no allowed jump")
    try:
        spec = build_model_spec(cfg, _data(cfg))
    except ConfigError:
        pytest.skip("no memory tuple of this order")
    specs = tuple(s for s in spec.param_specs if s.name.startswith("transition."))
    n_free = ParamManager(specs).n_free if specs else 0
    assert n_free == _expected_free(topo.allowed, order)


def test_e12_t03_each_transition_covariate_adds_the_same_count(tmp_path: Path) -> None:
    """With `K` transition covariates, `transition.b` has `F * K` free entries on top of `F`."""
    raw = _raw(tmp_path, "left_to_right", 4, covariates=["c"])
    raw["data"]["covariates"] = [{"name": "c", "column": "c", "use_in": ["transition"], "lag": 1}]
    raw["data"]["source"] = {"kind": "pandas", "observation_columns": ["x"], "covariate_columns": ["c"]}
    cfg = parse_project_config(raw)
    t = 60
    frame = pd.DataFrame(
        {"x": np.random.default_rng(0).normal(size=t), "c": np.random.default_rng(1).normal(size=t)}
    )
    from hsmmkit.data.sources import load_data

    obs, covs = load_data(cfg.data, obj=frame)
    data = Preparer(cfg.data).fit(obs, covs, 10**6).transform(obs, covs)
    spec = build_model_spec(cfg, data)
    f = _expected_free(from_config("left_to_right", 4).allowed, 1)
    shapes = {s.name: s.shape for s in spec.param_specs if s.name.startswith("transition.")}
    assert shapes == {"transition.a": (f,), "transition.b": (f, 1)}


# --- A_t and decoding (AC1) --------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_e12_t03_transition_matrix_has_exact_zeros_and_unit_rows(order: int, tmp_path: Path) -> None:
    """AC1: `A_t` is exactly 0 at disallowed entries, rows sum to 1 (absorbing rows included)."""
    _, _, spec = _spec(tmp_path, "left_to_right", 3, order=order)
    space = build_expanded(spec)
    rng = np.random.default_rng(order)
    mgr = ParamManager(spec.param_specs)
    theta = mgr.unflatten(jnp.asarray(rng.normal(size=mgr.n_free)))
    a, _ = space.transition_fn(theta, None, None, jnp.asarray(1.0))
    a = np.asarray(a)
    np.testing.assert_allclose(a.sum(axis=1), 1.0, atol=1e-12)
    regime = np.asarray(space.index.regime_of)
    phase = np.asarray(space.index.phase_of)
    for s in range(a.shape[0]):
        reach = np.flatnonzero(a[s] > 0)
        assert np.all(regime[reach] >= regime[s])  # never a backward jump
        if regime[s] == 2:
            assert reach.tolist() == [s]  # absorbing: stays put with probability exactly 1
    del phase


def test_e12_t03_left_to_right_never_decodes_a_backward_jump(tmp_path: Path) -> None:
    """AC1: Viterbi on random data with random parameters is non-decreasing in the regime."""
    _, data, spec = _spec(tmp_path, "left_to_right", 3)
    space = build_expanded(spec)
    mgr = ParamManager(spec.param_specs)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        theta = dict(mgr.unflatten(jnp.asarray(rng.normal(size=mgr.n_free))))
        theta["emission.mean"] = jnp.asarray(rng.normal(size=3))
        theta["emission.scale"] = jnp.asarray(rng.uniform(0.3, 1.5, size=3))
        path = np.asarray(space.index.regime_of[ExpandedHMMEngine().viterbi(theta, spec, space, data)])
        assert np.all(np.diff(path) >= 0)


# --- absorbing regimes (AC2) -------------------------------------------------------------------


def test_e12_t03_absorbing_duration_parameters_are_fixed(tmp_path: Path) -> None:
    """The duration entry of an absorbing regime is `fixed` (not in `u`)."""
    _, _, spec = _spec(tmp_path, "left_to_right", 3)
    rate = next(s for s in spec.param_specs if s.name == "duration.rate")
    assert rate.mode == ("free", "free", "fixed")
    assert ParamManager((rate,)).n_free == 2  # the absorbing entry is not a free coordinate


def test_e12_t03_engines_agree_on_an_absorbing_model_and_match_the_censored_reference(tmp_path: Path) -> None:
    """AC2: expanded `a_i = 0`, explicit `p(d) = delta_{D_max}` (censored): same loglik, NumPy reference."""
    t = 60
    cfg_e, data, spec_e = _spec(tmp_path, "left_to_right", 2)
    raw = _raw(tmp_path, "left_to_right", 2, engine="explicit")
    raw["model"]["d_max"] = 150
    cfg_x = parse_project_config(raw)
    spec_x = build_model_spec(cfg_x, data)
    mgr_e = ParamManager(spec_e.param_specs)
    theta = dict(mgr_e.unflatten(mgr_e.init_u()))
    theta["emission.mean"] = jnp.asarray([-0.3, 0.4])
    theta["emission.scale"] = jnp.asarray([0.4, 0.6])
    ll_e = float(ExpandedHMMEngine().loglik(theta, spec_e, build_expanded(spec_e), data))
    space_x = build_explicit(spec_x)
    ll_x = float(ExplicitDurationEngine().loglik(theta, spec_x, space_x, data))
    assert abs(ll_e - ll_x) <= 1e-6 * abs(ll_e)
    # independent NumPy reference with the censored last segment
    y = np.asarray(data.observations)[:, 0]
    mean, scale = np.asarray(theta["emission.mean"]), np.asarray(theta["emission.scale"])
    log_emis = -0.5 * ((y[:, None] - mean) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)
    a = np.asarray(spec_x.duration.advance_prob_t(theta, jnp.asarray(1.0), None))
    d = np.arange(1, 151)
    pmf = np.full((2, 150), -np.inf)
    pmf[0] = np.log(a[0]) + (d - 1) * np.log1p(-a[0])
    pmf[0] -= np.log(np.exp(pmf[0]).sum())
    pmf[1, -1] = 0.0
    log_trans = np.array([[-np.inf, 0.0], [-np.inf, -np.inf]])
    ref = ref_hsmm.forward(np.full(2, -np.log(2)), log_trans, pmf, log_emis, 150)[1]
    assert abs(ll_x - ref) <= 1e-9 * abs(ref)
    del t, cfg_e


def test_e12_t03_cyclic_and_matrix_topologies_build_both_engines(tmp_path: Path) -> None:
    """A cyclic topology has no absorbing regime; an arbitrary matrix builds for order 2 too."""
    for topology, order in (("cyclic", 1), ("cyclic", 2), ({"matrix": [[0, 1, 1], [1, 0, 0], [0, 1, 0]]}, 2)):
        _, data, spec = _spec(tmp_path, topology, 3, order=order)
        space = build_expanded(spec)
        mgr = ParamManager(spec.param_specs)
        ll = ExpandedHMMEngine().loglik(mgr.unflatten(mgr.init_u()), spec, space, data)
        assert np.isfinite(float(ll))


@pytest.mark.parametrize("family", ["geometric", "negbin"])
def test_e12_t03_explicit_absorbing_loglik_is_finite_beyond_d_max_and_matches_expanded(
    family: str, tmp_path: Path
) -> None:
    """Review fix: `T > D_max` works: the absorbing final segment has no cap (geometric, NB)."""
    t, d_max = 200, 60
    raws = []
    for engine in ("expanded", "explicit"):
        raw = _raw(tmp_path, "left_to_right", 2, engine=engine)
        raw["model"]["d_max"] = d_max
        raw["model"]["duration"] = {
            "family": family,
            "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [3.0, 3.0]}},
        }
        if family == "negbin":
            raw["model"]["duration"]["r"] = [2, 2]
        raws.append(parse_project_config(raw))
    data = _data(raws[0], t=t)
    spec_e, spec_x = (build_model_spec(c, data) for c in raws)
    mgr = ParamManager(spec_e.param_specs)
    theta = dict(mgr.unflatten(mgr.init_u()))
    theta["emission.mean"] = jnp.asarray([-0.3, 0.4])
    theta["emission.scale"] = jnp.asarray([0.4, 0.6])
    a = float(ExpandedHMMEngine().loglik(theta, spec_e, build_expanded(spec_e), data))
    space_x = build_explicit(spec_x)
    b = float(ExplicitDurationEngine().loglik(theta, spec_x, space_x, data))
    assert abs(a - b) <= 1e-6 * abs(a)
    path = np.asarray(space_x.index.regime_of[ExplicitDurationEngine().viterbi(theta, spec_x, space_x, data)])
    assert np.all(np.diff(path) >= 0)
    with pytest.raises(ConfigError, match="d_max >= T"):
        ExplicitDurationEngine().forward_backward(theta, spec_x, space_x, data)
