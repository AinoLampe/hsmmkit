"""E3-T06: lattice generator, ordering wiring, post-sorting and the identifiability guard."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.config.schema import EmissionCfg, LatticeCfg, ParamCfg
from regimekit.config.yaml_io import load_config
from regimekit.finance import FinanceConfig
from regimekit.model.duration.geometric import GeometricDuration
from regimekit.model.emission.gaussian import GaussianFamily
from regimekit.model.lattice import (
    apply_ordering,
    build,
    has_identifiability_guard,
    post_sort_states,
    validate_lattice,
)
from regimekit.model.params import ParamManager
from regimekit.model.spec import ModelSpec
from regimekit.model.transition import JumpTransition, build_memory_layout
from regimekit.utils.errors import ConfigError

APPENDIX_A = Path(__file__).parent.parent / "fixtures" / "configs" / "appendix_a.yaml"


# --- AC1: 3 x 2 grid -> 6 fixed states with the expected value table -----------------------------


def test_e3_t06_grid_product_yields_expected_value_table() -> None:
    """`mu x sigma` = 3 x 2 -> 6 states, `mu` slowest, exactly the Cartesian product."""
    lattice = build({"mean": [-0.3, 0.0, 0.2], "scale": [0.15, 0.35]})
    assert lattice.n_states == 6
    np.testing.assert_array_equal(lattice.values["mean"], [-0.3, -0.3, 0.0, 0.0, 0.2, 0.2])
    np.testing.assert_array_equal(lattice.values["scale"], [0.15, 0.35, 0.15, 0.35, 0.15, 0.35])
    pairs = set(zip(lattice.values["mean"], lattice.values["scale"], strict=True))
    assert pairs == set(itertools.product([-0.3, 0.0, 0.2], [0.15, 0.35]))


def test_e3_t06_to_param_cfgs_are_fixed_and_drive_gbm_specs() -> None:
    """`to_param_cfgs` gives `mode="fixed"` configs whose values survive `GaussianFamily.from_config`."""
    lattice = build({"mean": [-0.3, 0.0, 0.2], "scale": [0.15, 0.35]})
    cfgs = lattice.to_param_cfgs()
    assert {k: c.mode for k, c in cfgs.items()} == {"mean": "fixed", "scale": "fixed"}
    emission = EmissionCfg(family="gaussian", scale_by_dt=True, ito_correction=True, params=cfgs)
    _, specs = GaussianFamily.from_config(emission, 6, 1, ())
    by_name = {s.name: s for s in specs}
    np.testing.assert_array_equal(by_name["emission.mean"].value[:, 0], lattice.values["mean"])
    np.testing.assert_array_equal(by_name["emission.scale"].value[:, 0], lattice.values["scale"])
    assert ParamManager(specs).n_free == 0


@pytest.mark.parametrize(
    ("grids", "needle"),
    [
        ({}, "at least one"),
        ({"mean": []}, "empty"),
        ({"mean": [0.1, 0.1]}, "duplicate"),
        ({"mean": [float("nan")]}, "finite"),
    ],
)
def test_e3_t06_bad_grids_raise_with_path(grids: dict[str, list[float]], needle: str) -> None:
    """Empty, duplicated or non-finite grids are `ConfigError`s carrying `model.lattice.grids`."""
    with pytest.raises(ConfigError, match=needle) as info:
        build(grids)
    assert info.value.path is not None
    assert info.value.path.startswith("model.lattice.grids")


def test_e3_t06_validate_lattice_checks_n_states_equals_product() -> None:
    """`validate_lattice` passes for a matching product, raises for a mismatch, ignores no-lattice configs."""
    cfg = load_config(APPENDIX_A, FinanceConfig)
    validate_lattice(cfg)  # Appendix A has no lattice
    good = cfg.model_copy(
        update={
            "model": cfg.model.model_copy(
                update={
                    "n_states": 6,
                    "lattice": LatticeCfg(grids={"mean": [-0.3, 0.0, 0.2], "scale": [0.15, 0.35]}),
                }
            )
        }
    )
    validate_lattice(good)
    bad = good.model_copy(update={"model": good.model.model_copy(update={"n_states": 5})})
    with pytest.raises(ConfigError, match="6 states") as info:
        validate_lattice(bad)
    assert info.value.path == "model.lattice.grids"


# --- AC2: `increasing` yields strictly increasing sigma through ParamManager ----------------------


def test_e3_t06_increasing_constraint_gives_increasing_sigma_for_random_u() -> None:
    """A GBM spec built with `constraint: increasing` maps any `u` to strictly increasing `sigma`."""
    emission = EmissionCfg(
        family="gaussian",
        scale_by_dt=True,
        ito_correction=True,
        params={
            "mean": ParamCfg(mode="free"),
            "scale": ParamCfg(mode="free", constraint="increasing"),
        },
    )
    _, specs = GaussianFamily.from_config(emission, 4, 1, ())
    manager = ParamManager(specs)
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = jnp.asarray(rng.normal(scale=3.0, size=manager.n_free))
        sigma = np.asarray(manager.unflatten(u)["emission.scale"]).ravel()
        assert np.all(np.diff(sigma) > 0)


def test_e3_t06_apply_ordering_wires_increasing_and_squeezes_single_series() -> None:
    """`apply_ordering` on the default `(N, 1)` sigma spec gives an `(N,)` increasing spec."""
    specs = GaussianFamily(scale_by_dt=True, ito_correction=True).param_specs(3, 1, ())
    ordered = apply_ordering(specs, "emission.scale")
    by_name = {s.name: s for s in ordered}
    assert by_name["emission.scale"].constraint == "increasing"
    assert by_name["emission.scale"].shape == (3,)
    assert by_name["emission.mean"].constraint == "none"
    manager = ParamManager(ordered)
    rng = np.random.default_rng(1)
    for _ in range(20):
        sigma = np.asarray(
            manager.unflatten(jnp.asarray(rng.normal(scale=2.0, size=manager.n_free)))["emission.scale"]
        )
        assert np.all(np.diff(sigma) > 0)


def test_e3_t06_apply_ordering_rejects_multi_series_and_unknown_names() -> None:
    """Ordering across `M > 1` series and unknown spec names are `ConfigError`s with the spec name."""
    with pytest.raises(ConfigError, match="single series") as info:
        apply_ordering(
            GaussianFamily(scale_by_dt=True, ito_correction=True).param_specs(3, 2, ()), "emission.scale"
        )
    assert info.value.path == "emission.scale"
    with pytest.raises(ConfigError, match="no parameter spec"):
        apply_ordering(
            GaussianFamily(scale_by_dt=True, ito_correction=True).param_specs(3, 1, ()), "emission.nope"
        )


# --- identifiability guard --------------------------------------------------------------------------


def _model_spec(
    specs: tuple, *, n_states: int = 3, order: int = 1, conditioning: str = "current"
) -> ModelSpec:
    return ModelSpec(
        n_states=n_states,
        n_series=1,
        emission=GaussianFamily(scale_by_dt=True, ito_correction=True),
        duration=GeometricDuration(),
        transition=JumpTransition(),
        order=order,
        duration_conditioning=conditioning,  # type: ignore[arg-type]
        engine="expanded",
        d_max=10,
        initial="uniform",
        param_specs=specs,
    )


def test_e3_t06_identifiability_guard_cases() -> None:
    """Guard is on for `increasing`, and for an all-fixed lattice; off for plain free specs."""
    free = GaussianFamily(scale_by_dt=True, ito_correction=True).param_specs(3, 1, ())
    assert not has_identifiability_guard(_model_spec(free))
    assert has_identifiability_guard(_model_spec(apply_ordering(free, "emission.scale")))

    lattice = build({"mean": [-0.3, 0.0, 0.2], "scale": [0.15]})
    emission = EmissionCfg(
        family="gaussian", scale_by_dt=True, ito_correction=True, params=lattice.to_param_cfgs()
    )
    _, fixed_specs = GaussianFamily.from_config(emission, 3, 1, ())
    assert has_identifiability_guard(_model_spec(fixed_specs))

    mixed = tuple(
        replace(s, mode="fixed", value=np.full(s.shape, 0.1)) if s.name == "emission.mean" else s
        for s in free
    )
    assert not has_identifiability_guard(_model_spec(mixed))  # sigma still free
    assert not has_identifiability_guard(_model_spec(()))  # nothing to identify with


# --- AC3: post_sort_states -------------------------------------------------------------------------


def _row_lookup(n_states: int, order: int) -> tuple[np.ndarray, dict[tuple[int, ...], int]]:
    layout = build_memory_layout(n_states, order)
    return layout.tuples, {tuple(int(x) for x in t): r for r, t in enumerate(layout.tuples)}


def _random_theta(
    rng: np.random.Generator, n_states: int, order: int, *, covariates: bool, pair: bool
) -> dict[str, jnp.ndarray]:
    n_rows = build_memory_layout(n_states, order).tuples.shape[0]
    theta: dict[str, jnp.ndarray] = {
        "emission.mean": jnp.asarray(rng.normal(size=(n_states, 1))),
        "emission.scale": jnp.asarray(rng.permutation(np.linspace(0.1, 0.5, n_states))[:, None]),
        "duration.rate": jnp.asarray(
            rng.uniform(0.05, 0.5, size=(n_states * (n_states - 1) if pair else n_states,))
        ),
    }
    if covariates:
        theta["transition.a"] = jnp.asarray(rng.normal(size=(n_rows, n_states - 2)))
        theta["transition.b"] = jnp.asarray(rng.normal(size=(n_rows, n_states - 2, 2)))
    else:
        theta["transition.pi_rows"] = jnp.asarray(rng.dirichlet(np.ones(n_states - 1), size=n_rows))
    return theta


@pytest.mark.parametrize(
    ("n_states", "order", "covariates", "pair"),
    [
        (3, 1, False, False),
        (3, 1, True, False),
        (4, 1, True, False),
        (3, 2, False, False),
        (3, 2, True, True),
        (4, 2, True, False),
    ],
)
def test_e3_t06_post_sort_sorts_sigma_and_permutes_blocks_consistently(
    n_states: int, order: int, covariates: bool, pair: bool
) -> None:
    """Shuffled sigma comes back sorted; mu/rate follow it; `jump_probs` is invariant under relabelling."""
    rng = np.random.default_rng(n_states * 10 + order)
    theta = _random_theta(rng, n_states, order, covariates=covariates, pair=pair)
    family = JumpTransition(cov_idx=(0, 1) if covariates else ())
    spec = _model_spec((), n_states=n_states, order=order, conditioning="pair" if pair else "current")

    sorted_theta, perm = post_sort_states(theta, spec, by="emission.scale")

    old_sigma = np.asarray(theta["emission.scale"])[:, 0]
    new_sigma = np.asarray(sorted_theta["emission.scale"])[:, 0]
    assert np.all(np.diff(new_sigma) > 0)
    np.testing.assert_array_equal(new_sigma, old_sigma[perm])
    np.testing.assert_array_equal(
        np.asarray(sorted_theta["emission.mean"]), np.asarray(theta["emission.mean"])[perm]
    )
    assert sorted(perm.tolist()) == list(range(n_states))

    # duration rates
    old_rate, new_rate = np.asarray(theta["duration.rate"]), np.asarray(sorted_theta["duration.rate"])
    if pair:
        pairs = [(i, j) for i in range(n_states) for j in range(n_states) if i != j]
        idx = {p: k for k, p in enumerate(pairs)}
        for (a, b), k in idx.items():
            assert new_rate[k] == old_rate[idx[(perm[a], perm[b])]]
    else:
        np.testing.assert_array_equal(new_rate, old_rate[perm])

    # transition invariance: P_new[t, row(u), j'] == P_old[t, row(perm[u]), perm[j']]
    memory_layout = build_memory_layout(n_states, order)
    tuples, row_of = _row_lookup(n_states, order)
    z = jnp.asarray(rng.normal(size=(5, 2))) if covariates else None
    p_old = np.asarray(family.jump_probs(theta, z, memory_layout))
    p_new = np.asarray(family.jump_probs(sorted_theta, z, memory_layout))
    for r_new, tup in enumerate(tuples):
        r_old = row_of[tuple(int(perm[x]) for x in tup)]
        np.testing.assert_allclose(p_new[:, r_new, :], p_old[:, r_old, :][:, perm], atol=1e-12)
    np.testing.assert_allclose(p_new.sum(axis=-1), 1.0, atol=1e-12)


def test_e3_t06_post_sort_of_sorted_theta_is_identity() -> None:
    """Already-ascending sigma gives the identity permutation and unchanged arrays."""
    rng = np.random.default_rng(0)
    theta = _random_theta(rng, 3, 1, covariates=True, pair=False)
    theta["emission.scale"] = jnp.asarray([[0.1], [0.2], [0.3]])
    spec = _model_spec((), n_states=3)
    sorted_theta, perm = post_sort_states(theta, spec)
    np.testing.assert_array_equal(perm, [0, 1, 2])
    for name, value in theta.items():
        np.testing.assert_allclose(np.asarray(sorted_theta[name]), np.asarray(value), atol=1e-15)


def test_e3_t06_post_sort_rejects_unknown_key_and_unknown_blocks() -> None:
    """Sorting by an absent key, or with an unrecognised block present, raises `ConfigError`."""
    theta = _random_theta(np.random.default_rng(0), 3, 1, covariates=False, pair=False)
    spec = _model_spec((), n_states=3)
    with pytest.raises(ConfigError, match="not in theta"):
        post_sort_states(theta, spec, by="emission.nope")
    with pytest.raises(ConfigError, match="relabel"):
        post_sort_states({**theta, "mystery.block": jnp.zeros(3)}, spec)


def test_e3_t06_post_sort_permutes_state_axis_blocks_including_nu_and_coefficients() -> None:
    """Every `(N, ...)` emission/duration block (nu, gamma, c, initial logits) is gathered along states."""
    rng = np.random.default_rng(3)
    theta = _random_theta(rng, 3, 1, covariates=False, pair=False)
    theta.update(
        {
            "emission.nu": jnp.asarray([5.0, 7.0, 9.0]),
            "emission.gamma": jnp.asarray(rng.normal(size=(3, 2))),
            "duration.c": jnp.asarray(rng.normal(size=(3, 2))),
            "initial.logits": jnp.asarray([0.2, 0.3, 0.5]),
        }
    )
    sorted_theta, perm = post_sort_states(theta, _model_spec((), n_states=3))
    for name in ("emission.nu", "emission.gamma", "duration.c", "initial.logits"):
        np.testing.assert_array_equal(np.asarray(sorted_theta[name]), np.asarray(theta[name])[perm])
    with pytest.raises(ConfigError, match="expanded states"):
        post_sort_states({**theta, "initial.logits": jnp.ones(5)}, _model_spec((), n_states=3))


def test_e3_t06_pair_rates_match_state_space_rate_index_under_relabelling() -> None:
    """Per expanded state, the effective pair rate follows relabelling (checked via `StateSpace`)."""
    from regimekit.statespace.builders import build_expanded

    n_states = 3
    spec = _model_spec((), n_states=n_states, order=2, conditioning="pair")
    theta = _random_theta(np.random.default_rng(11), n_states, 2, covariates=False, pair=True)
    sorted_theta, perm = post_sort_states(theta, spec)
    space = build_expanded(spec)
    idx = space.index
    mem = np.asarray(idx.memory_of)
    reg = np.asarray(idx.regime_of)
    ph = np.asarray(idx.phase_of)
    rate_of = np.asarray(space.rate_index_of)
    old = np.asarray(theta["duration.rate"])[rate_of]
    new = np.asarray(sorted_theta["duration.rate"])[rate_of]
    key_of = {(tuple(mem[s]), int(reg[s]), int(ph[s])): s for s in range(reg.shape[0])}
    for (m, i, p), s_new in key_of.items():
        s_old = key_of[(tuple(int(perm[x]) for x in m), int(perm[i]), p)]
        assert new[s_new] == old[s_old]


def test_e3_t06_guard_ignores_free_nu_and_coefficients_when_location_scale_fixed() -> None:
    """Documented deviation: fixed mu/sigma with free `nu`/`gamma` still counts as identified."""
    specs = tuple(
        replace(s, mode="fixed", value=np.full(s.shape, 0.2))
        if s.name in ("emission.mean", "emission.scale")
        else s
        for s in GaussianFamily(scale_by_dt=True, ito_correction=True).param_specs(3, 1, ())
    )
    from regimekit.model.params.spec import ParamSpec

    extra = ParamSpec(name="emission.nu", shape=(3,), domain="nu_gt_2", mode="free", init=np.full(3, 8.0))
    assert has_identifiability_guard(_model_spec((*specs, extra)))
