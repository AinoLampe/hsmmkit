"""E11-T08: product emission of blocks: config, masking, per-series-product regression, supports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.contracts import ObservationSet
from hsmmkit.data.prepare import Preparer
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.model.build import build_model_spec
from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.emission.poisson import PoissonFamily
from hsmmkit.model.emission.product import BlockProductEmission, ProductEmission
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.statespace.builders import build_expanded
from hsmmkit.utils.errors import ConfigError, DataError
from tests.helpers.core_config import core_raw


def _raw(tmp_path: Path, emission: dict[str, Any]) -> dict[str, Any]:
    raw = core_raw(tmp_path)
    raw["data"]["source"]["observation_columns"] = ["a", "b", "c"]
    raw["model"]["emission"] = emission
    return raw


def _mixed_emission() -> dict[str, Any]:
    return {
        "family": "product",
        "params": {},
        "blocks": [
            {
                "family": "gaussian",
                "columns": [0, 1],
                "covariance": "full",
                "params": {"mean": {"mode": "free", "init": [0.0, 0.5]}, "chol": {"mode": "free"}},
            },
            {
                "family": "poisson",
                "columns": [2],
                "scale_by_dt": False,
                "params": {"rate": {"mode": "free", "init": [2.0, 6.0]}},
            },
        ],
    }


def _data(cfg: Any, y: np.ndarray, series_support: tuple[str, ...] | None = None) -> Any:  # noqa: ANN401
    times = np.arange(y.shape[0]).astype("datetime64[D]").astype("datetime64[ns]")
    obs = ObservationSet(("a", "b", "c"), times, y, "memory", "s")
    prep = Preparer(cfg.data, series_support=series_support)  # type: ignore[arg-type]
    return prep.fit(obs, [], train_end=times[-1]).transform(obs, [])


def _y(t: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(size=t), rng.normal(size=t), rng.poisson(4.0, size=t).astype(float)])


def test_e11_t08_config_builds_prefixed_specs_and_per_series_supports(tmp_path: Path) -> None:
    """A full Gaussian block and a Poisson block: `emission.b{b}.<name>` specs, per-column supports."""
    cfg = parse_project_config(_raw(tmp_path, _mixed_emission()))
    data = _data(cfg, _y(), ("real", "real", "count"))
    assert data.series_support == ("real", "real", "count")
    assert data.support == "count"
    assert data.observations.dtype == jnp.float64
    spec = build_model_spec(cfg, data)
    assert isinstance(spec.emission, BlockProductEmission)
    assert spec.emission.series_support == ("real", "real", "count")
    names = [s.name for s in spec.param_specs if s.name.startswith("emission")]
    assert names == ["emission.b0.mean", "emission.b0.chol", "emission.b1.eta"]
    ll = float(
        ExpandedHMMEngine().loglik(
            ParamManager(spec.param_specs).unflatten(ParamManager(spec.param_specs).init_u()),
            spec,
            build_expanded(spec),
            data,
        )
    )
    assert np.isfinite(ll)


def test_e11_t08_columns_must_partition_the_series(tmp_path: Path) -> None:
    """Missing, repeated, out-of-range and named columns are `ConfigError`s, so is stray `blocks`."""
    data = _data(parse_project_config(_raw(tmp_path, _mixed_emission())), _y(), ("real", "real", "count"))
    for cols in ([0, 1], [0, 1, 1], [0, 1, 5]):
        em = _mixed_emission()
        em["blocks"][1]["columns"] = cols[2:] or [1]
        em["blocks"][0]["columns"] = cols[:2] if len(cols) > 2 else cols
        if cols == [0, 1]:
            em["blocks"][1]["columns"] = [0]
        with pytest.raises(ConfigError) as err:
            build_model_spec(parse_project_config(_raw(tmp_path, em)), data)
        assert err.value.path.startswith("model.emission.blocks")
    em = _mixed_emission()
    em["blocks"][1]["columns"] = ["c"]
    with pytest.raises(ConfigError, match="names are not supported"):
        build_model_spec(parse_project_config(_raw(tmp_path, em)), data)
    raw = _raw(tmp_path, {"family": "gaussian", "params": {}, "blocks": _mixed_emission()["blocks"]})
    with pytest.raises(ConfigError) as err2:
        build_model_spec(parse_project_config(raw), data)
    assert err2.value.path == "model.emission.blocks"


def test_e11_t08_out_of_support_poisson_column_is_named(tmp_path: Path) -> None:
    """AC3: a negative or fractional value in the Poisson column is rejected, naming the column."""
    cfg = parse_project_config(_raw(tmp_path, _mixed_emission()))
    y = _y()
    y[7, 2] = -1.0
    with pytest.raises(DataError, match="series c"):
        _data(cfg, y, ("real", "real", "count"))
    y = _y()
    y[3, 2] = 1.5
    with pytest.raises(DataError, match="series c"):
        _data(cfg, y, ("real", "real", "count"))
    y = _y()
    y[3, 0] = -1.5  # a real column accepts anything finite
    assert _data(cfg, y, ("real", "real", "count")).observations.shape == (60, 3)


def test_e11_t08_masking_joint_block_drops_the_row_independent_blocks_keep_the_rest() -> None:
    """AC4: a missing series masks its whole full-covariance block row; independent blocks keep the rest."""
    full = GaussianFamily(covariance="full")
    pois = PoissonFamily(scale_by_dt=False)
    indep = ProductEmission.of(GaussianFamily(), 2)
    prod = BlockProductEmission.of((full, pois), ((0, 1), (2,)))
    rng = np.random.default_rng(1)
    theta = {
        "emission.b0.mean": jnp.asarray(rng.normal(size=(2, 2))),
        "emission.b0.chol": jnp.asarray([[1.0, 0.2, 0.8], [0.7, -0.1, 1.2]]),
        "emission.b1.eta": jnp.asarray([[0.5], [1.5]]),
    }
    y = jnp.asarray([[0.3, -0.2, 2.0], [0.1, 0.4, 3.0]])
    dt = jnp.ones(2)
    mask = jnp.asarray([[True, False, True], [True, True, True]])
    got = np.asarray(prod.log_prob_masked(y, mask, dt, theta, None))
    block0 = np.asarray(
        full.log_prob(
            y[:, :2],
            dt,
            {"emission.mean": theta["emission.b0.mean"], "emission.chol": theta["emission.b0.chol"]},
            None,
        )
    )
    block1 = np.asarray(pois.log_prob(y[:, 2:], dt, {"emission.eta": theta["emission.b1.eta"]}, None))
    np.testing.assert_allclose(got[0], block1[0], atol=1e-14)  # block 0 dropped for row 0
    np.testing.assert_allclose(got[1], block0[1] + block1[1], atol=1e-14)
    # independent blocks keep the other series' contribution
    prod2 = BlockProductEmission.of((indep, pois), ((0, 1), (2,)))
    th2 = {
        "emission.b0.mean": theta["emission.b0.mean"],
        "emission.b0.scale": jnp.ones((2, 2)),
        "emission.b1.eta": theta["emission.b1.eta"],
    }
    got2 = np.asarray(prod2.log_prob_masked(y, mask, dt, th2, None))
    expect = stats.norm.logpdf(0.3, np.asarray(th2["emission.b0.mean"])[:, 0], 1.0) + block1[0]
    np.testing.assert_allclose(got2[0], expect, atol=1e-12)


def test_e11_t08_independent_blocks_reproduce_the_per_series_product_bitwise(tmp_path: Path) -> None:
    """AC2: per-series blocks (GBM-style options) equal the E9-T01 `ProductEmission` bitwise (masks too)."""
    base = GaussianFamily(scale_by_dt=True, ito_correction=True)
    old = ProductEmission.of(base, 2)
    new = BlockProductEmission.of((base, base), ((0,), (1,)))
    rng = np.random.default_rng(2)
    mean, scale = rng.normal(0, 0.05, (3, 2)), rng.uniform(0.1, 0.3, (3, 2))
    th_old = {"emission.mean": jnp.asarray(mean), "emission.scale": jnp.asarray(scale)}
    th_new = {
        "emission.b0.mean": jnp.asarray(mean[:, :1]),
        "emission.b0.scale": jnp.asarray(scale[:, :1]),
        "emission.b1.mean": jnp.asarray(mean[:, 1:]),
        "emission.b1.scale": jnp.asarray(scale[:, 1:]),
    }
    y = jnp.asarray(rng.normal(0, 0.1, (50, 2)))
    dt = jnp.asarray(rng.uniform(0.002, 0.02, 50))
    mask = jnp.asarray(rng.random((50, 2)) > 0.2)
    a = np.asarray(old.log_prob_masked(y, mask, dt, th_old, None))
    b = np.asarray(new.log_prob_masked(y, mask, dt, th_new, None))
    assert np.array_equal(a, b)
    assert np.array_equal(
        np.asarray(old.log_prob(y, dt, th_old, None)), np.asarray(new.log_prob(y, dt, th_new, None))
    )


def test_e11_t08_m_step_partial_has_holes_for_blocks_without_a_closed_form(tmp_path: Path) -> None:
    """A diagonal Gaussian block has no M-step: `m_step` is `None`, `m_step_partial` keeps the others."""
    diag, pois = GaussianFamily(), PoissonFamily(scale_by_dt=False)
    prod = BlockProductEmission.of((diag, pois), ((0,), (1,)))
    rng = np.random.default_rng(3)
    y = jnp.asarray(np.column_stack([rng.normal(size=80), rng.poisson(3.0, 80)]))
    gamma = jnp.asarray(rng.dirichlet(np.ones(2), size=80))
    st = prod.sufficient_stats(y, jnp.ones(80), gamma, None)
    assert prod.m_step(st) is None
    partial = prod.m_step_partial(st)
    assert partial[0] is None
    assert partial[1] is not None
    assert set(partial[1]) == {"emission.b1.eta"}
    assert prod.conjugate_prior() is None  # the diagonal Gaussian has no conjugate prior


def test_e11_t08_sample_and_cdf_place_blocks_at_their_columns() -> None:
    """Draws and CDFs of block `b` land in `columns[b]` (here interleaved: Poisson in the middle)."""
    gauss, pois = GaussianFamily(), PoissonFamily(scale_by_dt=False)
    prod = BlockProductEmission.of((ProductEmission.of(gauss, 2), pois), ((0, 2), (1,)))
    theta = {
        "emission.b0.mean": jnp.asarray([[-5.0, 5.0], [0.0, 0.0]]),
        "emission.b0.scale": jnp.asarray([[0.01, 0.01], [1.0, 1.0]]),
        "emission.b1.eta": jnp.asarray([[3.0], [0.0]]),
    }
    import jax

    n = 4000
    s = np.asarray(prod.sample(jax.random.key(0), jnp.zeros(n, dtype=jnp.int32), jnp.ones(n), theta, None))
    np.testing.assert_allclose(s[:, 0].mean(), -5.0, atol=0.01)
    np.testing.assert_allclose(s[:, 2].mean(), 5.0, atol=0.01)
    assert abs(s[:, 1].mean() - np.exp(3.0)) < 0.5
    c = np.asarray(prod.cdf(jnp.asarray([[-5.0, 20.0, 5.0]]), jnp.ones(1), theta, None))
    assert c.shape == (1, 2, 3)
    np.testing.assert_allclose(c[0, 0], [0.5, stats.poisson.cdf(20, np.exp(3.0)), 0.5], atol=1e-6)


def test_e11_t08_required_series_support_follows_the_blocks(tmp_path: Path) -> None:
    """`required_series_support` orders the block supports by column (and is `None` without blocks)."""
    from hsmmkit.model.build import required_series_support

    em = _mixed_emission()
    em["blocks"][0]["columns"] = [0, 2]
    em["blocks"][1]["columns"] = [1]
    assert required_series_support(parse_project_config(_raw(tmp_path, em))) == ("real", "count", "real")
    assert required_series_support(core_config_plain(tmp_path)) is None


def core_config_plain(tmp_path: Path) -> Any:  # noqa: ANN401
    """A config without blocks."""
    return parse_project_config(core_raw(tmp_path))


def test_e11_t08_engine_loglik_equals_the_per_series_product_bitwise(tmp_path: Path) -> None:
    """AC2 at engine level: per-series blocks and the E9-T01 shorthand give a bitwise equal loglik."""
    rng = np.random.default_rng(5)
    y = rng.normal(size=(120, 2))
    y[10, 1] = np.nan
    times = np.arange(120).astype("datetime64[D]").astype("datetime64[ns]")
    obs = ObservationSet(("a", "b"), times, y, "memory", "s")

    def prepared(emission: dict[str, Any]) -> tuple[Any, Any, Any]:
        raw = core_raw(tmp_path)
        raw["data"]["source"]["observation_columns"] = ["a", "b"]
        raw["model"]["emission"] = emission
        cfg = parse_project_config(raw)
        data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
        spec = build_model_spec(cfg, data)
        return data, spec, build_expanded(spec)

    short = {
        "family": "gaussian",
        "params": {
            "mean": {"mode": "free", "init": [0.1, -0.2]},
            "scale": {"mode": "free", "init": [0.7, 1.4]},
        },
    }
    blocks = {
        "family": "product",
        "params": {},
        "blocks": [{"family": "gaussian", "columns": [c], "params": short["params"]} for c in (0, 1)],
    }
    data_a, spec_a, space_a = prepared(short)
    data_b, spec_b, space_b = prepared(blocks)
    ta = dict(ParamManager(spec_a.param_specs).unflatten(ParamManager(spec_a.param_specs).init_u()))
    tb = dict(ParamManager(spec_b.param_specs).unflatten(ParamManager(spec_b.param_specs).init_u()))
    for name, value in ta.items():
        if name.startswith("emission."):
            for b in (0, 1):
                tb[f"emission.b{b}.{name.removeprefix('emission.')}"] = value[:, b : b + 1]
    la = ExpandedHMMEngine().loglik(ta, spec_a, space_a, data_a)
    lb = ExpandedHMMEngine().loglik(tb, spec_b, space_b, data_b)
    assert float(la) == float(lb)
    post_a = ExpandedHMMEngine().forward_backward(ta, spec_a, space_a, data_a).smoothed
    post_b = ExpandedHMMEngine().forward_backward(tb, spec_b, space_b, data_b).smoothed
    assert np.array_equal(np.asarray(post_a), np.asarray(post_b))


def test_e11_t08_msar_block_is_rejected(tmp_path: Path) -> None:
    """An `msar` block needs lags / the step state space: rejected with its path (review finding)."""
    em = _mixed_emission()
    em["blocks"][1] = {"family": "msar", "columns": [2], "ar": {"p": 1}, "params": {}}
    data = _data(parse_project_config(_raw(tmp_path, _mixed_emission())), _y(), ("real", "real", "count"))
    with pytest.raises(ConfigError) as err:
        build_model_spec(parse_project_config(_raw(tmp_path, em)), data)
    assert err.value.path == "model.emission.blocks[1].family"


def test_e11_t08_series_support_survives_the_prepared_dataset_round_trip(tmp_path: Path) -> None:
    """`write_prepared` / `read_prepared` keep `series_support` (review finding)."""
    from hsmmkit.persistence.results import read_prepared, write_prepared

    cfg = parse_project_config(_raw(tmp_path, _mixed_emission()))
    data = _data(cfg, _y(), ("real", "real", "count"))
    path = write_prepared(data, tmp_path / "prepared.parquet")
    back = read_prepared(path)
    assert back.series_support == ("real", "real", "count")
    assert back == data
