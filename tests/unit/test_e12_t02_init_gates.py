"""E12-T02 unit: removed gate rows, pooled inits stay inside sequences, `simulate_batch` guards."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.config import gates
from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.estimation.init import _mean_run_length_steps, pool_sequences, rolling_kmeans
from hsmmkit.inference.simulate import simulate_batch
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw


def _spec(tmp_path: Path, *, groups: int = 1, per_group: bool = False) -> object:
    raw = core_raw(tmp_path)
    if per_group:
        raw["model"]["initial_state"] = "free"
        raw["model"]["initial_groups"] = "per_group"
    return build_model_spec_from(parse_project_config(raw), BuildInputs(1, {}, "steps", 1.0, groups))


def test_e12_t02_the_multi_sequence_gate_rows_are_gone() -> None:
    """The three E12-T02 rows (`data.sequences`, `sequence_column`, `initial_groups`) left `GATES`."""
    assert {"data.sequences", "data.source.sequence_column", "model.initial_groups"}.isdisjoint(gates.GATES)


def test_e12_t02_keys_of_the_removed_gates_now_build(tmp_path: Path) -> None:
    """`data.sequences`, `sequence_column` and `initial_groups: per_group` pass the build."""
    raw = core_raw(tmp_path)
    raw["data"]["sequences"] = {"bucket_edges": [8, 16]}
    raw["data"]["source"]["sequence_column"] = "sid"
    raw["model"]["initial_state"] = "free"
    raw["model"]["initial_groups"] = "per_group"
    cfg = parse_project_config(raw)
    gates.check(cfg)
    spec = build_model_spec_from(cfg, BuildInputs(1, {}, "steps", 1.0, 2))
    assert spec.n_groups == 2


def _batch_of_two_scales(n: int = 40) -> SequenceBatch:
    """Two sequences: quiet (sd 0.1) then loud (sd 1.0), so windows over the seam would mix the scales."""
    spec = _spec(Path("/tmp/x"))
    manager = ParamManager(spec.param_specs)  # type: ignore[attr-defined]
    theta = dict(manager.unflatten(manager.init_u()))
    quiet = simulate_batch(theta, spec, [n], jax.random.key(0)).sequences[0]  # type: ignore[arg-type]
    rng = np.random.default_rng(0)
    import dataclasses

    def with_values(ds: object, sd: float) -> object:
        return dataclasses.replace(ds, observations=jnp.asarray(rng.normal(0.0, sd, size=(n, 1))))  # type: ignore[type-var]

    return SequenceBatch((with_values(quiet, 0.1), with_values(quiet, 1.0)), (), ("q", "l"))  # type: ignore[arg-type]


def test_e12_t02_pooled_rolling_kmeans_windows_stay_inside_sequences(tmp_path: Path) -> None:
    """With `segments` the two scales are recovered; windows across the seam would blend them."""
    batch = _batch_of_two_scales()
    pooled, starts = pool_sequences(batch)
    assert starts.tolist() == [0, 40]
    spec = _spec(tmp_path)
    theta = rolling_kmeans(pooled, spec, jax.random.key(1), segments=starts)  # type: ignore[arg-type]
    sigma = np.sort(np.asarray(theta["emission.scale"]).ravel())
    np.testing.assert_allclose(sigma, [0.1, 1.0], rtol=0.35)


def test_e12_t02_run_lengths_do_not_cross_a_break() -> None:
    """A label run that spans two pooled sequences is cut at the sequence start."""
    labels = np.array([0, 0, 0, 0, 1, 1])
    assert _mean_run_length_steps(labels, 2)[0] == 4.0
    assert _mean_run_length_steps(labels, 2, np.array([2]))[0] == 2.0


def test_e12_t02_simulate_batch_group_guards(tmp_path: Path) -> None:
    """`groups` need a per-group model, one name per sequence, and a per-group model needs `groups`."""
    shared = _spec(tmp_path)
    grouped = _spec(tmp_path, groups=2, per_group=True)
    m_shared = ParamManager(shared.param_specs)  # type: ignore[attr-defined]
    m_group = ParamManager(grouped.param_specs)  # type: ignore[attr-defined]
    th_shared = dict(m_shared.unflatten(m_shared.init_u()))
    th_group = dict(m_group.unflatten(m_group.init_u()))
    key = jax.random.key(0)
    with pytest.raises(ConfigError, match="per_group"):
        simulate_batch(th_shared, shared, [5, 5], key, groups=["a", "b"])  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="needs groups"):
        simulate_batch(th_group, grouped, [5, 5], key)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="differ in length"):
        simulate_batch(th_group, grouped, [5, 5], key, groups=["a"])  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match=">= 1"):
        simulate_batch(th_shared, shared, [0], key)  # type: ignore[arg-type]
    batch = simulate_batch(th_group, grouped, [5, 6], key, groups=["a", "b"])  # type: ignore[arg-type]
    assert batch.group_names == ("a", "b")
    assert [s.observations.shape[0] for s in batch.sequences] == [5, 6]
