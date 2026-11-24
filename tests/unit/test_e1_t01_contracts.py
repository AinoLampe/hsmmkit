"""E1-T01: `PreparedDataset` (and friends) validation, windowing, and pytree behaviour."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.data.contracts import (
    BoundaryConstraints,
    CovariateSeries,
    ObservationSet,
    PreparedDataset,
    PrepStats,
)
from regimekit.utils.errors import DataError


def _times(t: int, start: str = "2020-01-01") -> np.ndarray:
    """Return `t` strictly increasing daily timestamps starting at `start`."""
    return np.datetime64(start) + np.arange(t).astype("timedelta64[D]")


def _prep_stats(cov_names: tuple[str, ...] = ()) -> PrepStats:
    """Return a minimal `PrepStats` sized for `cov_names`."""
    k = len(cov_names)
    return PrepStats(
        cov_mean=np.zeros(k),
        cov_std=np.ones(k),
        train_window=("2020-01-01", "2020-06-01"),
        cov_names=cov_names,
        transforms={},
        lags={},
    )


def make_dataset(
    t: int = 6,
    m: int = 2,
    *,
    clock: str = "trading",
    dt_value: float | None = None,
    cov_transition: jax.Array | None = None,
    cov_names: dict[str, tuple[str, ...]] | None = None,
    support: str = "real",
    n_categories: int | None = None,
    observations: jax.Array | None = None,
    mask: jax.Array | None = None,
    labels: jax.Array | None = None,
    boundaries: BoundaryConstraints | None = None,
    group: str | None = None,
) -> PreparedDataset:
    """Build a minimal, valid-by-default `PreparedDataset` for tests to perturb."""
    if dt_value is None:
        dt_value = 1.0 if clock == "steps" else 1.0 / 252
    if observations is None:
        observations = jnp.zeros((t, m))
    if mask is None:
        mask = jnp.ones((t, m), dtype=bool)
    if cov_names is None:
        cov_names = {"transition": (), "duration": (), "emission": ()}
    return PreparedDataset(
        times=_times(t),
        observations=observations,
        dt=jnp.full((t,), dt_value),
        mask=mask,
        cov_transition=cov_transition,
        cov_duration=None,
        cov_emission=None,
        cov_names=cov_names,
        clock=clock,
        nominal_dt=dt_value,
        prep_stats=_prep_stats(),
        snapshot_ids=("snap-1",),
        support=support,
        n_categories=n_categories,
        labels=labels,
        boundaries=boundaries,
        group=group,
    )


def test_e1_t01_prep_stats_hash_consistent_with_eq() -> None:
    """Two structurally-equal `PrepStats` instances hash equally (required: it sits in pytree aux data)."""
    p1 = _prep_stats(cov_names=("a", "b"))
    p2 = _prep_stats(cov_names=("a", "b"))
    assert p1 == p2
    assert hash(p1) == hash(p2)

    p3 = _prep_stats(cov_names=("a", "c"))
    assert p1 != p3


def test_e1_t01_observation_set_equality_and_unhashable() -> None:
    """`ObservationSet` has safe structural equality and is intentionally unhashable."""
    o1 = ObservationSet(
        series_names=("A",),
        times=_times(3),
        values=np.zeros((3, 1)),
        source="x",
        snapshot_id="s",
        meta={"interval": "1d", "adjusted": True},
    )
    o2 = dataclasses.replace(o1)
    assert o1 == o2
    assert o1 != dataclasses.replace(o1, source="y")
    assert o1 != object()
    with pytest.raises(TypeError):
        hash(o1)


def test_e1_t01_covariate_series_equality_and_unhashable() -> None:
    """`CovariateSeries` has safe structural equality and is intentionally unhashable."""
    c1 = CovariateSeries(
        name="log_vix",
        times=_times(3),
        values=np.zeros(3),
        source="fred",
        series_id="VIXCLS",
        snapshot_id="s",
        frequency="1d",
    )
    c2 = dataclasses.replace(c1)
    assert c1 == c2
    assert c1 != dataclasses.replace(c1, name="other")
    assert c1 != object()
    with pytest.raises(TypeError):
        hash(c1)


# --- AC6: defaults validate and behave as before the amendment -----------------------------


def test_e1_t01_default_dataset_validates() -> None:
    """A dataset with amendment defaults (`support='real'`, no labels/boundaries/group) validates."""
    make_dataset().validate()


def test_e1_t01_steps_clock_with_dt_ones_validates() -> None:
    """`clock='steps'` with `dt = ones(T)` validates."""
    make_dataset(clock="steps", dt_value=1.0).validate()


def test_e1_t01_steps_clock_with_wrong_dt_fails_on_dt_path() -> None:
    """`clock='steps'` with `dt = 1/252` fails on path `dt`."""
    with pytest.raises(DataError) as exc_info:
        make_dataset(clock="steps", dt_value=1.0 / 252).validate()
    assert exc_info.value.path == "dt"


# --- AC1: each validate() rule has a failing test with a typed DataError + path ------------


def test_e1_t01_non_monotonic_times_fails() -> None:
    """Non-monotonic `times` fails on path `times`."""
    ds = make_dataset()
    bad_times = ds.times.copy()
    bad_times[0], bad_times[1] = bad_times[1], bad_times[0]
    ds = dataclasses.replace(ds, times=bad_times)
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "times"


def test_e1_t01_dt_not_positive_fails() -> None:
    """A non-positive `dt` entry fails on path `dt`."""
    ds = make_dataset()
    ds = dataclasses.replace(ds, dt=ds.dt.at[0].set(0.0))
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "dt"


def test_e1_t01_non_finite_observation_where_masked_fails() -> None:
    """A non-finite observation where `mask` is True fails on path `observations`."""
    ds = make_dataset()
    ds = dataclasses.replace(ds, observations=ds.observations.at[0, 0].set(jnp.nan))
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "observations"


def test_e1_t01_non_finite_observation_where_unmasked_is_fine() -> None:
    """A non-finite observation where `mask` is False does not fail validation."""
    ds = make_dataset()
    mask = ds.mask.at[0, 0].set(False)
    ds = dataclasses.replace(ds, observations=ds.observations.at[0, 0].set(jnp.nan), mask=mask)
    ds.validate()


def test_e1_t01_covariate_wrong_row_count_fails() -> None:
    """A covariate block with `T` rows mismatching the dataset's `T` fails on its own path."""
    ds = make_dataset(
        cov_transition=jnp.zeros((3, 1)),
        cov_names={"transition": ("log_vix",), "duration": (), "emission": ()},
    )
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "cov_transition"


def test_e1_t01_cov_names_length_mismatch_fails() -> None:
    """`cov_names` length mismatching the covariate block's `K` fails on path `cov_names`."""
    ds = make_dataset(
        cov_transition=jnp.zeros((6, 2)),
        cov_names={"transition": ("log_vix",), "duration": (), "emission": ()},
    )
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "cov_names"


def test_e1_t01_support_real_with_integer_dtype_fails() -> None:
    """`support='real'` with non-float observations fails on path `observations`."""
    ds = make_dataset(observations=jnp.zeros((6, 2), dtype=jnp.int32))
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "observations"


def test_e1_t01_support_count_requires_nonnegative_integers() -> None:
    """`support='count'` rejects negative values and accepts non-negative integers."""
    ds = make_dataset(support="count", observations=jnp.full((6, 2), -1, dtype=jnp.int32))
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "observations"
    make_dataset(support="count", observations=jnp.ones((6, 2), dtype=jnp.int32)).validate()


def test_e1_t01_support_binary_requires_zero_or_one() -> None:
    """`support='binary'` rejects values outside `{0, 1}` and accepts `{0, 1}`."""
    ds = make_dataset(support="binary", observations=jnp.full((6, 2), 2.0))
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "observations"
    make_dataset(support="binary", observations=jnp.zeros((6, 2))).validate()


def test_e1_t01_support_categorical_requires_n_categories() -> None:
    """`support='categorical'` without `n_categories` fails on path `n_categories`."""
    ds = make_dataset(support="categorical", observations=jnp.zeros((6, 2), dtype=jnp.int32))
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "n_categories"


def test_e1_t01_support_categorical_out_of_range_fails() -> None:
    """`support='categorical'` rejects values outside `[0, n_categories)`."""
    ds = make_dataset(
        support="categorical",
        n_categories=3,
        observations=jnp.full((6, 2), 5, dtype=jnp.int32),
    )
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "observations"
    make_dataset(
        support="categorical", n_categories=3, observations=jnp.zeros((6, 2), dtype=jnp.int32)
    ).validate()


def test_e1_t01_labels_wrong_shape_fails() -> None:
    """`labels` not shaped `(T, N)` fails on path `labels`."""
    ds = make_dataset(labels=jnp.ones((5, 2), dtype=bool))
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "labels"


def test_e1_t01_labels_correct_shape_validates() -> None:
    """`labels` shaped `(T, N)` bool validates."""
    make_dataset(labels=jnp.ones((6, 3), dtype=bool)).validate()


def test_e1_t01_boundaries_out_of_range_fails() -> None:
    """A boundary index outside `[1, T-1]` fails on path `boundaries`."""
    ds = make_dataset(boundaries=BoundaryConstraints(must_switch_at=(0,)))
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "boundaries"

    ds = make_dataset(boundaries=BoundaryConstraints(must_not_switch_at=(6,)))
    with pytest.raises(DataError) as exc_info:
        ds.validate()
    assert exc_info.value.path == "boundaries"


def test_e1_t01_boundaries_in_range_validates() -> None:
    """Boundary indices inside `[1, T-1]` validate."""
    make_dataset(boundaries=BoundaryConstraints(must_switch_at=(1, 4), must_not_switch_at=(2,))).validate()


# --- AC2: window() ---------------------------------------------------------------------------


def test_e1_t01_window_full_range_equals_original() -> None:
    """`window(0, T)` equals the original dataset."""
    ds = make_dataset(
        cov_transition=jnp.arange(12.0).reshape(6, 2),
        cov_names={"transition": ("a", "b"), "duration": (), "emission": ()},
        labels=jnp.ones((6, 3), dtype=bool),
        boundaries=BoundaryConstraints(must_switch_at=(2,), must_not_switch_at=(4,)),
        group="g1",
    )
    assert ds.window(0, 6) == ds


def test_e1_t01_window_preserves_metadata_and_slices_arrays() -> None:
    """`window()` slices array fields and preserves `cov_names`/`prep_stats`/`snapshot_ids`/etc."""
    ds = make_dataset(
        cov_transition=jnp.arange(12.0).reshape(6, 2),
        cov_names={"transition": ("a", "b"), "duration": (), "emission": ()},
        labels=jnp.arange(18).reshape(6, 3) % 2 == 0,
    )
    w = ds.window(1, 4)
    assert w.observations.shape == (3, 2)
    assert w.dt.shape == (3,)
    assert w.mask.shape == (3, 2)
    assert w.cov_transition.shape == (3, 2)
    assert w.labels.shape == (3, 3)
    assert w.cov_names == ds.cov_names
    assert w.prep_stats == ds.prep_stats
    assert w.snapshot_ids == ds.snapshot_ids
    assert w.support == ds.support
    assert w.group == ds.group
    assert w.nominal_dt == ds.nominal_dt  # kept from parent, not recomputed


def test_e1_t01_window_reindexes_and_drops_boundaries() -> None:
    """`window()` re-indexes boundary indices to the window and drops out-of-window entries."""
    ds = make_dataset(boundaries=BoundaryConstraints(must_switch_at=(1, 3), must_not_switch_at=(4,)))
    w = ds.window(2, 6)
    assert w.boundaries is not None
    assert w.boundaries.must_switch_at == (1,)  # index 3 - 2
    assert w.boundaries.must_not_switch_at == (2,)  # index 4 - 2


def test_e1_t01_tree_flatten_unflatten_round_trip_preserves_all_fields() -> None:
    """Flattening then unflattening reconstructs an equal dataset, including `labels`."""
    ds = make_dataset(
        cov_transition=jnp.zeros((6, 1)),
        cov_names={"transition": ("a",), "duration": (), "emission": ()},
        labels=jnp.ones((6, 2), dtype=bool),
        boundaries=BoundaryConstraints(must_switch_at=(2,)),
        group="g1",
    )
    leaves, treedef = jax.tree_util.tree_flatten(ds)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == ds
    assert rebuilt.labels is not None


# --- AC3: pytree behaviour ---------------------------------------------------------------------


def test_e1_t01_tree_leaves_are_only_array_fields() -> None:
    """`tree_leaves` returns only the JAX-array fields (metadata fields are not leaves)."""
    ds = make_dataset(
        cov_transition=jnp.zeros((6, 1)),
        cov_names={"transition": ("a",), "duration": (), "emission": ()},
        labels=jnp.ones((6, 2), dtype=bool),
    )
    leaves = jax.tree_util.tree_leaves(ds)
    # observations, dt, mask, cov_transition, labels (cov_duration/cov_emission are None -> no leaf)
    assert len(leaves) == 5
    for leaf in leaves:
        assert isinstance(leaf, jax.Array)


def test_e1_t01_jit_of_function_taking_prepared_dataset_works() -> None:
    """`jax.jit` of a function taking a `PreparedDataset` works and caches correctly."""
    ds = make_dataset()

    @jax.jit
    def total(d: PreparedDataset) -> jax.Array:
        return jnp.sum(d.observations) + jnp.sum(d.dt)

    result = total(ds)
    assert float(result) == pytest.approx(float(jnp.sum(ds.dt)))
    assert total._cache_size() == 1
    total(ds)
    assert total._cache_size() == 1  # no recompilation for the identical dataset

    ds2 = dataclasses.replace(ds, times=_times(6, start="2021-01-01"))
    total(ds2)
    assert total._cache_size() == 2  # different `times` metadata -> new compile, as expected

    # A structurally-equal but distinct `prep_stats` object must NOT force recompilation
    # (PrepStats sits in the pytree's aux data; its __hash__ must match its __eq__, or jax would
    # treat every "new but equal" PrepStats instance as a different static structure).
    ds3 = dataclasses.replace(ds, prep_stats=_prep_stats())
    total(ds3)
    assert total._cache_size() == 2  # same PrepStats content -> cache reused, no recompilation
