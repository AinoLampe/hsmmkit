"""E12-T01a: `SequenceBatch`, padding to buckets, pooled `Preparer.transform_batch`, group ids."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import DataSection
from hsmmkit.data.batch import PaddedBatch, SequenceBatch, split_sequences
from hsmmkit.data.contracts import CovariateSeries, ObservationSet
from hsmmkit.data.prepare import Preparer
from hsmmkit.data.sources import load_data
from hsmmkit.presets.fixtures import segmentation_gaussian_ltr
from hsmmkit.utils.errors import ConfigError

END = 10**6  # a train window that covers every step of a time-less sequence


def _section(**extra: dict[str, Any]) -> DataSection:
    raw: dict[str, Any] = {
        "source": {"kind": "pandas", "observation_columns": ["x", "y"], "sequence_column": "sid"},
        "clock": "steps",
        "sequences": {"bucket_edges": [8, 16]},
        **extra,
    }
    return DataSection.model_validate(raw)


def _frame(lengths: tuple[int, ...], seed: int = 0, groups: tuple[str, ...] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for i, n in enumerate(lengths):
        df = pd.DataFrame({"sid": f"s{i}", "x": rng.normal(i, 1.0, n), "y": rng.normal(-i, 2.0, n)})
        if groups is not None:
            df["g"] = groups[i]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _batch(lengths: tuple[int, ...], **extra: dict[str, Any]) -> tuple[SequenceBatch, Preparer, Any]:
    section = _section(**extra)
    obs, covs = load_data(section, obj=_frame(lengths))
    prep = Preparer(section).fit(obs, covs, END)
    return prep.transform_batch(obs, covs), prep, (obs, covs)


# --- AC1/AC2: padding and unpadding -----------------------------------------------------------------------


def test_e12_t01a_unpad_of_padded_returns_every_sequence_bitwise() -> None:
    """AC1: `unpad(padded())` equals each sequence's arrays; `lengths` and `ids` are preserved."""
    batch, _, _ = _batch((5, 11, 7))
    assert batch.ids == ("s0", "s1", "s2")
    assert batch.lengths.tolist() == [5, 11, 7]
    padded = batch.padded()
    for name in ("observations", "mask", "dt"):
        for seq, back in zip(batch.sequences, batch.unpad(getattr(padded, name)), strict=True):
            assert np.array_equal(np.asarray(getattr(seq, name)), np.asarray(back))
    for seq, back in zip(batch.sequences, batch.unpad(padded.valid), strict=True):
        assert bool(jnp.all(back))
        assert back.shape[0] == seq.observations.shape[0]
    flat = padded.observations.reshape(padded.observations.shape[0], padded.observations.shape[1], -1)
    for seq, back in zip(batch.sequences, batch.unpad(flat.transpose(0, 2, 1), time_axis=2), strict=True):
        assert np.array_equal(np.asarray(seq.observations).T, np.asarray(back))


def test_e12_t01a_shapes_are_bucketed_and_b_is_a_power_of_two() -> None:
    """AC2: `T_max` = smallest edge >= max length (else rounded up to a multiple of the last edge)."""
    batch, _, _ = _batch((5, 11, 7))
    p = batch.padded()
    assert batch.t_max == 16
    assert p.observations.shape == (4, 16, 2)  # B = 3 -> 4 (one empty sequence)
    assert p.lengths.tolist() == [5, 11, 7, 0]
    assert not bool(p.valid[3].any())
    assert not bool(p.mask[3].any())
    assert bool(jnp.all(p.dt > 0))
    assert bool(jnp.all(p.valid.sum(axis=1) == p.lengths))
    long, _, _ = _batch((40, 3))
    assert long.t_max == 48  # 40 rounded up to a multiple of the last edge (16)
    one, _, _ = _batch((8,))
    assert one.t_max == 8
    assert one.padded().observations.shape[0] == 1


def test_e12_t01a_single_dataset_is_a_batch_of_one() -> None:
    """`SequenceBatch.of(ds)`: `B = 1`, unchanged arrays."""
    batch, _, _ = _batch((6,))
    ds = batch.sequences[0]
    one = SequenceBatch.of(ds, (8,))
    assert one.padded().observations.shape == (1, 8, 2)
    assert np.array_equal(np.asarray(one.unpad(one.padded().observations)[0]), np.asarray(ds.observations))


def test_e12_t01a_padded_batch_is_a_pytree_of_arrays() -> None:
    """Every leaf of `PaddedBatch` is a `jnp` array; it flattens and jits."""
    import jax

    batch, _, _ = _batch((5, 7))
    p = batch.padded()
    leaves = jax.tree_util.tree_leaves(p)
    assert all(isinstance(x, jax.Array) for x in leaves)
    assert isinstance(jax.jit(lambda b: b)(p), PaddedBatch)


def test_e12_t01a_mismatched_sequences_are_rejected() -> None:
    """Sequences must share `M`, covariate layout and clock."""
    a, _, _ = _batch((6,))
    other = DataSection.model_validate(
        {
            "source": {"kind": "pandas", "observation_columns": ["x"], "sequence_column": "sid"},
            "clock": "steps",
        }
    )
    obs, covs = load_data(other, obj=_frame((6,)))
    ds = Preparer(other).fit(obs, covs, END).transform(obs, covs)
    with pytest.raises(Exception, match="differs"):
        SequenceBatch((a.sequences[0], ds))


# --- AC3: pooled statistics and lag isolation ------------------------------------------------------


def test_e12_t01a_standardization_uses_the_pooled_training_rows() -> None:
    """AC3: `PrepStats.obs_mean/obs_std` are those of the concatenated rows of all sequences."""
    section = _section(preparation={"standardize_observations": True})
    frame = _frame((30, 50, 20))
    obs, covs = load_data(section, obj=frame)
    prep = Preparer(section).fit(obs, covs, END)
    values = frame[["x", "y"]].to_numpy()
    np.testing.assert_allclose(prep.prep_stats.obs_mean, values.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(prep.prep_stats.obs_std, values.std(axis=0), rtol=1e-12)
    batch = prep.transform_batch(obs, covs)
    pooled = np.concatenate([np.asarray(s.observations) for s in batch.sequences])
    np.testing.assert_allclose(pooled.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(pooled.std(axis=0), 1.0, rtol=1e-12)


def test_e12_t01a_own_lags_never_cross_sequence_boundaries() -> None:
    """AC3 (property-style): the first `p` steps of each sequence are masked, lags stay within it."""
    section = _section()
    frame = _frame((9, 12, 6))
    obs, covs = load_data(section, obj=frame)
    batch = Preparer(section, own_lags=2).fit(obs, covs, END).transform_batch(obs, covs)
    for i, seq in enumerate(batch.sequences):
        mask = np.asarray(seq.mask)
        assert not mask[:2].any()
        assert mask[2:].all()
        own = frame[frame.sid == f"s{i}"][["x", "y"]].to_numpy()
        z = np.asarray(seq.cov_emission)
        np.testing.assert_array_equal(z[2:, 0:2], own[1:-1])  # lag 1 of the *same* sequence
        np.testing.assert_array_equal(z[2:, 2:4], own[:-2])
    # a spike in one sequence changes nothing in the others
    spiked = frame.copy()
    spiked.loc[spiked.sid == "s1", "x"] += 1e6
    obs2, covs2 = load_data(section, obj=spiked)
    other = Preparer(section, own_lags=2).fit(obs2, covs2, END).transform_batch(obs2, covs2)
    assert np.array_equal(
        np.asarray(other.sequences[0].cov_emission), np.asarray(batch.sequences[0].cov_emission)
    )
    assert np.array_equal(
        np.asarray(other.sequences[2].cov_emission), np.asarray(batch.sequences[2].cov_emission)
    )


def test_e12_t01a_transform_refuses_several_sequences() -> None:
    """`transform` is single-sequence: rows of several sequences need `transform_batch`."""
    section = _section()
    obs, covs = load_data(section, obj=_frame((6, 7)))
    prep = Preparer(section).fit(obs, covs, END)
    with pytest.raises(ConfigError, match="transform_batch"):
        prep.transform(obs, covs)


def test_e12_t01a_covariates_are_split_with_their_sequences() -> None:
    """Row-aligned covariate columns follow their rows; a misaligned series is a `ConfigError`."""
    obs = ObservationSet(
        ("x",),
        None,
        np.arange(6.0).reshape(-1, 1),
        "memory",
        "s",
        sequence_ids=np.array(["a", "a", "b", "b", "b", "a"]),
    )
    cov = CovariateSeries("c", None, np.arange(10.0, 16.0), "memory", "c", "s", "grid")
    parts = split_sequences(obs, [cov])
    assert [p[0] for p in parts] == ["a", "b"]
    np.testing.assert_array_equal(parts[0][2][0].values, [10.0, 11.0, 15.0])
    np.testing.assert_array_equal(parts[1][1].values[:, 0], [2.0, 3.0, 4.0])
    bad = CovariateSeries("c", None, np.arange(3.0), "memory", "c", "s", "grid")
    with pytest.raises(ConfigError, match="row-aligned"):
        split_sequences(obs, [bad])


# --- AC4: groups -----------------------------------------------------------------------------------


def test_e12_t01a_group_column_becomes_group_and_group_index() -> None:
    """`data.sequences.group_by` -> `PreparedDataset.group` and `PaddedBatch.group_index`."""
    section = _section(sequences={"group_by": "g", "bucket_edges": [8, 16]})
    obs, covs = load_data(section, obj=_frame((5, 6, 7), groups=("b", "a", "b")))
    batch = Preparer(section).fit(obs, covs, END).transform_batch(obs, covs)
    assert [s.group for s in batch.sequences] == ["b", "a", "b"]
    assert batch.group_names == ("a", "b")
    assert batch.padded().group_index.tolist() == [1, 0, 1, 0]


def test_e12_t01a_group_varying_within_a_sequence_is_a_config_error() -> None:
    """AC4: `ConfigError(path="data.sequences.group_by")`."""
    section = _section(sequences={"group_by": "g", "bucket_edges": [8]})
    frame = _frame((5, 6), groups=("a", "b"))
    frame.loc[2, "g"] = "b"
    obs, covs = load_data(section, obj=frame)
    with pytest.raises(ConfigError) as err:
        Preparer(section).fit(obs, covs, END).transform_batch(obs, covs)
    assert err.value.path == "data.sequences.group_by"


# --- AC5: the segmentation preset ------------------------------------------------------------------


def test_e12_t01a_segmentation_preset_loads_50_sequences_with_the_generator_lengths() -> None:
    """AC5: the `core/segmentation_gaussian_ltr` fixture gives 50 sequences of the generated lengths."""
    frame = segmentation_gaussian_ltr(seed=3)["data/sensor_windows.parquet"]
    expected = frame.groupby("session_id", sort=False).size().to_numpy()
    section = DataSection.model_validate(
        {
            "source": {
                "kind": "pandas",
                "observation_columns": ["acc_x", "acc_y", "acc_z"],
                "sequence_column": "session_id",
            },
            "clock": "steps",
            "sequences": {"bucket_edges": [256, 512, 1024, 2048]},
            "preparation": {"standardize_observations": True},
        }
    )
    obs, covs = load_data(section, obj=frame)
    batch = Preparer(section).fit(obs, covs, END).transform_batch(obs, covs)
    assert len(batch.sequences) == 50
    assert batch.lengths.tolist() == expected.tolist()
    assert batch.t_max == 512
    assert batch.padded().observations.shape == (64, 512, 3)


def test_e12_t01a_group_edge_cases_are_rejected() -> None:
    """A missing group value, `group_by` without a sequence column and a half-grouped batch are errors."""
    from hsmmkit.utils.errors import DataError

    section = _section(sequences={"group_by": "g", "bucket_edges": [8]})
    frame = _frame((4, 5), groups=("a", "b"))
    frame.loc[0, "g"] = None
    with pytest.raises(DataError) as err:
        load_data(section, obj=frame)
    assert err.value.path == "data.sequences.group_by"
    no_seq = DataSection.model_validate(
        {
            "source": {"kind": "pandas", "observation_columns": ["x", "y"]},
            "clock": "steps",
            "sequences": {"group_by": "g"},
        }
    )
    obs, covs = load_data(no_seq, obj=_frame((4, 5), groups=("a", "b")))
    with pytest.raises(ConfigError) as err2:
        Preparer(no_seq).fit(obs, covs, END)
    assert err2.value.path == "data.sequences.group_by"
    batch, _, _ = _batch((5, 6))
    import dataclasses

    mixed = SequenceBatch((dataclasses.replace(batch.sequences[0], group="a"), batch.sequences[1]))
    with pytest.raises(DataError):
        _ = mixed.group_names


def test_e12_t01a_train_window_spans_all_timed_sequences() -> None:
    """`PrepStats.train_window` is the min and max over the pooled training rows of every sequence."""
    section = DataSection.model_validate(
        {
            "source": {
                "kind": "pandas",
                "observation_columns": ["x"],
                "sequence_column": "sid",
                "time_column": "t",
            },
            "clock": "calendar",
        }
    )
    a = pd.DataFrame({"sid": "A", "t": pd.date_range("2020-06-01", periods=5), "x": np.arange(5.0)})
    b = pd.DataFrame({"sid": "B", "t": pd.date_range("2019-01-01", periods=5), "x": np.arange(5.0)})
    obs, covs = load_data(section, obj=pd.concat([a, b], ignore_index=True))
    prep = Preparer(section).fit(obs, covs, "2021-01-01")
    first, last = prep.prep_stats.train_window
    assert first < "2019-06"
    assert last > "2020-06"
