"""E12-T01a property tests: bucket selection and padding round trip for arbitrary length sets."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from hsmmkit.config.schema import DataSection
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.data.prepare import Preparer
from hsmmkit.data.sources import load_data


def _batch(lengths: list[int], edges: list[int]) -> SequenceBatch:
    section = DataSection.model_validate(
        {
            "source": {"kind": "pandas", "observation_columns": ["x"], "sequence_column": "sid"},
            "clock": "steps",
            "sequences": {"bucket_edges": edges},
        }
    )
    rng = np.random.default_rng(sum(lengths))
    frame = pd.concat(
        [pd.DataFrame({"sid": i, "x": rng.normal(size=n)}) for i, n in enumerate(lengths)], ignore_index=True
    )
    obs, covs = load_data(section, obj=frame)
    return Preparer(section).fit(obs, covs, 10**6).transform_batch(obs, covs)


@given(
    st.lists(st.integers(min_value=2, max_value=60), min_size=1, max_size=9),
    st.lists(st.integers(min_value=1, max_value=80), min_size=1, max_size=4, unique=True),
)
def test_e12_t01a_bucket_is_the_smallest_edge_and_padding_round_trips(
    lengths: list[int], edges: list[int]
) -> None:
    """`T_max` is the smallest edge >= max length (or a multiple of the last edge); `B` a power of two."""
    edges = sorted(edges)
    batch = _batch(lengths, edges)
    longest = max(lengths)
    geq = [e for e in edges if e >= longest]
    expected = geq[0] if geq else -(-longest // edges[-1]) * edges[-1]
    assert batch.t_max == expected
    p = batch.padded()
    b = p.observations.shape[0]
    assert b >= len(lengths)
    assert b & (b - 1) == 0
    assert b < 2 * len(lengths) or b == 1
    assert p.observations.shape[1] == expected
    assert p.lengths.tolist() == [*lengths, *([0] * (b - len(lengths)))]
    assert bool(jnp.all(p.valid.sum(axis=1) == p.lengths))
    for seq, back in zip(batch.sequences, batch.unpad(p.observations), strict=True):
        assert np.array_equal(np.asarray(seq.observations), np.asarray(back))
    assert not bool(p.mask[~p.valid].any())
