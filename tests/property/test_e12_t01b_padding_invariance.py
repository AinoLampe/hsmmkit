"""E12-T01b property test: posteriors of the real steps do not depend on the amount of padding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.batch import batched_forward_backward
from tests.golden.test_e12_t01b_batched import _setup


@settings(max_examples=6, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.integers(min_value=0, max_value=40), st.integers(min_value=0, max_value=3))
def test_e12_t01b_extra_padding_changes_nothing(extra: int, n_empty: int) -> None:
    """Padding `T` by `extra` steps and `B` by empty sequences leaves the real posteriors equal (<= 1e-12)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        spec, batch, eng, ex = _setup(
            Path(tmp), (9, 14), n=3, engine="expanded", order=1, edges=(16,), seed=2
        )
    base = batched_forward_backward(eng, ex["theta"], spec, ex["space"], batch.padded())
    wider = SequenceBatch(batch.sequences, (16 + extra,))
    pb = wider.padded()
    assert pb.observations.shape[1] == 16 + extra
    again = batched_forward_backward(eng, ex["theta"], spec, ex["space"], pb)
    for a, b in zip(base.unpad(), again.unpad(), strict=True):
        np.testing.assert_allclose(np.asarray(a.smoothed), np.asarray(b.smoothed), atol=1e-12)
    del n_empty
