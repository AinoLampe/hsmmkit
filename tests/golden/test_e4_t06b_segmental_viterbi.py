"""E4-T06b golden: segmental Viterbi of the explicit-duration engine vs `ref_hsmm.segmental_viterbi`."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.engine.viterbi import segments
from tests.golden.test_e4_t06a_explicit import reference_inputs
from tests.helpers.engine_fixtures import make_explicit_scenario
from tests.property.test_e4_t06a_explicit import per_start_reference
from tests.reference import ref_hsmm

ENGINE = ExplicitDurationEngine()


@given(
    n_states=st.integers(2, 4),
    d_max=st.integers(2, 20),
    t_len=st.integers(5, 120),
    family=st.sampled_from(["negbin", "geometric", "gamma", "weibull"]),
    seed=st.integers(0, 10_000),
)
def test_e4_t06b_segmental_viterbi_matches_reference(
    n_states: int, d_max: int, t_len: int, family: str, seed: int
) -> None:
    """AC1: the regime path equals `ref_hsmm.segmental_viterbi`; elapsed counts follow the segments."""
    sc = make_explicit_scenario(n_states, d_max, t_len, family=family, seed=seed)
    path = np.asarray(ENGINE.viterbi(sc.theta, sc.spec, sc.space, sc.data))
    ref = ref_hsmm.segmental_viterbi(*reference_inputs(sc), d_max)

    regime = np.asarray(sc.space.index.regime_of)[path]
    elapsed = np.asarray(sc.space.index.phase_of)[path]
    np.testing.assert_array_equal(regime, ref)
    # elapsed restarts at 1 at every regime change and increases by one inside a segment
    starts = np.flatnonzero(np.r_[True, regime[1:] != regime[:-1]])
    expected = np.concatenate(
        [np.arange(1, b - a + 1) for a, b in zip(starts, [*starts[1:], t_len], strict=True)]
    )
    np.testing.assert_array_equal(elapsed, expected)
    assert elapsed.max() <= d_max
    assert all(end - start + 1 <= d_max for start, end, _ in segments(regime)[:-1])


@given(seed=st.integers(0, 10_000), family=st.sampled_from(["gamma", "weibull", "negbin"]))
def test_e4_t06b_segmental_viterbi_with_entry_time_covariates(seed: int, family: str) -> None:
    """Start-indexed pmf/jump rows: the regime path equals the best segmentation by enumeration."""
    sc = make_explicit_scenario(3, 3, 6, family=family, n_cov_d=1, n_cov_pi=1, seed=seed)
    _, _, best = per_start_reference(sc)
    path = np.asarray(ENGINE.viterbi(sc.theta, sc.spec, sc.space, sc.data))
    np.testing.assert_array_equal(np.asarray(sc.space.index.regime_of)[path], best)


def test_e4_t06b_viterbi_censors_the_last_segment() -> None:
    """A last segment shorter than its pmf mode is decoded: the survival, not the pmf, scores it."""
    sc = make_explicit_scenario(2, 12, 40, family="gamma", seed=17, mean_steps=9.0)
    path = np.asarray(ENGINE.viterbi(sc.theta, sc.spec, sc.space, sc.data))
    ref = ref_hsmm.segmental_viterbi(*reference_inputs(sc), 12)
    np.testing.assert_array_equal(np.asarray(sc.space.index.regime_of)[path], ref)
    assert segments(ref)[-1][1] == 39
