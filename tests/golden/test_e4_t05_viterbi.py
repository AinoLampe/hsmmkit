"""E4-T05 golden: Viterbi path vs `ref_hmm.viterbi`, optimality, and `segments` RLE."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.viterbi import segments
from tests.golden.test_e4_t03a_forward import reference_inputs
from tests.helpers.engine_fixtures import Scenario, make_scenario
from tests.reference import ref_hmm


def path_logprob(path: np.ndarray, log_rho: np.ndarray, log_a: np.ndarray, log_emis: np.ndarray) -> float:
    """Joint log-probability `log p(path, y)` under the reference inputs."""
    total = log_rho[path[0]] + log_emis[0, path[0]]
    for t in range(1, len(path)):
        total += log_a[t, path[t - 1], path[t]] + log_emis[t, path[t]]
    return float(total)


def assert_same_up_to_phase_ties(
    path: np.ndarray, ref: np.ndarray, sc: Scenario, inputs: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Identical paths, or an exact phase-timing tie (same regimes, same joint log-probability)."""
    if np.array_equal(path, ref):
        return
    regime_of = np.asarray(sc.space.index.regime_of)
    np.testing.assert_array_equal(regime_of[path], regime_of[ref])
    np.testing.assert_allclose(path_logprob(path, *inputs), path_logprob(ref, *inputs), rtol=0, atol=1e-9)


@given(
    n_states=st.integers(2, 4),
    order=st.integers(1, 2),
    t_len=st.integers(5, 120),
    seed=st.integers(0, 10_000),
    data=st.data(),
)
def test_e4_t05_viterbi_matches_reference(
    n_states: int, order: int, t_len: int, seed: int, data: st.DataObject
) -> None:
    """AC1: the expanded path equals `ref_hmm.viterbi`, up to exact mathematical ties.

    The phase chain is negative binomial: within a regime segment, the moment of each phase advance
    can move by a step at identical joint probability whenever the advance probabilities of those
    steps agree (equal `dt`), a genuine tie that floating-point noise resolves differently in JAX and
    NumPy. So when the paths differ, they must (a) agree on the regime path and (b) have equal joint
    log-probability to 1e-9; otherwise (no tie) they must be identical.
    """
    r = data.draw(st.lists(st.integers(1, 3), min_size=n_states, max_size=n_states))
    n_cov = data.draw(st.integers(0, 2))
    sc = make_scenario(
        n_states, r, order, t_len, n_cov_pi=n_cov, n_cov_d=n_cov, seed=seed, irregular=bool(seed % 2)
    )
    path = np.asarray(ExpandedHMMEngine().viterbi(sc.theta, sc.spec, sc.space, sc.data))
    inputs = reference_inputs(sc)
    ref = ref_hmm.viterbi(*inputs)
    assert_same_up_to_phase_ties(path, ref, sc, inputs)


@given(seed=st.integers(0, 10_000))
def test_e4_t05_viterbi_beats_posterior_sampled_paths(seed: int) -> None:
    """AC2: the Viterbi path is at least as probable as 100 paths drawn from the smoothed marginals."""
    sc = make_scenario(3, [2, 1, 2], 1, 40, seed=seed)
    engine = ExpandedHMMEngine()
    path = np.asarray(engine.viterbi(sc.theta, sc.spec, sc.space, sc.data))
    inputs = reference_inputs(sc)
    best = path_logprob(path, *inputs)
    smoothed = np.asarray(engine.forward_backward(sc.theta, sc.spec, sc.space, sc.data).smoothed)
    rng = np.random.default_rng(seed)
    s = smoothed.shape[1]
    for _ in range(100):
        cand = np.array([rng.choice(s, p=row / row.sum()) for row in smoothed])
        assert best >= path_logprob(cand, *inputs) - 1e-9


def test_e4_t05_viterbi_with_masked_steps_matches_reference() -> None:
    """Masked steps (zero emission) do not change the decode relative to the reference."""
    sc = make_scenario(3, [2, 2, 1], 2, 50, seed=5)
    mask = np.ones((50, 1), dtype=bool)
    mask[10:14] = False
    fields = {f: getattr(sc.data, f) for f in sc.data.__dataclass_fields__}
    data = sc.data.__class__(**{**fields, "mask": mask})
    path = np.asarray(ExpandedHMMEngine().viterbi(sc.theta, sc.spec, sc.space, data))
    inputs = reference_inputs(sc, mask=mask[:, 0])
    assert_same_up_to_phase_ties(path, ref_hmm.viterbi(*inputs), sc, inputs)


def test_e4_t05_segments_single_segment() -> None:
    """AC3: a constant path is one segment with inclusive end."""
    assert segments(np.array([2, 2, 2, 2])) == [(0, 3, 2)]
    assert segments(np.array([1])) == [(0, 0, 1)]


def test_e4_t05_segments_alternating_and_ends() -> None:
    """AC3: alternating paths give length-1 segments; first/last runs keep their bounds."""
    assert segments(np.array([0, 1, 0, 1])) == [(0, 0, 0), (1, 1, 1), (2, 2, 0), (3, 3, 1)]
    assert segments(np.array([0, 0, 1, 1, 1, 2])) == [(0, 1, 0), (2, 4, 1), (5, 5, 2)]
    assert segments(np.array([], dtype=int)) == []
