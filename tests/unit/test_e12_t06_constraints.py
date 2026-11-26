"""E12-T06 unit: label encodings, boundaries, `ConstraintError`, full-label Viterbi, no recompilation."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hsmmkit.cli.app import exit_code_of
from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.data.contracts import BoundaryConstraints
from hsmmkit.data.labels import allowed_from_values, attach_label_file, boundaries_of
from hsmmkit.data.prepare import Preparer
from hsmmkit.data.sources import load_data
from hsmmkit.engine import expanded_hmm
from hsmmkit.engine.batch import batched_loglik, batched_viterbi
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.utils.errors import ConstraintError, DataError, HsmmkitError
from tests.helpers.core_config import core_raw
from tests.helpers.engine_fixtures import Scenario, make_explicit_scenario, make_scenario

N = 3


def test_e12_t06_constraint_error_carries_step_and_sequence_and_maps_to_exit_code_3() -> None:
    """`ConstraintError(step, sequence)` is an `HsmmkitError`; the CLI exit code is 3 (data / constraint)."""
    err = ConstraintError("no path", path="data.labels", step=7, sequence="s2")
    assert isinstance(err, HsmmkitError)
    assert (err.step, err.sequence, err.path) == (7, "s2", "data.labels")
    assert "step 7" in str(err)
    assert "'s2'" in str(err)
    assert exit_code_of(err) == 3


def test_e12_t06_label_encodings() -> None:
    """Ids (NaN / -1 unlabelled), bitmask sets and list-valued sets all give the allowed matrix."""
    ids = allowed_from_values([0, np.nan, 2, -1, None], N)
    assert ids.tolist() == [[1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1]]
    sets = allowed_from_values([0b011, np.nan, 0b100, -1], N, allowed_sets=True)
    assert sets.tolist() == [[1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]]
    lists = allowed_from_values([[0, 2], None, np.array([1])], N, allowed_sets=True)
    assert lists.tolist() == [[1, 0, 1], [1, 1, 1], [0, 1, 0]]


@pytest.mark.parametrize(
    ("values", "kw"),
    [([3], {}), ([0.5], {}), (["a"], {}), ([0], {"allowed_sets": True}), ([0b1000], {"allowed_sets": True})],
)
def test_e12_t06_bad_labels_are_data_errors(values: list[Any], kw: dict[str, Any]) -> None:
    """An id outside `[0, N)`, a non-integer, a string, an empty set or a high bit raises `DataError`."""
    with pytest.raises(DataError):
        allowed_from_values(values, N, **kw)


def test_e12_t06_boundaries_drop_indices_outside_the_sequence_and_reject_conflicts() -> None:
    """Indices >= T are dropped (a configured list applies to every sequence); conflicts and 0 raise."""
    assert boundaries_of([2, 50], [4], 10) == BoundaryConstraints((2,), (4,))
    assert boundaries_of([50], [], 10) is None
    with pytest.raises(DataError, match="both"):
        boundaries_of([3], [3], 10)
    with pytest.raises(DataError, match=">= 1"):
        boundaries_of([0], [], 10)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sid, n in (("a", 8), ("b", 6)):
        for t in range(n):
            rows.append(
                {"sid": sid, "x": rng.normal(), "reg": float(min(t // 3, 1)) if t % 2 == 0 else np.nan}
            )
    return pd.DataFrame(rows)


def _cfg(tmp_path: Path, labels: dict[str, Any]) -> Any:  # noqa: ANN401
    raw = core_raw(tmp_path)
    raw["data"] = {
        "source": {"kind": "pandas", "observation_columns": ["x"], "sequence_column": "sid"},
        "clock": "steps",
        "labels": labels,
    }
    return parse_project_config(raw)


def test_e12_t06_preparer_builds_labels_and_boundaries_per_sequence(tmp_path: Path) -> None:
    """Column labels become `(T, N)` allowed matrices; boundaries are clipped to each sequence."""
    cfg = _cfg(
        tmp_path,
        {
            "source": "column",
            "column": "reg",
            "boundaries": {"must_switch_at": [3, 7], "must_not_switch_at": [1]},
        },
    )
    obs, covs = load_data(cfg.data, obj=_frame())
    batch = Preparer(cfg.data, n_regimes=2).fit(obs, covs, 10**6).transform_batch(obs, covs)
    a, b = batch.sequences
    assert a.labels is not None
    assert a.labels.shape == (8, 2)
    assert np.asarray(a.labels)[0].tolist() == [True, False]  # t=0 labelled regime 0
    assert np.asarray(a.labels)[1].tolist() == [True, True]  # NaN: unlabelled
    assert np.asarray(a.labels)[6].tolist() == [False, True]  # t=6 -> regime 1
    assert a.boundaries == BoundaryConstraints((3, 7), (1,))
    assert b.boundaries == BoundaryConstraints((3,), (1,))  # 7 >= T_b = 6 is dropped
    padded = batch.padded()
    assert padded.boundary_code is not None
    assert padded.boundary_code[0, 3] == 1
    assert padded.boundary_code[1, 1] == 2


def test_e12_t06_label_file_agrees_with_the_column(tmp_path: Path) -> None:
    """A `(sequence_id, t, regime)` file gives the same labels as the equivalent column."""
    frame = _frame()
    rows = []
    for sid, grp in frame.groupby("sid", sort=False):
        for t, v in enumerate(grp["reg"].tolist()):
            if not np.isnan(v):
                rows.append({"sequence_id": sid, "t": t, "regime": int(v)})
    path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    col = _cfg(tmp_path, {"source": "column", "column": "reg"})
    fil = _cfg(tmp_path, {"source": "file", "path": str(path)})
    a = Preparer(col.data, n_regimes=2).fit(*load_data(col.data, obj=frame), 10**6)
    b = Preparer(fil.data, n_regimes=2).fit(*load_data(fil.data, obj=frame), 10**6)
    la = a.transform_batch(*load_data(col.data, obj=frame))
    lb = b.transform_batch(*load_data(fil.data, obj=frame))
    for x, y in zip(la.sequences, lb.sequences, strict=True):
        assert np.array_equal(np.asarray(x.labels), np.asarray(y.labels))


def test_e12_t06_label_file_errors(tmp_path: Path) -> None:
    """Missing columns and entries that match no row are `DataError`s."""
    obs, _ = load_data(_cfg(tmp_path, {"source": "column", "column": "reg"}).data, obj=_frame())
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"t": [0]}).to_csv(bad, index=False)
    with pytest.raises(DataError, match="needs the columns"):
        attach_label_file(obs, bad)
    extra = tmp_path / "extra.csv"
    pd.DataFrame({"sequence_id": ["a"], "t": [99], "regime": [0]}).to_csv(extra, index=False)
    with pytest.raises(DataError, match="match no observation row"):
        attach_label_file(obs, extra)


def _constrain(
    sc: Scenario, allowed: np.ndarray | None, switch: tuple[int, ...] = (), stay: tuple[int, ...] = ()
) -> Scenario:
    boundaries = BoundaryConstraints(switch, stay) if switch or stay else None
    data = dataclasses.replace(
        sc.data, labels=None if allowed is None else jnp.asarray(allowed), boundaries=boundaries
    )
    return dataclasses.replace(sc, data=data)


def _segmented_truth(t_len: int, seg: int) -> np.ndarray:
    """A regime path with segments of `seg` steps cycling through the regimes."""
    return (np.arange(t_len) // seg) % N


def test_e12_t06_full_labels_reproduce_the_labels_under_viterbi_both_engines() -> None:
    """AC1: fully labelled data decode to exactly the labels (expanded NB r=2 and explicit gamma)."""
    t_len = 18
    truth = _segmented_truth(t_len, 3)
    allowed = np.eye(N, dtype=bool)[truth]
    for sc in (
        make_scenario(N, [2, 2, 2], 1, t_len, seed=1),
        make_explicit_scenario(N, 6, t_len, family="gamma", seed=2),
    ):
        con = _constrain(sc, allowed)
        engine = ExpandedHMMEngine() if con.spec.engine == "expanded" else ExplicitDurationEngine()
        path = np.asarray(engine.viterbi(con.theta, con.spec, con.space, con.data))
        assert np.array_equal(np.asarray(con.space.index.regime_of)[path], truth)


def test_e12_t06_infeasible_boundaries_raise_constraint_error_with_step_and_sequence() -> None:
    """AC3: two `must_switch_at` steps 1 apart with NB `r = 3` everywhere name the step and the sequence."""
    t_len = 12
    sc = make_scenario(N, [3, 3, 3], 1, t_len, seed=3)
    ok = dataclasses.replace(sc.data, boundaries=BoundaryConstraints((6,), ()))
    bad = dataclasses.replace(sc.data, boundaries=BoundaryConstraints((5, 6), ()))
    engine = ExpandedHMMEngine()
    assert np.isfinite(float(engine.loglik(sc.theta, sc.spec, sc.space, ok)))
    with pytest.raises(ConstraintError) as info:
        engine.loglik(sc.theta, sc.spec, sc.space, bad)
    assert info.value.step == 6
    batch = SequenceBatch((ok, bad), (), ("fine", "broken"))
    with pytest.raises(ConstraintError) as batch_info:
        batched_loglik(engine, sc.theta, sc.spec, sc.space, batch.padded(), batch.ids)
    assert batch_info.value.sequence == "broken"
    assert batch_info.value.step == 6


def test_e12_t06_a_compiled_engine_serves_labelled_and_unlabelled_sequences() -> None:
    """AC5: partial labels and all-`True` labels share one compilation (masks are dynamic arrays)."""
    sc = make_scenario(N, [2, 2, 2], 1, 16, seed=4)
    free = _constrain(sc, np.ones((16, N), dtype=bool))
    partial = np.ones((16, N), dtype=bool)
    partial[3] = [False, True, False]
    labelled = _constrain(sc, partial)
    engine = ExpandedHMMEngine()
    engine.loglik(free.theta, free.spec, free.space, free.data)
    size = expanded_hmm._forward_core._cache_size()
    engine.loglik(labelled.theta, labelled.spec, labelled.space, labelled.data)
    other = np.ones((16, N), dtype=bool)
    other[8] = [True, False, False]
    again = _constrain(sc, other)
    engine.loglik(again.theta, again.spec, again.space, again.data)
    assert expanded_hmm._forward_core._cache_size() == size


@pytest.mark.parametrize("engine_name", ["expanded", "explicit"])
def test_e12_t06_batch_with_labels_and_boundaries_equals_the_single_sequence_runs(engine_name: str) -> None:
    """Batched loglik / Viterbi under per-sequence labels and boundaries match the single-sequence engines."""
    t_len = 12
    sc = (
        make_scenario(N, [2, 1, 2], 1, t_len, seed=5)
        if engine_name == "expanded"
        else make_explicit_scenario(N, 6, t_len, family="gamma", seed=5)
    )
    truth = _segmented_truth(t_len, 4)
    a_labels = np.ones((t_len, N), dtype=bool)
    a_labels[0] = np.eye(N, dtype=bool)[truth[0]]
    a_labels[5] = np.eye(N, dtype=bool)[truth[5]]
    seqs = [
        _constrain(sc, a_labels, (4,), (2,)).data,
        _constrain(sc, np.ones((t_len, N), dtype=bool), (), (3, 9)).data,
        _constrain(sc, np.eye(N, dtype=bool)[truth]).data,
    ]
    batch = SequenceBatch(tuple(seqs), (), ("a", "b", "c"))
    engine = ExpandedHMMEngine() if engine_name == "expanded" else ExplicitDurationEngine()
    pb = batch.padded()
    got = np.asarray(batched_loglik(engine, sc.theta, sc.spec, sc.space, pb, batch.ids))[:3]
    want = [float(engine.loglik(sc.theta, sc.spec, sc.space, s)) for s in seqs]
    np.testing.assert_allclose(got, want, rtol=1e-10)
    paths = np.asarray(batched_viterbi(engine, sc.theta, sc.spec, sc.space, pb))
    for i, s in enumerate(seqs):
        single = np.asarray(engine.viterbi(sc.theta, sc.spec, sc.space, s))
        assert np.array_equal(paths[i, :t_len], single)


def test_e12_t06_window_drops_boundaries_at_or_before_the_window_start() -> None:
    """A boundary at the first step of a window is not a switch inside it (index 0 would be invalid)."""
    sc = make_scenario(N, [2, 2, 2], 1, 20, seed=6)
    data = dataclasses.replace(
        sc.data, clock="steps", dt=jnp.ones(20), nominal_dt=1.0, boundaries=BoundaryConstraints((5, 8), (10,))
    )
    win = data.window(5, 15)
    assert win.boundaries == BoundaryConstraints((3,), (5,))
    win.validate()


def test_e12_t06_expanded_viterbi_of_infeasible_constraints_is_a_constraint_error() -> None:
    """`viterbi_path` reports an infeasible constraint set as `ConstraintError` (exit code 3)."""
    from hsmmkit.engine.viterbi import viterbi_path

    sc = make_scenario(N, [3, 3, 3], 1, 12, seed=3)
    bad = dataclasses.replace(sc.data, boundaries=BoundaryConstraints((5, 6), ()))
    with pytest.raises(ConstraintError):
        viterbi_path(sc.theta, sc.spec, sc.space, bad)
