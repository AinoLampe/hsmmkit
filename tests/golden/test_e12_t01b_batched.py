"""E12-T01b golden: batched engines equal the single-sequence engines (both engines, padding invariance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.data.prepare import Preparer
from hsmmkit.data.sources import load_data
from hsmmkit.engine.batch import batched_forward_backward, batched_loglik, batched_viterbi
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.model.build import build_model_spec
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.statespace.builders import build_expanded, build_explicit
from tests.helpers.core_config import core_raw


def _setup(
    tmp_path: Path,
    lengths: tuple[int, ...],
    *,
    n: int,
    engine: str,
    order: int,
    edges: tuple[int, ...],
    seed: int,
    topology: str = "full",
) -> tuple[Any, SequenceBatch, Any, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    raw = core_raw(tmp_path)
    m = raw["model"]
    m.update({"n_states": n, "engine": engine, "d_max": 20})
    m["emission"]["params"] = {
        "mean": {"mode": "free", "init": [0.4 * i for i in range(n)]},
        "scale": {"mode": "free", "init": [0.5 + 0.2 * i for i in range(n)]},
    }
    m["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [6.0] * n
    m["transition"]["order"] = order
    m["topology"] = topology
    if engine == "expanded":
        m["transition"]["covariates"] = ["c"]
    raw_model = {
        **raw,
        "data": {"source": {"kind": "pandas", "observation_columns": ["x"]}, "clock": "steps"},
    }
    if engine == "expanded":
        raw_model["data"]["covariates"] = [{"name": "c", "column": "c", "use_in": ["transition"], "lag": 1}]
        raw_model["data"]["source"]["covariate_columns"] = ["c"]
    raw["data"] = {
        "source": {
            "kind": "pandas",
            "observation_columns": ["x"],
            "sequence_column": "sid",
            "covariate_columns": ["c"] if engine == "expanded" else [],
        },
        "clock": "steps",
        "sequences": {"bucket_edges": list(edges)},
        "covariates": [{"name": "c", "column": "c", "use_in": ["transition"], "lag": 1}]
        if engine == "expanded"
        else [],
    }
    cfg = parse_project_config(raw)
    frame = pd.concat(
        [
            pd.DataFrame({"sid": i, "x": rng.normal(size=ln), "c": rng.normal(size=ln)})
            for i, ln in enumerate(lengths)
        ],
        ignore_index=True,
    )
    obs, covs = load_data(cfg.data, obj=frame)
    batch = Preparer(cfg.data).fit(obs, covs, 10**6).transform_batch(obs, covs)
    spec = build_model_spec(parse_project_config(raw_model), batch.sequences[0])  # the fit gate is E12-T02
    mgr = ParamManager(spec.param_specs)
    theta = dict(mgr.unflatten(jnp.asarray(rng.normal(size=mgr.n_free) * 0.5)))
    theta["emission.mean"] = jnp.asarray(rng.normal(size=n))
    theta["emission.scale"] = jnp.asarray(rng.uniform(0.4, 1.5, size=n))
    space = build_explicit(spec) if engine == "explicit" else build_expanded(spec)
    engine_obj = ExplicitDurationEngine() if engine == "explicit" else ExpandedHMMEngine()
    return spec, batch, engine_obj, {"theta": theta, "space": space}


@pytest.mark.parametrize(("engine", "order"), [("expanded", 1), ("expanded", 2), ("explicit", 1)])
@pytest.mark.parametrize("seed", range(3))
def test_e12_t01b_batched_equals_single_sequence(engine: str, order: int, seed: int, tmp_path: Path) -> None:
    """AC1/AC2: loglik sum <= 1e-10 relative, posteriors and Viterbi paths equal the unpadded runs."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 5))
    lengths = tuple(int(x) for x in rng.integers(5, 60, size=int(rng.integers(2, 6))))
    spec, batch, eng, extra = _setup(
        tmp_path, lengths, n=n, engine=engine, order=order, edges=(32, 64), seed=seed
    )
    theta, space = extra["theta"], extra["space"]
    lengths = tuple(int(x) for x in batch.lengths)  # warm-up rows are trimmed per sequence
    pb = batch.padded()
    ll_b = np.asarray(batched_loglik(eng, theta, spec, space, pb))
    singles = [float(eng.loglik(theta, spec, space, ds)) for ds in batch.sequences]
    np.testing.assert_allclose(ll_b[: len(singles)], singles, rtol=1e-10)
    assert abs(ll_b.sum() - sum(singles)) <= 1e-10 * abs(sum(singles))
    assert np.all(ll_b[len(singles) :] == 0.0)

    post = batched_forward_backward(eng, theta, spec, space, pb)
    for b, (ds, p) in enumerate(zip(batch.sequences, post.unpad(), strict=True)):
        ref = eng.forward_backward(theta, spec, space, ds)
        np.testing.assert_allclose(np.asarray(p.smoothed), np.asarray(ref.smoothed), atol=1e-12)
        np.testing.assert_allclose(float(p.loglik), float(ref.loglik), rtol=1e-10)
        assert p.smoothed.shape[0] == lengths[b]
    paths = np.asarray(batched_viterbi(eng, theta, spec, space, pb))
    for b, ds in enumerate(batch.sequences):
        ref_path = np.asarray(eng.viterbi(theta, spec, space, ds))
        regime_b = np.asarray(space.index.regime_of)[paths[b, : lengths[b]]]
        regime_r = np.asarray(space.index.regime_of)[ref_path]
        np.testing.assert_array_equal(regime_b, regime_r)
        assert np.all(paths[b, lengths[b] :] == -1)


@pytest.mark.parametrize("engine", ["expanded", "explicit"])
def test_e12_t01b_padding_amount_does_not_change_the_result(engine: str, tmp_path: Path) -> None:
    """AC2: the same sequences under different bucket edges (more padding) give the same posteriors."""
    lengths = (7, 19, 12)
    results = []
    for edges in ((20,), (32,), (64,)):
        spec, batch, eng, extra = _setup(tmp_path, lengths, n=3, engine=engine, order=1, edges=edges, seed=4)
        post = batched_forward_backward(eng, extra["theta"], spec, extra["space"], batch.padded())
        results.append([np.asarray(p.smoothed) for p in post.unpad()])
    for other in results[1:]:
        for a, b in zip(results[0], other, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12)


def test_e12_t01b_compile_count_is_one_per_bucket(tmp_path: Path) -> None:
    """AC3: three buckets with varying `B` compile at most once per distinct `(B_pad, T_bucket)` shape."""
    from hsmmkit.engine import batch as batch_mod

    spec, batch, eng, extra = _setup(
        tmp_path, (5, 9, 14, 30, 40), n=2, engine="expanded", order=1, edges=(16, 32, 64), seed=1
    )
    seqs = batch.sequences
    shapes = set()
    before = batch_mod._batched_core._cache_size()
    for pick in ((0, 1), (0, 1, 2), (1, 2), (3,), (3, 4), (0, 3), (4,)):
        sub = SequenceBatch(tuple(seqs[i] for i in pick), (16, 32, 64))
        pb = sub.padded()
        shapes.add(pb.observations.shape)
        batched_loglik(eng, extra["theta"], spec, extra["space"], pb)
    # repeating a bucket does not recompile
    mid = batch_mod._batched_core._cache_size()
    batched_loglik(
        eng, extra["theta"], spec, extra["space"], SequenceBatch((seqs[0], seqs[2]), (16, 32, 64)).padded()
    )
    assert batch_mod._batched_core._cache_size() == mid
    assert mid - before <= len(shapes)


def test_e12_t01b_decode_of_a_sequence_batch_matches_single_decodes(tmp_path: Path) -> None:
    """`decode(fit, SequenceBatch)` gives per-sequence results equal to the single-sequence decodes."""
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.inference.decode import decode

    spec, batch, _, extra = _setup(
        tmp_path, (12, 25, 8), n=3, engine="expanded", order=1, edges=(32,), seed=7
    )
    mgr = ParamManager(spec.param_specs)
    fit = FitResult(
        estimator="map",
        params=extra["theta"],
        uparams=mgr.flatten(extra["theta"]),
        objective=0.0,
        loglik=0.0,
        restarts=pd.DataFrame(),
        converged=True,
        n_iter=0,
        std_errors=None,
        diagnostics={},
        spec=spec,
        prep_stats=batch.sequences[0].prep_stats,
        config_resolved={},
    )
    batched = decode(fit, batch)
    assert len(batched) == 3
    for seq, res in zip(batch.sequences, batched, strict=True):
        single = decode(fit, seq)
        np.testing.assert_array_equal(res.viterbi, single.viterbi)
        np.testing.assert_allclose(np.asarray(res.smoothed), np.asarray(single.smoothed), atol=1e-12)
        assert abs(res.loglik - single.loglik) <= 1e-10 * abs(single.loglik)
        assert res.segments == single.segments


def test_e12_t01b_absorbing_topology_and_a_single_step_sequence_under_padding(tmp_path: Path) -> None:
    """Review fix: an absorbing regime adds no mass for starts in the padding; a short sequence works."""
    spec, batch, eng, extra = _setup(
        tmp_path,
        (9, 20, 14, 5),
        n=3,
        engine="explicit",
        order=1,
        edges=(32,),
        seed=3,
        topology="left_to_right",
    )
    theta, space = extra["theta"], extra["space"]
    ll_b = np.asarray(batched_loglik(eng, theta, spec, space, batch.padded()))
    singles = [float(eng.loglik(theta, spec, space, ds)) for ds in batch.sequences]
    np.testing.assert_allclose(ll_b[: len(singles)], singles, rtol=1e-10)
