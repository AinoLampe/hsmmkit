"""`decode`: a fitted model + data -> `PosteriorResult` (E6-T02).

Concept refs: §5.4 (`PosteriorResult`), §5.9. Rebuilds `space`/`engine` from `fit.spec` the same way
`estimation.map.fit` does, runs forward-backward and Viterbi, and assembles the regime-level
filtered/smoothed marginals, the RLE segments and the information criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.batch import batched_forward_backward, batched_viterbi
from hsmmkit.engine.criteria import information_criteria
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.engine.ffbs import ffbs_expanded, ffbs_explicit
from hsmmkit.engine.viterbi import segments
from hsmmkit.inference.results import PosteriorResult
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.statespace.builders import build_expanded, build_explicit
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.rng import PURPOSE_FFBS, key_for


def _regime_samples(
    fit: FitResult, space: StateSpace, pb: PaddedBatch, samples: int, seed: int
) -> np.ndarray:
    """`(R, B, T_max)` posterior regime paths by FFBS (`-1` on padding)."""
    spec = fit.spec
    key = key_for(seed, PURPOSE_FFBS)
    if spec.engine == "explicit":
        return np.asarray(ffbs_explicit(key, fit.params, spec, space, pb, samples))
    states = np.asarray(ffbs_expanded(key, fit.params, spec, space, pb, samples))
    regimes = np.asarray(space.index.regime_of)[np.maximum(states, 0)]
    return np.where(states >= 0, regimes, -1)


def boundary_probability(paths: np.ndarray) -> np.ndarray:
    """`(T,)` probability that a new segment starts at step `t >= 1`, estimated from `(R, T)` path samples.

    A new segment is a regime change from step `t - 1`; entry 0 is 0 (the first segment starts there).
    """
    out = np.zeros(paths.shape[1])
    out[1:] = (paths[:, 1:] != paths[:, :-1]).mean(axis=0)
    return out


def _decode_batch(
    fit: FitResult, batch: SequenceBatch, space: StateSpace, samples: int = 0, seed: int = 0
) -> list[PosteriorResult]:
    """One batched forward-backward and Viterbi pass; per-sequence results cut to their own lengths."""
    spec = fit.spec
    eng = ExplicitDurationEngine() if spec.engine == "explicit" else ExpandedHMMEngine()
    pb = batch.padded()
    posterior = batched_forward_backward(eng, fit.params, spec, space, pb, batch.ids)
    paths = np.asarray(batched_viterbi(eng, fit.params, spec, space, pb))
    n_free = ParamManager(spec.param_specs).n_free
    drawn = _regime_samples(fit, space, pb, samples, seed) if samples > 0 else None
    out: list[PosteriorResult] = []
    for b, (seq, post) in enumerate(zip(batch.sequences, posterior.unpad(), strict=True)):
        regime_path = np.asarray(space.index.regime_of[paths[b, : seq.observations.shape[0]]])
        ic = information_criteria(
            float(post.loglik), n_free, np.asarray(post.smoothed), mask=np.asarray(seq.mask)
        )
        out.append(
            PosteriorResult(
                times=seq.times,
                filtered=post.regime_filtered(),
                smoothed=post.regime_smoothed(),
                viterbi=regime_path,
                segments=segments(regime_path),
                augmented=post,
                loglik=float(post.loglik),
                aic=ic.aic,
                bic=ic.bic,
                icl=ic.icl,
                pruned_states=fit.pruned_states,
                path_samples=None if drawn is None else drawn[:, b, : seq.observations.shape[0]],
                boundary_prob=None
                if drawn is None
                else boundary_probability(drawn[:, b, : seq.observations.shape[0]]),
            )
        )
    return out


if TYPE_CHECKING:
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.engine.base import InferenceEngine
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.statespace.index import StateSpace


@overload
def decode(
    fit: FitResult,
    data: PreparedDataset,
    engine: InferenceEngine | None = None,
    *,
    samples: int = 0,
    seed: int = 0,
) -> PosteriorResult: ...


@overload
def decode(
    fit: FitResult,
    data: SequenceBatch,
    engine: InferenceEngine | None = None,
    *,
    samples: int = 0,
    seed: int = 0,
) -> list[PosteriorResult]: ...


def decode(
    fit: FitResult,
    data: PreparedDataset | SequenceBatch,
    engine: InferenceEngine | None = None,
    *,
    samples: int = 0,
    seed: int = 0,
) -> PosteriorResult | list[PosteriorResult]:
    """Decode `data` under `fit.spec`/`fit.params`: filtered/smoothed marginals, Viterbi, IC.

    A `SequenceBatch` is decoded in one batched pass and gives one `PosteriorResult` per sequence.

    `engine` defaults to the engine `fit.spec.engine` names (`ExpandedHMMEngine`/
    `ExplicitDurationEngine`, default construction); pass one explicitly to reuse a configured
    instance (e.g. a non-default `chunk_size`).

    `samples=R > 0` also draws `R` exact posterior regime paths by forward-filtering backward-sampling
    (`engine.ffbs`, key `key_for(seed, PURPOSE_FFBS)`): `path_samples (R, T)` and `boundary_prob (T,)`.

    Raises:
        NumericalError: the engine's log-likelihood is non-finite (`information_criteria`; CLI
            exit code 4).
        ConfigError: `samples < 0`, or samples of an explicit model with duration-dependent jumps.
    """
    spec = fit.spec
    space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
    if samples < 0:
        raise ConfigError(f"samples must be >= 0, got {samples}", path="--samples")
    if isinstance(data, SequenceBatch):
        return _decode_batch(fit, data, space, samples, seed)
    eng = (
        engine
        if engine is not None
        else (ExplicitDurationEngine() if spec.engine == "explicit" else ExpandedHMMEngine())
    )

    posterior = eng.forward_backward(fit.params, spec, space, data)
    path = np.asarray(eng.viterbi(fit.params, spec, space, data))
    regime_path = np.asarray(space.index.regime_of[path])
    segs = segments(regime_path)

    manager = ParamManager(spec.param_specs)
    loglik = float(posterior.loglik)
    smoothed = np.asarray(posterior.smoothed)
    ic = information_criteria(loglik, manager.n_free, smoothed, mask=np.asarray(data.mask))

    drawn = (
        _regime_samples(fit, space, SequenceBatch.of(data).padded(), samples, seed)[:, 0, :]
        if samples > 0
        else None
    )
    return PosteriorResult(
        times=data.times,
        filtered=posterior.regime_filtered(),
        smoothed=posterior.regime_smoothed(),
        viterbi=regime_path,
        segments=segs,
        augmented=posterior,
        loglik=loglik,
        aic=ic.aic,
        bic=ic.bic,
        icl=ic.icl,
        pruned_states=fit.pruned_states,
        path_samples=drawn,
        boundary_prob=None if drawn is None else boundary_probability(drawn),
    )
