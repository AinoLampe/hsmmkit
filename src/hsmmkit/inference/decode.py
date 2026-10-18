"""`decode`: a fitted model + data -> `PosteriorResult` (E6-T02).

Concept refs: §5.4 (`PosteriorResult`), §5.9. Rebuilds `space`/`engine` from `fit.spec` the same way
`estimation.map.fit` does, runs forward-backward and Viterbi, and assembles the regime-level
filtered/smoothed marginals, the RLE segments and the information criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from regimekit.engine.criteria import information_criteria
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.engine.explicit_duration import ExplicitDurationEngine
from regimekit.engine.viterbi import segments
from regimekit.inference.results import PosteriorResult
from regimekit.model.params.manager import ParamManager
from regimekit.statespace.builders import build_expanded, build_explicit

if TYPE_CHECKING:
    from regimekit.data.contracts import PreparedDataset
    from regimekit.engine.base import InferenceEngine
    from regimekit.estimation.results import FitResult


def decode(fit: FitResult, data: PreparedDataset, engine: InferenceEngine | None = None) -> PosteriorResult:
    """Decode `data` under `fit.spec`/`fit.params`: filtered/smoothed marginals, Viterbi, IC.

    `engine` defaults to the engine `fit.spec.engine` names (`ExpandedHMMEngine`/
    `ExplicitDurationEngine`, default construction); pass one explicitly to reuse a configured
    instance (e.g. a non-default `chunk_size`).

    Raises:
        NumericalError: the engine's log-likelihood is non-finite (`information_criteria`; CLI
            exit code 4).
    """
    spec = fit.spec
    space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
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
    )
