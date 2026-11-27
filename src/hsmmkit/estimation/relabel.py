"""Post-hoc relabelling of posterior draws (E13-T06b; Stephens 2000).

Without an identifiability guard (an ordering constraint or fixed location / scale parameters) the
posterior of
a hidden-state model is invariant under the permutations of the state labels and a sampler can switch between
the `N!` modes. `relabel_draws` maps every draw to the labelling closest to a *pivot* (the MAP optimum): the
permutation `sigma` minimizes `sum_i || f_{sigma(i)}(draw) - f_i(pivot) ||^2` over the emission location
parameters standardized by the pivot's spread across states (an assignment problem, solved exactly by the
vendored Hungarian algorithm of E14-T03), and is applied to every state-indexed block (emission, duration
rows, jump rows and columns, initial distribution) by `model.lattice.permute_states`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hsmmkit.evaluation.metrics.matching import hungarian
from hsmmkit.model.lattice import permute_states

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec

__all__ = ["location_features", "relabel_draws", "relabel_permutation"]

_LOCATION_NAMES = ("emission.mean", "emission.m", "emission.mu", "emission.eta")
_SCALE_NAMES = ("emission.scale", "emission.s", "emission.sigma")


def location_features(theta: Mapping[str, np.ndarray], n_states: int) -> np.ndarray:
    """`(N, K)` state features: emission locations and log scales (all emission parameters as a fallback)."""
    names = [n for n in (*_LOCATION_NAMES, *_SCALE_NAMES) if n in theta]
    if not names:
        names = [n for n in theta if n.startswith("emission.")]
    cols = []
    for name in names:
        arr = np.asarray(theta[name], dtype=np.float64)
        if name in _SCALE_NAMES:
            arr = np.log(np.maximum(arr, 1e-300))  # regimes that differ by volatility only
        if arr.ndim >= 1 and arr.shape[0] == n_states:
            cols.append(arr.reshape(n_states, -1))
    if not cols:
        return np.zeros((n_states, 1))
    return np.concatenate(cols, axis=1)


def relabel_permutation(features: np.ndarray, pivot: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """`perm` (`perm[new] = old`) minimizing the standardized squared distance to the pivot states."""
    diff = (pivot[:, None, :] - features[None, :, :]) / scale  # (pivot i, draw j, K)
    cost = np.sum(diff**2, axis=-1)
    rows, cols = hungarian(cost)
    perm = np.empty(pivot.shape[0], dtype=np.int64)
    perm[rows] = cols
    return perm


def relabel_draws(
    draws: Mapping[str, np.ndarray], spec: ModelSpec, pivot: Params
) -> tuple[dict[str, np.ndarray], float]:
    """Relabel draws `(chain, draw, *shape)` towards `pivot`; returns `(draws, fraction relabelled)`."""
    n = spec.n_states
    first = next(iter(draws.values()))
    n_chain, n_draw = first.shape[:2]
    pivot_np = {k: np.asarray(v) for k, v in pivot.items()}
    f_pivot = location_features(pivot_np, n)
    spread = np.std(f_pivot, axis=0)
    scale = np.where(spread > 1e-12, spread, np.maximum(np.abs(f_pivot).mean(axis=0), 1.0))
    out = {k: np.array(v, copy=True) for k, v in draws.items()}
    changed = 0
    identity = np.arange(n)
    for c in range(n_chain):
        for d in range(n_draw):
            theta = {k: v[c, d] for k, v in out.items()}
            perm = relabel_permutation(location_features(theta, n), f_pivot, scale)
            if np.array_equal(perm, identity):
                continue
            changed += 1
            permuted = permute_states(theta, spec, perm)
            for k, v in permuted.items():
                out[k][c, d] = np.asarray(v)
    return out, changed / max(n_chain * n_draw, 1)
