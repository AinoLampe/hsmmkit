"""Deterministic PRNG key derivation. No global RNG anywhere (overview §4.5)."""

from __future__ import annotations

import hashlib

import jax

# Purpose ids (CLAUDE.md §3.5): ints only.
PURPOSE_FORECAST = 10
PURPOSE_SIMULATE = 11
PURPOSE_BACKTEST_FIT = 12
PURPOSE_BACKTEST_FORECAST = 13
PURPOSE_EM_INIT = 14
PURPOSE_FFBS = 15
PURPOSE_MCMC = 16
FORECAST_ORIGIN = 1
FORECAST_COVARIATES = 2
FORECAST_PATHS = 3


def key_for(seed: int, *ids: int) -> jax.Array:
    """Return a deterministic PRNG key: `jax.random.key(seed)` folded with each id in order."""
    key = jax.random.key(seed)
    for purpose_id in ids:
        key = jax.random.fold_in(key, purpose_id)
    return key


def split_n(key: jax.Array, n: int) -> jax.Array:
    """Split `key` into `n` independent keys."""
    return jax.random.split(key, n)


def derive_seed(seed: int, *ids: int) -> int:
    """Deterministically derive a plain int seed from `seed` and `ids` (for non-JAX RNGs).

    Unlike `key_for`, this returns a Python `int`, for call sites that need a per-origin seed
    (e.g. `MonteCarloForecaster`) rather than a `jax.Array` key.
    """
    payload = ",".join(str(x) for x in (seed, *ids)).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)
