"""Blocked Gibbs sampler for fully conjugate models (E13-T06b; Part II §14.9 (b), §14.7).

One sweep per padded batch:

1. the state paths `s ~ p(s | theta, y)` by forward-filtering backward-sampling (`engine.ffbs`, one draw per
   sequence, expanded engine);
2. the sufficient statistics of the sampled paths: one-hot regime weights for the emission statistics,
observed
   stay / advance counts per rate row and exit counts per jump row from the consecutive expanded states;
3. `theta_block ~ posterior(prior, statistics)` for every block with the conjugate samplers of
`model.conjugate`
   (Normal-Gamma / Normal-Wishart, Gamma, Beta, Dirichlet; `jax.random`).

Chains are `vmap`ped inside one `lax.scan` over the sweeps; the key of chain `c`, sweep `k` is
`fold_in(fold_in(key_for(seed, PURPOSE_MCMC, c), 2), k)`, so the draws are reproducible per seed. The
priors are
those of the VB estimator (weakly informative defaults, Gaussian priors scaled to the data). Every
component must be
conjugate (`estimation.vb.vb_treatment`); otherwise a `ConfigError` lists the non-conjugate components.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from hsmmkit.engine.ffbs import ffbs_expanded
from hsmmkit.estimation.mstep import transition_mstep
from hsmmkit.estimation.vb import _Context, check_vb_supported
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.rng import PURPOSE_MCMC, key_for
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

__all__ = ["check_gibbs_supported", "run_gibbs"]


def check_gibbs_supported(spec: ModelSpec, pb: PaddedBatch) -> None:
    """Raise `ConfigError` unless every component is conjugate (the list names the others)."""
    treat = check_vb_supported(spec, pb)
    point = [name for name, kind in treat.table().items() if kind == "point"]
    if point:
        raise ConfigError(
            f"sampler 'gibbs' needs conjugate components; not conjugate: {point} "
            "(covariate links, ordered / fixed / bounded parameters, irregular steps, "
            "non-conjugate families); use sampler: nuts",
            path="estimation.mcmc.sampler",
        )


def _path_stats(paths: Array, space: StateSpace, n_regimes: int) -> SimpleNamespace:
    """Counts of one batch of expanded-state paths `(B, T)` (`-1` marks padding)."""
    idx = space.index
    n_rate, (n_row, n_reg) = space.n_rates, space.entry_state_of.shape
    valid = paths >= 0
    s = jnp.maximum(paths, 0)
    prev, cur = s[:, :-1], s[:, 1:]
    v = valid[:, 1:] & valid[:, :-1]
    same = prev == cur
    k_prev = space.rate_index_of[prev]
    stay = jnp.zeros(n_rate).at[k_prev].add(jnp.where(v & same, 1.0, 0.0))
    advance = jnp.zeros(n_rate).at[k_prev].add(jnp.where(v & ~same, 1.0, 0.0))
    leaves = v & ~same & space.is_last_phase[prev] & (idx.regime_of[prev] != idx.regime_of[cur])
    jump = (
        jnp.zeros((n_row, n_reg))
        .at[idx.row_of_state[prev], idx.regime_of[cur]]
        .add(jnp.where(leaves, 1.0, 0.0))
    )
    gamma = jax.nn.one_hot(idx.regime_of[s], n_regimes) * valid[..., None]
    return SimpleNamespace(gamma_regime=gamma, stay=stay, advance=advance, jump=jump)


def run_gibbs(
    theta0: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    *,
    seed: int,
    chains: int,
    warmup: int,
    draws: int,
    keep_paths_every: int = 0,
) -> tuple[dict[str, Array], Array | None]:
    """Draws `{name: (chain, draw, *shape)}` of the constrained parameters and, on request, the paths.

    The paths are `(chain, kept_draws, B, T)` regime indices (`-1` on padding) of every `keep_paths_every`-th
    kept sweep (`None` when `keep_paths_every == 0`).
    """
    check_gibbs_supported(spec, pb)
    treat = check_vb_supported(spec, pb)
    ctx = _Context(spec, space, pb, treat)
    n_reg = spec.n_states
    keep_paths = keep_paths_every > 0

    def sweep(theta: Params, key: Array) -> tuple[Params, Array]:
        k_path, k_em, k_dur, k_jump = jax.random.split(key, 4)
        paths = ffbs_expanded(k_path, theta, spec, space, pb, 1)[0]
        stats = _path_stats(paths, space, n_reg)
        q = ctx._update(stats, pb)  # type: ignore[arg-type]
        new = dict(theta)
        for name, value in ctx.prior.sample_params(k_em, q["emission"]).items():
            new[name] = jnp.reshape(value, jnp.shape(theta[name]))
        a = jax.random.beta(k_dur, q["dur"]["a"], q["dur"]["b"])
        for name, value in spec.duration.from_advance_prob(a, ctx.dt).items():  # type: ignore[attr-defined]
            new[name] = jnp.reshape(value, jnp.shape(theta[name]))
        gam = jnp.where(ctx.mask, jax.random.gamma(k_jump, jnp.where(ctx.mask, q["jump"], 1.0)), 0.0)
        new = dict(transition_mstep(spec, space, new, SimpleNamespace(jump=gam), False))  # type: ignore[arg-type]
        regimes = jnp.where(paths >= 0, space.index.regime_of[jnp.maximum(paths, 0)], -1)
        return new, regimes.astype(jnp.int16)

    vsweep: Any = jax.vmap(sweep)
    bases = jnp.stack([jax.random.fold_in(key_for(seed, PURPOSE_MCMC, c), 2) for c in range(chains)])
    sweep_keys = jax.vmap(lambda b: jax.vmap(lambda k: jax.random.fold_in(b, k))(jnp.arange(warmup + draws)))(
        bases
    )  # (chain, sweep)
    theta_chains = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (chains, *jnp.shape(x))), theta0)

    def step(theta: Params, keys: Array) -> tuple[Params, tuple[Params, Array | None]]:
        new, regimes = vsweep(theta, keys)
        return new, (new, regimes if keep_paths else None)

    _, (stacked, paths_out) = jax.lax.scan(step, theta_chains, jnp.swapaxes(sweep_keys, 0, 1))
    out = {k: jnp.swapaxes(v, 0, 1)[:, warmup:] for k, v in stacked.items()}  # (chain, draw, *shape)
    paths = None
    if keep_paths and paths_out is not None:
        paths = jnp.swapaxes(paths_out, 0, 1)[:, warmup:][:, ::keep_paths_every]
    return out, paths
