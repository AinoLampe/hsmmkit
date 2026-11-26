"""`PerStateDuration`: a different duration family per regime (E12-T05, FR-DU03).

`model.duration.per_state: [{family, params}, ...]` (one entry per regime, `engine: explicit`). Each member
is built for one regime (`n_states = 1`) by its own `from_config`; the parameters of regime `i` are named
`duration.s{i}.<name>` and the member sees them under its usual `duration.<name>` names. Phase families
(`geometric`, `negbin`) are allowed as members through their explicit `log_pmf`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp

from hsmmkit.model.duration.base import Params
from hsmmkit.model.params.spec import ParamSpec
from hsmmkit.utils.typing import Array


def prefix_specs(specs: tuple[ParamSpec, ...], i: int) -> tuple[ParamSpec, ...]:
    """The member specs of regime `i` renamed `duration.<name>` -> `duration.s{i}.<name>`."""
    import dataclasses

    return tuple(
        dataclasses.replace(sp, name=f"duration.s{i}.{sp.name.removeprefix('duration.')}") for sp in specs
    )


@dataclass(frozen=True)
class PerStateDuration:
    """One duration family per regime; `families[i]` is built for a single regime.

    Attributes:
        families: the members, in regime order (each a `DurationFamily` for `n_states = 1`).
    """

    families: tuple[Any, ...]
    name: str = "per_state"
    representation: Literal["phase", "explicit"] = "explicit"
    supports_explicit: bool = True

    @property
    def cov_idx(self) -> tuple[int, ...]:
        """Entry-time duration covariates (the members share `model.duration.covariates`)."""
        return tuple(self.families[0].cov_idx) if self.families else ()

    def _view(self, theta: Params, i: int) -> Params:
        prefix = f"duration.s{i}."
        return {f"duration.{k[len(prefix) :]}": v for k, v in theta.items() if k.startswith(prefix)}

    # --- parameters ---------------------------------------------------------------------------------

    def param_specs(self, n_states: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """The members' specs, prefixed `duration.s{i}.`."""
        del n_states
        out: tuple[ParamSpec, ...] = ()
        for i, fam in enumerate(self.families):
            out += prefix_specs(tuple(fam.param_specs(1, cov_names)), i)
        return out

    # --- the phase API does not exist ------------------------------------------------------------------

    def n_phases(self, i: int) -> int:
        """Not defined: explicit durations only."""
        raise NotImplementedError("per-state durations have no phase representation")

    def advance_prob_t(self, theta: Params, dt: Array | float, z_d_t: Array | None = None) -> Array:
        """Not defined: explicit durations only."""
        raise NotImplementedError("per-state durations have no phase advance probability")

    # --- numerics ------------------------------------------------------------------------------

    def log_pmf(self, theta: Params, d_max: int, nominal_dt: float, z_entry: Array | None) -> Array:
        """`(N, D_max)` stacked member log-pmfs (`(T, N, D_max)` with entry-time covariates)."""
        rows = [
            jnp.asarray(f.log_pmf(self._view(theta, i), d_max, nominal_dt, z_entry))
            for i, f in enumerate(self.families)
        ]
        return jnp.concatenate(rows, axis=-2)

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`(N,)` analytic mean sojourn of each member, in clock units."""
        return jnp.concatenate(
            [
                jnp.reshape(f.mean_sojourn(self._view(theta, i), nominal_dt), (-1,))
                for i, f in enumerate(self.families)
            ]
        )

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """Each member's `init_from_mean_sojourn` for its own regime, under the prefixed names."""
        out: Params = {}
        for i, f in enumerate(self.families):
            for k, v in f.init_from_mean_sojourn(jnp.asarray(mean_steps)[i : i + 1], nominal_dt).items():
                out[f"duration.s{i}.{k.removeprefix('duration.')}"] = v
        return out

    def mean_sojourn_steps_truncated(
        self, theta: Params, d_max: int, dt: float, z_entry: Array | None = None
    ) -> Array:
        """`(N,)` mean of the truncated, renormalized pmf of each regime, in steps."""
        p = jnp.exp(self.log_pmf(theta, d_max, dt, z_entry))
        return jnp.sum(p * jnp.arange(1, d_max + 1, dtype=jnp.float64), axis=-1)


__all__ = ["PerStateDuration", "prefix_specs"]
