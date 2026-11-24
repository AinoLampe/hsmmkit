"""`ProductEmission`: independent per-series emissions sharing one regime process (E9-T01, FR-M12).

For `M > 1` series the joint emission density given regime `i` is the product of the series' own
densities, `f_i(y_t) = prod_m f_i^{(m)}(y_{t,m})`, so `log f_i = sum_m log f_i^{(m)}`. A series that is not
observed at step `t` (`mask[t, m] == False`) contributes `0` to that sum while the other series keep
their contribution (the engines' older rule dropped the whole step). The wrapped family supplies the
per-series densities (`log_prob_series`, shape `(T, N, M)`); GBM and Student-t already evaluate every
series independently, so the wrapper adds the masking and the handling of parameters shared across
series (`model.emission.per_series: false`, spec shape `(N,)` instead of `(N, M)`).

Everything else (`param_specs`, `cdf`, `sample`, the carry hooks) is delegated to the base family;
`n_series == 1` models keep the bare family (`model/build.py`), so `M = 1` results are untouched.
Per-series *family* choice (one family per series) is a follow-up: it needs per-series parameter names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import jax.numpy as jnp

from regimekit.model.emission.base import EmissionFamily, Support
from regimekit.model.params import ParamSpec
from regimekit.plugins import register
from regimekit.utils.typing import Array


@register("emission", "product")
@dataclass(frozen=True)
class ProductEmission:
    """Product of independent per-series emission densities over a shared regime (`n_series` series).

    Attributes:
        base: the wrapped family; it must provide `log_prob_series` and `series_param_names`
            (GBM and Student-t do).
        n_series: number of series `M`.
        name, native_time, support, scale_by_dt: copied from `base` (build with `ProductEmission.of`).
    """

    base: Any
    n_series: int
    name: str
    native_time: Literal["continuous", "discrete"]
    support: Support
    scale_by_dt: bool

    @classmethod
    def of(cls, base: EmissionFamily, n_series: int) -> ProductEmission:
        """Wrap `base`, copying its identity attributes."""
        return cls(
            base=base,
            n_series=n_series,
            name=base.name,
            native_time=base.native_time,
            support=base.support,
            scale_by_dt=base.scale_by_dt,
        )

    def _per_series_theta(self, theta: dict[str, Array]) -> dict[str, Array]:
        """`theta` with series-shared `(N,)` parameters tiled to `(N, M)` (unchanged when per series)."""
        names: tuple[str, ...] = self.base.series_param_names
        return {
            k: jnp.tile(v[:, None], (1, self.n_series)) if k in names and v.ndim == 1 else v
            for k, v in theta.items()
        }

    # --- protocol: parameter layout ---------------------------------------------------------

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """The base family's specs (a series axis `(N, M)` where the family has one)."""
        return cast("tuple[ParamSpec, ...]", self.base.param_specs(n_states, n_series, cov_names))

    # --- protocol: numerics ------------------------------------------------------------------

    def log_prob_series(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` per-regime, per-series log-density."""
        return cast("Array", self.base.log_prob_series(y, dt, self._per_series_theta(theta), z))

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` log-density summed over all series (no mask: every series observed)."""
        return jnp.sum(self.log_prob_series(y, dt, theta, z), axis=-1)

    def log_prob_masked(
        self, y: Array, mask: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> Array:
        """`(T, N)` log-density where only observed series contribute: `sum_m mask[t, m] * log f^{(m)}`.

        `y` must be finite at masked entries (the engines pass `where(mask, y, 0)`), so the unselected
        branch of the `where` is finite and its gradient cannot turn into NaN (CLAUDE.md §5).
        """
        per_series = self.log_prob_series(y, dt, theta, z)  # (T, N, M)
        return jnp.sum(jnp.where(mask[:, None, :], per_series, 0.0), axis=-1)

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` per-regime, per-series predictive CDF."""
        return cast("Array", self.base.cdf(y, dt, self._per_series_theta(theta), z))

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, M)` samples: one independent draw per series from the active regime's density."""
        return cast("Array", self.base.sample(key, regime, dt, self._per_series_theta(theta), z))

    def init_carry(self) -> object:
        """No path-dependent state (the base families have none)."""
        return self.base.init_carry()

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Delegated to the base family (identity for GBM / Student-t)."""
        return self.base.step_carry(carry, y_t, theta)

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array) -> object | None:
        """Delegated to the base family."""
        return cast("object | None", self.base.sufficient_stats(y, dt, gamma))
