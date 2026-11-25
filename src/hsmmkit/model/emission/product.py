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

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import jax
import jax.numpy as jnp

from hsmmkit.model.emission.base import BaseEmission, EmissionFamily, Support
from hsmmkit.model.params import ParamSpec
from hsmmkit.plugins import REGISTRY, register
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EmissionCfg


@dataclass(frozen=True)
class ProductEmission(BaseEmission):
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

    def moments(
        self, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> tuple[Array, Array] | None:
        """Delegated to the base family (independent series: the moments are per series)."""
        return cast(
            "tuple[Array, Array] | None", self.base.moments(regime, dt, self._per_series_theta(theta), z)
        )

    def init_carry(self) -> object:
        """No path-dependent state (the base families have none)."""
        return self.base.init_carry()

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Delegated to the base family (identity for GBM / Student-t)."""
        return self.base.step_carry(carry, y_t, theta)

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array, z: Array | None = None) -> object | None:
        """Delegated to the base family."""
        return cast("object | None", self.base.sufficient_stats(y, dt, gamma, z))


@dataclass(frozen=True)
class BlockPrior:
    """Product of the blocks' conjugate priors: `update(stats)` updates each block with its own statistics."""

    priors: tuple[Any, ...]

    def update(self, stats: tuple[Any, ...]) -> tuple[Any, ...]:
        """Per-block posterior hyperparameters."""
        return tuple(p.update(s) for p, s in zip(self.priors, stats, strict=True))


@register("emission", "product")
@dataclass(frozen=True)
class BlockProductEmission(BaseEmission):
    """Independent **blocks** of series, each block any registered family (E11-T08, FR-M12, FR-EM07).

    `f_i(y_t) = prod_b f_i^{(b)}(y_t[cols_b])`; the blocks partition the `M` observation columns. A block of
    one series is the bare family; a block of several series is the family itself when it is joint
    (full-covariance Gaussian, multivariate Student-t, `msar`, a mixture) and the independent
    `ProductEmission` of the series otherwise. Parameters are prefixed `emission.b{b}.<name>`.

    Masking: a joint block drops its whole row when any of its series is unobserved (marginalizing a partially
    missing joint Gaussian is backlog); an independent block keeps the other series' contribution (E9-T01).

    Attributes:
        blocks: the block families (each for the block's own series count).
        columns: the observation columns of each block (indices into `observation_columns`).
        n_series: `M`.
    """

    blocks: tuple[Any, ...]
    columns: tuple[tuple[int, ...], ...]
    n_series: int
    name: str = "product"
    native_time: Literal["continuous", "discrete"] = "discrete"
    support: Support = "real"
    scale_by_dt: bool = False

    joint: ClassVar[bool] = True
    """Handles its own per-block masking: `model/build.py` does not wrap it in a `ProductEmission`."""

    @classmethod
    def of(cls, blocks: tuple[Any, ...], columns: tuple[tuple[int, ...], ...]) -> BlockProductEmission:
        """Assemble from the block families and their columns (a partition of `0..M-1`)."""
        n_series = sum(len(c) for c in columns)
        if sorted(i for c in columns for i in c) != list(range(n_series)):
            raise ConfigError(
                f"block columns must partition the columns 0..{n_series - 1}, got {list(columns)}",
                path="model.emission.blocks",
            )
        supports = [b.support for b in blocks]
        return cls(
            blocks=blocks,
            columns=columns,
            n_series=n_series,
            native_time="continuous" if any(b.native_time == "continuous" for b in blocks) else "discrete",
            support=next((s for s in supports if s != "real"), "real"),
            scale_by_dt=all(b.scale_by_dt for b in blocks),
        )

    @property
    def series_support(self) -> tuple[Support, ...]:
        """Support of each observation column."""
        out: list[Support] = ["real"] * self.n_series
        for block, cols in zip(self.blocks, self.columns, strict=True):
            for c in cols:
                out[c] = block.support
        return tuple(out)

    # --- parameter layout ---------------------------------------------------------------------------

    @staticmethod
    def _prefix(b: int) -> str:
        return f"emission.b{b}."

    def _block_theta(self, theta: dict[str, Array], b: int) -> dict[str, Array]:
        p = self._prefix(b)
        return {f"emission.{k[len(p) :]}": v for k, v in theta.items() if k.startswith(p)}

    @staticmethod
    def _prefixed(specs: tuple[ParamSpec, ...], b: int) -> tuple[ParamSpec, ...]:
        return tuple(
            dataclasses.replace(sp, name=BlockProductEmission._prefix(b) + sp.name.removeprefix("emission."))
            for sp in specs
        )

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """The blocks' specs, prefixed `emission.b{b}.`."""
        del n_series
        out: tuple[ParamSpec, ...] = ()
        for b, (block, cols) in enumerate(zip(self.blocks, self.columns, strict=True)):
            out += self._prefixed(tuple(block.param_specs(n_states, len(cols), cov_names)), b)
        return out

    @classmethod
    def from_config(
        cls, cfg: EmissionCfg, n_states: int, n_series: int, cov_names: tuple[str, ...]
    ) -> tuple[BlockProductEmission, tuple[ParamSpec, ...]]:
        """`family: product`, `blocks: [{family, columns, params, ...}]` -> family and prefixed specs."""
        if not cfg.blocks:
            raise ConfigError("family 'product' requires model.emission.blocks", path="model.emission.blocks")
        blocks: list[Any] = []
        columns: list[tuple[int, ...]] = []
        specs: list[ParamSpec] = []
        for b, bcfg in enumerate(cfg.blocks):
            path = f"model.emission.blocks[{b}]"
            if bcfg.columns is None:
                raise ConfigError("every block needs its `columns`", path=f"{path}.columns")
            if any(isinstance(c, str) for c in bcfg.columns):
                raise ConfigError(
                    "block columns are indices into observation_columns (names are not supported yet)",
                    path=f"{path}.columns",
                )
            cols = tuple(int(c) for c in bcfg.columns)
            if bcfg.family == "product" or not REGISTRY.has("emission", bcfg.family):
                raise ConfigError(f"unknown block family {bcfg.family!r}", path=f"{path}.family")
            family, block_specs = REGISTRY.get("emission", bcfg.family).from_config(
                bcfg, n_states, len(cols), cov_names
            )
            if len(cols) > 1 and not getattr(family, "joint", False):
                family = ProductEmission.of(family, len(cols))
            blocks.append(family)
            columns.append(cols)
            specs.extend(cls._prefixed(tuple(block_specs), b))
        product = cls.of(tuple(blocks), tuple(columns))
        if product.n_series != n_series:
            raise ConfigError(
                f"the blocks cover {product.n_series} columns but the data has {n_series} series",
                path="model.emission.blocks",
            )
        return product, tuple(specs)

    # --- numerics ---------------------------------------------------------------------------------------

    def _y(self, y: Array, b: int) -> Array:
        return y[:, jnp.asarray(self.columns[b])]

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` sum of the blocks' log-densities (every series observed)."""
        total = 0.0
        for b, block in enumerate(self.blocks):
            total = total + block.log_prob(self._y(y, b), dt, self._block_theta(theta, b), z)
        return cast("Array", total)

    def log_prob_masked(
        self, y: Array, mask: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> Array:
        """`(T, N)`: joint blocks drop a row with any unobserved series, independent blocks keep the rest."""
        total = 0.0
        for b, block in enumerate(self.blocks):
            yb, mb, tb = self._y(y, b), self._y(mask, b), self._block_theta(theta, b)
            if hasattr(block, "log_prob_masked"):
                ll = block.log_prob_masked(yb, mb, dt, tb, z)
            else:
                ll = jnp.where(jnp.all(mb, axis=-1)[:, None], block.log_prob(yb, dt, tb, z), 0.0)
            total = total + ll
        return cast("Array", total)

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)`: each block's CDFs placed at its columns."""
        out = None
        for b, block in enumerate(self.blocks):
            c = block.cdf(self._y(y, b), dt, self._block_theta(theta, b), z)
            if out is None:
                out = jnp.zeros((c.shape[0], c.shape[1], self.n_series), dtype=c.dtype)
            out = out.at[:, :, jnp.asarray(self.columns[b])].set(c)
        assert out is not None
        return out

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, M)`: independent block draws (one key per block) placed at their columns."""
        keys = jax.random.split(key, len(self.blocks))
        out = jnp.zeros((dt.shape[0], self.n_series))
        for b, block in enumerate(self.blocks):
            draw = block.sample(keys[b], regime, dt, self._block_theta(theta, b), z)
            out = out.at[:, jnp.asarray(self.columns[b])].set(draw.astype(out.dtype))
        return out

    def moments(
        self, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> tuple[Array, Array] | None:
        """Per-column mean and variance (`None` if any block has none)."""
        mean = jnp.zeros((dt.shape[0], self.n_series))
        var = jnp.zeros((dt.shape[0], self.n_series))
        for b, block in enumerate(self.blocks):
            m = block.moments(regime, dt, self._block_theta(theta, b), z)
            if m is None:
                return None
            idx = jnp.asarray(self.columns[b])
            mean, var = mean.at[:, idx].set(m[0]), var.at[:, idx].set(m[1])
        return mean, var

    def init_carry(self) -> object:
        """No path-dependent state (blocks with a recurrent emission are not supported)."""
        return None

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Identity."""
        del y_t, theta
        return carry

    # --- estimation hooks -----------------------------------------------------------------------------

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array, z: Array | None = None) -> tuple[Any, ...]:
        """The tuple of the blocks' statistics (a block without any has `None`)."""
        return tuple(
            block.sufficient_stats(self._y(y, b), dt, gamma, z) for b, block in enumerate(self.blocks)
        )

    def m_step_partial(self, stats: object) -> tuple[dict[str, Array] | None, ...]:
        """Per block: its (prefixed) closed-form update, or `None` (that block needs ECM)."""
        assert isinstance(stats, tuple)
        out: list[dict[str, Array] | None] = []
        for b, (block, st) in enumerate(zip(self.blocks, stats, strict=True)):
            hat = None if st is None else block.m_step(st)
            out.append(
                None
                if hat is None
                else {self._prefix(b) + k.removeprefix("emission."): v for k, v in hat.items()}
            )
        return tuple(out)

    def m_step(self, stats: object) -> dict[str, Array] | None:
        """All blocks' updates merged, or `None` if any block has no closed form."""
        parts = self.m_step_partial(stats)
        if any(p is None for p in parts):
            return None
        return {k: v for p in parts if p is not None for k, v in p.items()}

    def conjugate_prior(self) -> BlockPrior | None:
        """The product of the blocks' priors if every block has one."""
        priors = tuple(block.conjugate_prior() for block in self.blocks)
        return None if any(p is None for p in priors) else BlockPrior(priors)

    # --- state relabelling ----------------------------------------------------------------------------

    def default_sort_name(self, theta: dict[str, Array]) -> str | None:
        """The parameter that orders the states (`post_sort_states`): first scale-like of a plain block."""
        for b, block in enumerate(self.blocks):
            if hasattr(block, "n_components"):
                continue
            for base in ("scale", "sigma", "s", "chol", "eta"):
                name = f"{self._prefix(b)}{base}"
                if name in theta:
                    return name
        return None

    def permute_state_axis(self, name: str, value: Array, perm: Array) -> Array:
        """Relabel the states of one parameter; delegated to the owning block when it has a hook."""
        b = int(name.split(".")[1][1:])
        block = self.blocks[b]
        hook = getattr(block, "permute_state_axis", None)
        local = "emission." + name.split(".", 2)[2]
        return value[perm] if hook is None else cast("Array", hook(local, value, perm))
