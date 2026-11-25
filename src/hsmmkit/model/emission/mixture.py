"""Per-state finite mixture of a base emission family (E11-T06, FR-EM05, FR-EM07).

    f_i(y) = sum_{k=1}^{K} w_ik g(y | theta_ik),     log f_i = logsumexp_k( log w_ik + log g(y | theta_ik) )

Config: `family: mixture`, `mixture: {components: K, base: <family>}`; `params` are the base family's, with
**component axis inside the state axis**: a per-state list of the base has `N * K` entries, state-major
(`[state 0 comp 0, state 0 comp 1, ..., state 1 comp 0, ...]`); a list of `K` entries is repeated for every
state. `params.weights` (optional) configures `emission.weights (N, K)`, rows on the simplex.

The base family is built for `N * K` pseudo-states (pseudo-state `i * K + k` is component `k` of state `i`);
with `M > 1` independent series the base is the `ProductEmission` of the series, so each component is a
product and the mixture is over components. `K == 1` is the base family, bitwise (no weights, no `logsumexp`).

Identifiability: `constraint: increasing` on a base parameter orders its components *within each state*
(`domain: increasing` / `increasing_positive` on an `(N, K)` row cell), single series only. Closed-form EM
hooks need the current parameters for the responsibilities: `sufficient_stats(..., theta=theta)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.model.emission.base import BaseEmission, Support
from hsmmkit.model.emission.product import ProductEmission
from hsmmkit.model.params.spec import ParamSpec
from hsmmkit.plugins import REGISTRY, register
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EmissionCfg, ParamCfg

WEIGHTS = "emission.weights"


@dataclass(frozen=True)
class MixturePrior:
    """Dirichlet prior on the weights times the base family's conjugate prior (`update(stats)`)."""

    base: Any
    alpha0: float = 1.0

    def update(self, stats: dict[str, Any]) -> dict[str, Any]:
        """`{"alpha": alpha0 + sum_t r_tik, "base": base posterior of the pooled statistics}`."""
        return {"alpha": self.alpha0 + stats["r_sum"], "base": self.base.update(stats["base"])}


def _safe_log(w: Array) -> Array:
    """`log w` with `-inf` (not NaN gradients) at `w == 0` (double `where`)."""
    return jnp.where(w > 0, jnp.log(jnp.where(w > 0, w, 1.0)), -jnp.inf)


@register("emission", "mixture")
@dataclass(frozen=True)
class MixtureFamily(BaseEmission):
    """`K`-component mixture of `base` (a family over `N * K` pseudo-states) in every regime.

    Attributes:
        base: the component family (`ProductEmission`-wrapped for `M > 1`).
        k: number of components `K` (the same in every state).
        ordered: names of the `(N, K)` parameters whose rows are increasing (identifiability).
    """

    base: Any
    k: int
    name: str = "mixture"
    native_time: Literal["continuous", "discrete"] = "discrete"
    support: Support = "real"
    scale_by_dt: bool = False
    ordered: tuple[str, ...] = field(default=())

    joint: ClassVar[bool] = True
    """The mixture couples the series of a component: `model/build.py` does not wrap it in a product."""

    @classmethod
    def of(cls, base: Any, k: int, ordered: tuple[str, ...] = ()) -> MixtureFamily:  # noqa: ANN401
        """Wrap `base`, copying its identity attributes."""
        return cls(
            base=base,
            k=k,
            native_time=base.native_time,
            support=base.support,
            scale_by_dt=base.scale_by_dt,
            ordered=ordered,
        )

    @property
    def n_components(self) -> int:
        """`K`."""
        return self.k

    # --- parameter layout --------------------------------------------------------------------------

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """The base specs for `N * K` pseudo-states plus `emission.weights (N, K)` (absent for `K == 1`)."""
        specs = tuple(self.base.param_specs(n_states * self.k, n_series, cov_names))
        return (*specs, _weights_spec(n_states, self.k, None)) if self.k > 1 else specs

    @classmethod
    def from_config(
        cls, cfg: EmissionCfg, n_states: int, n_series: int, cov_names: tuple[str, ...]
    ) -> tuple[MixtureFamily, tuple[ParamSpec, ...]]:
        """`model.emission` with `family: mixture` -> family and specs (see the module docstring)."""
        if cfg.mixture is None:
            raise ConfigError(
                "family 'mixture' requires model.emission.mixture", path="model.emission.mixture"
            )
        k, base_name = cfg.mixture.components, cfg.mixture.base
        if base_name == "mixture" or not REGISTRY.has("emission", base_name):
            raise ConfigError(
                f"unknown mixture base family {base_name!r}", path="model.emission.mixture.base"
            )
        base_cls = REGISTRY.get("emission", base_name)
        if not hasattr(base_cls, "from_config"):
            raise ConfigError(
                f"family {base_name!r} cannot be a mixture base", path="model.emission.mixture.base"
            )
        params = {
            key: _expand_component_axis(p, n_states, k) for key, p in cfg.params.items() if key != "weights"
        }
        base_cfg = cfg.model_copy(update={"family": base_name, "mixture": None, "params": params})
        base, base_specs = base_cls.from_config(base_cfg, n_states * k, n_series, cov_names)
        if n_series > 1 and not getattr(base, "joint", False):
            base = ProductEmission.of(base, n_series)
        if base.init_carry() is not None:
            raise ConfigError(
                f"family {base_name!r} has a recurrent state and cannot be a mixture base",
                path="model.emission.mixture.base",
            )
        if k == 1:  # the base family, untouched (its own state-axis ordering included)
            return cls.of(base, 1), tuple(base_specs)
        specs: list[ParamSpec] = []
        ordered: list[str] = []
        for spec in base_specs:
            if spec.constraint == "increasing":
                if spec.shape != (n_states * k,):
                    raise ConfigError(
                        "an increasing mixture parameter needs a single series (or per_series: false)",
                        path=spec.name,
                    )
                specs.append(_row_ordered(spec, n_states, k))
                ordered.append(spec.name)
            else:
                specs.append(spec)
        specs.append(_weights_spec(n_states, k, cfg.params.get("weights")))
        return cls.of(base, k, tuple(ordered)), tuple(specs)

    # --- parameter views -----------------------------------------------------------------------------

    def _base_theta(self, theta: dict[str, Array]) -> dict[str, Array]:
        """`theta` as the base sees it: `(N, K)` ordered rows flattened, weights dropped."""
        return {
            key: (v.reshape(-1) if key in self.ordered else v) for key, v in theta.items() if key != WEIGHTS
        }

    def _log_w(self, theta: dict[str, Array]) -> Array:
        return _safe_log(theta[WEIGHTS])  # (N, K)

    def _mix(self, ll: Array, theta: dict[str, Array]) -> Array:
        """`(T, N*K)` component log-densities -> `(T, N)` mixture log-density."""
        ll3 = ll.reshape(ll.shape[0], -1, self.k)
        return jax.scipy.special.logsumexp(ll3 + self._log_w(theta)[None], axis=-1)

    # --- protocol: numerics ----------------------------------------------------------------------------

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` mixture log-density."""
        ll = self.base.log_prob(y, dt, self._base_theta(theta), z)
        return ll if self.k == 1 else self._mix(ll, theta)

    def log_prob_masked(
        self, y: Array, mask: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> Array:
        """As `log_prob`, with unobserved series contributing 0 to each component (`ProductEmission` rule).

        A base without per-series masking drops the whole step when any series is missing.
        """
        bt = self._base_theta(theta)
        if hasattr(self.base, "log_prob_masked"):
            ll = self.base.log_prob_masked(y, mask, dt, bt, z)
        else:
            ll = jnp.where(jnp.all(mask, axis=-1)[:, None], self.base.log_prob(y, dt, bt, z), 0.0)
        return ll if self.k == 1 else self._mix(ll, theta)

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` weighted sum of the component CDFs."""
        c = self.base.cdf(y, dt, self._base_theta(theta), z)  # (T, N*K, M)
        if self.k == 1:
            return c  # type: ignore[no-any-return]
        c4 = c.reshape(c.shape[0], -1, self.k, c.shape[-1])
        return jnp.sum(theta[WEIGHTS][None, :, :, None] * c4, axis=2)

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """Component `k ~ Categorical(w_i)`, then a draw of the base family's pseudo-state `i K + k`."""
        bt = self._base_theta(theta)
        if self.k == 1:
            return self.base.sample(key, regime, dt, bt, z)  # type: ignore[no-any-return]
        k_comp, k_obs = jax.random.split(key)
        comp = jax.random.categorical(k_comp, self._log_w(theta)[regime]).astype(regime.dtype)
        return self.base.sample(k_obs, regime * self.k + comp, dt, bt, z)  # type: ignore[no-any-return]

    def moments(
        self, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> tuple[Array, Array] | None:
        """Mixture mean `sum w mu_k`, variance `sum w (var_k + mu_k^2) - mean^2` (`None`: base has none)."""
        bt = self._base_theta(theta)
        if self.k == 1:
            return self.base.moments(regime, dt, bt, z)  # type: ignore[no-any-return]
        parts = [self.base.moments(regime * self.k + c, dt, bt, z) for c in range(self.k)]
        if any(p is None for p in parts):
            return None
        w = theta[WEIGHTS][regime]  # (T, K)
        means = jnp.stack([p[0] for p in parts if p is not None], axis=1)  # (T, K, M)
        varis = jnp.stack([p[1] for p in parts if p is not None], axis=1)
        mean = jnp.sum(w[:, :, None] * means, axis=1)
        second = jnp.sum(w[:, :, None] * (varis + means**2), axis=1)
        return mean, second - mean**2

    def init_carry(self) -> object:
        """No path-dependent state."""
        return None

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Identity."""
        del y_t, theta
        return carry

    # --- estimation hooks ---------------------------------------------------------------------------------

    def sufficient_stats(
        self, y: Array, dt: Array, gamma: Array, z: Array | None = None, theta: dict[str, Array] | None = None
    ) -> dict[str, Any] | None:
        """Responsibilities `r_tik = gamma_ti softmax_k(log w_ik + log g_k(y_t))` and pooled base statistics.

        Returns `{"resp": (T, N, K), "r_sum": (N, K), "base": base.sufficient_stats(y, dt, r (T, N K), z)}`.
        The responsibilities depend on the current parameters, so `theta` is required (`None` without it);
        `K == 1` is the base family's statistics.
        """
        if self.k == 1:
            out: dict[str, Any] | None = self.base.sufficient_stats(y, dt, gamma, z)
            return out
        if theta is None:
            return None
        bt = self._base_theta(theta)
        ll = self.base.log_prob(y, dt, bt, z).reshape(y.shape[0], -1, self.k)
        resp = gamma[:, :, None] * jax.nn.softmax(ll + self._log_w(theta)[None], axis=-1)  # (T, N, K)
        base_stats = self.base.sufficient_stats(y, dt, resp.reshape(y.shape[0], -1), z)
        return {"resp": resp, "r_sum": jnp.sum(resp, axis=0), "base": base_stats}

    def m_step(self, stats: object) -> dict[str, Array] | None:
        """`w_ik = sum_t r_tik / sum_k sum_t r_tik` and the base M-step on the pooled statistics.

        With an ordering constraint the components of each state are sorted by the first ordered parameter
        (weights and every base parameter are permuted together). `None` if the base has no M-step.
        """
        if self.k == 1:
            return self.base.m_step(stats)  # type: ignore[no-any-return]
        if not isinstance(stats, dict) or stats.get("base") is None:
            return None
        base_hat = self.base.m_step(stats["base"])
        if base_hat is None:
            return None
        r_sum = stats["r_sum"]
        total = jnp.sum(r_sum, axis=-1, keepdims=True)
        weights = jnp.where(total > 0, r_sum / jnp.where(total > 0, total, 1.0), 1.0 / self.k)
        n = r_sum.shape[0]
        out = {key: (v.reshape(n, self.k) if key in self.ordered else v) for key, v in base_hat.items()}
        if self.ordered and self.ordered[0] in out:
            perm = jnp.argsort(out[self.ordered[0]], axis=1)  # (N, K)
            out = {
                key: _gather_components(v, perm, n, self.k, rows=key in self.ordered)
                for key, v in out.items()
            }
            weights = jnp.take_along_axis(weights, perm, axis=1)
        out[WEIGHTS] = weights
        return out

    def conjugate_prior(self) -> Any | None:  # noqa: ANN401
        """The base prior for `K == 1`; else a Dirichlet on the weights times the base prior (if any)."""
        base_prior = self.base.conjugate_prior()
        if base_prior is None or self.k == 1:
            return base_prior
        return MixturePrior(base_prior)

    # --- state relabelling (`model.lattice.post_sort_states`) ---------------------------------------

    def state_sort_key(self, theta: dict[str, Array], by: str) -> Array:
        """`(N,)` key of each state: the weighted mean over its components of `theta[by]` (first series)."""
        arr = theta[by]
        comp = arr if by in self.ordered else arr.reshape(arr.shape[0], -1)[:, 0].reshape(-1, self.k)
        return jnp.sum(theta[WEIGHTS] * comp, axis=-1) if self.k > 1 else comp[:, 0]

    def permute_state_axis(self, name: str, value: Array, perm: Array) -> Array:
        """Relabel the states of one emission parameter by `perm` (`new = perm[...]`): blocks of `K` rows."""
        if self.k == 1 or name == WEIGHTS or name in self.ordered:
            return value[perm]
        n = perm.shape[0]
        return value.reshape(n, self.k, *value.shape[1:])[perm].reshape(value.shape)


def _gather_components(v: Array, perm: Array, n: int, k: int, *, rows: bool) -> Array:
    """Permute the components of each state: `v` is `(N, K)` (`rows`) or `(N*K, ...)`."""
    shaped = v if rows else v.reshape(n, k, *v.shape[1:])
    idx = jnp.broadcast_to(perm.reshape(n, k, *([1] * (shaped.ndim - 2))), shaped.shape)
    out = jnp.take_along_axis(shaped, idx, axis=1)
    return out if rows else out.reshape(v.shape)


def _weights_spec(n_states: int, k: int, cfg: ParamCfg | None) -> ParamSpec:
    """`emission.weights (N, K)`: rows on the simplex, free (or fixed through `params.weights`)."""
    init = np.full((n_states, k), 1.0 / k)
    mode: Literal["free", "fixed", "tied", "bounded", "prior"] = "free"
    value = None
    if cfg is not None:
        if cfg.mode not in ("free", "fixed"):
            raise ConfigError(
                f"params.weights supports mode 'free' or 'fixed', not {cfg.mode!r}",
                path="model.emission.params.weights.mode",
            )
        mode = cfg.mode
        raw = cfg.values if mode == "fixed" else cfg.init
        if raw is not None:
            if isinstance(raw, str):
                raise ConfigError(
                    "weights init cannot be a string", path="model.emission.params.weights.init"
                )
            flat = np.atleast_1d(np.asarray(raw, dtype=np.float64))
            if flat.size == k:
                arr = np.tile(flat, (n_states, 1))
            elif flat.size == n_states * k:
                arr = flat.reshape(n_states, k)
            else:
                raise ConfigError(
                    f"weights need {k} (or {n_states * k}) entries, got {flat.size}",
                    path="model.emission.params.weights",
                )
            if bool(np.any(arr <= 0)):
                raise ConfigError("weights must be positive", path="model.emission.params.weights")
            arr = arr / arr.sum(axis=1, keepdims=True)
            init = arr
            value = arr if mode == "fixed" else None
        elif mode == "fixed":
            raise ConfigError(
                "fixed weights need params.weights.values", path="model.emission.params.weights"
            )
    return ParamSpec(
        name=WEIGHTS, shape=(n_states, k), domain="simplex_ref", mode=mode, value=value, init=init
    )


def _row_ordered(spec: ParamSpec, n_states: int, k: int) -> ParamSpec:
    """The flat `increasing` base spec `(N*K,)` as an `(N, K)` spec with every row increasing."""
    if spec.domain not in ("real", "positive"):
        raise ConfigError(
            f"constraint 'increasing' on a {spec.domain!r} parameter is not supported inside a mixture",
            path=spec.name,
        )
    if spec.mode in ("fixed", "tied") or not isinstance(spec.mode, str):
        raise ConfigError("an ordered mixture parameter must be free (or prior)", path=spec.name)
    init = None if spec.init is None else np.asarray(spec.init).reshape(n_states, k)
    if init is not None and bool(np.any(np.diff(init, axis=1) <= 0)):
        raise ConfigError(
            "the init of an increasing mixture parameter must increase along the components", path=spec.name
        )
    return ParamSpec(
        name=spec.name,
        shape=(n_states, k),
        domain="increasing" if spec.domain == "real" else "increasing_positive",
        mode=spec.mode,
        init=init,
        prior=spec.prior,
    )


def _expand_component_axis(cfg: ParamCfg, n_states: int, k: int) -> ParamCfg:
    """A per-component list (`K` entries) is repeated for every state (`N * K` entries, state-major)."""
    update: dict[str, Any] = {}
    for key in ("init", "values", "init_mean_sojourn_steps"):
        v = getattr(cfg, key, None)
        if isinstance(v, list) and len(v) == k and k != n_states * k:
            update[key] = list(v) * n_states
    return cfg.model_copy(update=update) if update else cfg
