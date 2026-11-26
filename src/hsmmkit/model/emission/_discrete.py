"""Shared machinery of the discrete emission families (E11-T04): a linear predictor with covariate links.

`eta_t = eta_i + gamma_i . z_t` per regime `i` (and series `m`); the subclasses map it to a mean (Poisson,
Bernoulli) or to category logits (Categorical). The columns of `z^e` named in `emission.covariates.mean`
feed the link; the parameters are `emission.eta` and `emission.gamma`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np

from hsmmkit.model.emission.base import BaseEmission
from hsmmkit.model.params.priors import DEFAULT_COEF_PRIOR
from hsmmkit.model.params.spec import ParamSpec
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

CLIP = 1e-12
"""Closed-form M-steps clip probabilities and rates to `[CLIP, 1 - CLIP]` (rates: `>= CLIP`)."""


class DiscreteBase(BaseEmission):
    """Linear-predictor mixin: `eta (N, J)` and `gamma (N, J, K_e)` (`J`: series, or categories - 1)."""

    series_param_names: ClassVar[tuple[str, ...]] = ("emission.eta", "emission.gamma")
    mean_idx: tuple[int, ...] = ()

    def _linear(self, theta: dict[str, Array], z: Array | None, t: int) -> Array:
        """`(T, N, J)` linear predictor `eta + gamma . z`."""
        eta = theta["emission.eta"]  # (N, J)
        out = jnp.broadcast_to(eta[None], (t, *eta.shape))
        if self.mean_idx and "emission.gamma" in theta and z is not None:
            out = out + jnp.einsum("tk,njk->tnj", z[:, jnp.asarray(self.mean_idx)], theta["emission.gamma"])
        return out

    def init_carry(self) -> object:
        """No path-dependent state."""
        return None

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Identity."""
        del y_t, theta
        return carry


def eta_spec(n_states: int, width: int, init: np.ndarray | None, prior: object | None = None) -> ParamSpec:
    """`emission.eta (N, J)`: real, free."""
    return ParamSpec(
        name="emission.eta",
        shape=(n_states, width),
        domain="real",
        mode="free",
        init=np.zeros((n_states, width)) if init is None else init,
        prior=prior,  # type: ignore[arg-type]
    )


def gamma_spec(n_states: int, width: int, n_coef: int) -> ParamSpec:
    """`emission.gamma (N, J, K_e)`: free, `Normal(0, 1)`-regularized."""
    return ParamSpec(
        name="emission.gamma",
        shape=(n_states, width, n_coef),
        domain="real",
        mode="free",
        init=np.zeros((n_states, width, n_coef)),
        prior=DEFAULT_COEF_PRIOR,
    )


def natural_init(
    values: list[float] | float | str | None, n_states: int, width: int, path: str
) -> np.ndarray | None:
    """`init` of a params entry (one value per state, or one for all) as an `(N, J)` array (or `None`)."""
    if values is None:
        return None
    if isinstance(values, str):  # `init: labels`: the `labels` init method fills it (E12-T06)
        return None
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if arr.shape == (1,):
        arr = np.full(n_states, float(arr[0]))
    if arr.shape != (n_states,):
        raise ConfigError(f"expected {n_states} per-state values, got shape {arr.shape}", path=path)
    return np.tile(arr[:, None], (1, width))


def safe_ratio(num: Array, den: Array) -> Array:
    """`num / den` where `den > 0`, else 0 (an empty regime ends at the clipping floor)."""
    ok = den > 0
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def check_free(cfg_param: object, path: str) -> None:
    """The discrete families estimate their parameters freely (`mode: free`) in 0.2.0."""
    mode = getattr(cfg_param, "mode", "free")
    if mode not in ("free",):
        raise ConfigError(f"only mode 'free' is supported, not {mode!r}", path=f"{path}.mode")
    for key in ("values", "tie", "bounds", "prior", "init_mean_sojourn_steps"):
        if getattr(cfg_param, key, None) is not None:
            raise ConfigError(f"'{key}' is not supported for this family", path=f"{path}.{key}")
    if getattr(cfg_param, "constraint", "none") != "none":
        raise ConfigError("'constraint' is not supported for this family", path=f"{path}.constraint")
