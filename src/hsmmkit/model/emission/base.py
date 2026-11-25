"""`EmissionFamily` protocol (E3-T01; replaces the E1-T05 signature stub).

Concept refs: §4.2, §5.5, §5.10; Part II §15.7 amendment 4 (`support`, `scale_by_dt`).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from hsmmkit.model.params import Params, ParamSpec
from hsmmkit.utils.typing import Array

Support = Literal["real", "count", "binary", "categorical"]


class EmissionFamily(Protocol):
    """Structural type for emission families (GBM, Student-t, ...).

    Attributes:
        name: Config registry key (e.g. `"gbm"`).
        native_time: Whether the family's natural clock is continuous or discrete.
        support: Observation support, checked against `PreparedDataset.support` at model build.
        scale_by_dt: `True` when mean/variance scale with `dt` (e.g. GBM); documented per family.
    """

    name: str
    native_time: Literal["continuous", "discrete"]
    support: Support
    scale_by_dt: bool

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """The `ParamSpec`s this family contributes for a model of this shape."""
        ...

    def log_prob(self, y: Array, dt: Array, theta: Params, z: Array | None) -> Array:
        """`y (T, M)`, `dt (T,)`, `z (T, K_e)` or `None` -> `(T, N)` log-densities (summed over `M`)."""
        ...

    def cdf(self, y: Array, dt: Array, theta: Params, z: Array | None) -> Array:
        """`y (T, M)` -> `(T, N, M)` per-regime, per-series predictive CDF (E5-T04's PIT diagnostic)."""
        ...

    def sample(self, key: Array, regime: Array, dt: Array, theta: Params, z: Array | None) -> Array:
        """`regime (T,)` int, `dt (T,)` -> `(T, M)` samples."""
        ...

    def init_carry(self) -> object:
        """Initial recursive state for families with path-dependent emissions (`None` for GBM)."""
        ...

    def step_carry(self, carry: object, y_t: Array, theta: Params) -> object:
        """Advance `carry` by one observed step (identity for GBM)."""
        ...

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array, z: Array | None = None) -> object | None:
        """EM sufficient statistics for regime weights `gamma (T, N)` (`None` when the family has none)."""
        ...

    def m_step(self, stats: object) -> Params | None:
        """Closed-form maximizer of the expected complete-data log-likelihood (`None`: not available)."""
        ...

    def conjugate_prior(self) -> Any | None:  # noqa: ANN401
        """The conjugate prior family of the parameters (`None`: no conjugate prior is known)."""
        ...

    def expected_log_prob(self, y: Array, dt: Array, q_theta: object, z: Array | None) -> Array | None:
        """`E_q[log f_i(y_t)]`, `(T, N)`, for variational Bayes (`None`: not available)."""
        ...

    def moments(self, regime: Array, dt: Array, theta: Params, z: Array | None) -> tuple[Array, Array] | None:
        """Analytic `(mean, variance)`, each `(T, M)`, of the regime `(T,)` density (`None`: not analytic)."""
        ...


class BaseEmission:
    """Mixin with the optional estimation hooks of `EmissionFamily`; each returns `None` (not available).

    Families inherit it and override only what they implement; estimators discover capabilities by calling
    the hooks and checking for `None` (never by family name).
    """

    def m_step(self, stats: object) -> Params | None:
        """Not available."""
        del stats
        return None

    def conjugate_prior(self) -> Any | None:  # noqa: ANN401
        """Not available."""
        return None

    def expected_log_prob(self, y: Array, dt: Array, q_theta: object, z: Array | None) -> Array | None:
        """Not available."""
        del y, dt, q_theta, z
        return None

    def moments(self, regime: Array, dt: Array, theta: Params, z: Array | None) -> tuple[Array, Array] | None:
        """Not available."""
        del regime, dt, theta, z
        return None
