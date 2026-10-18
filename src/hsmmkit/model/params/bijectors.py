"""Bijectors mapping unconstrained `u` to constrained `theta`, with log-det-Jacobians (E1-T03a).

Each bijector is an `eqx.Module` with `forward(u) -> theta`, `inverse(theta) -> u`,
`log_det_jac(u) -> Array` (a 0-d array: `log |det ∂θ/∂u|` summed over the array), and a static
`domain` name. `simplex_ref`/`pmf`/`increasing` operate on a single 1-D vector (batching over
states is `ParamManager`'s job, E1-T03b); the other domains are elementwise over any shape.
"""

from __future__ import annotations

from typing import Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp

from regimekit.utils.typing import Array, Float


class Bijector(Protocol):
    """Structural type for the `forward`/`inverse`/`log_det_jac` bijector interface."""

    def forward(self, u: Array) -> Array:
        """Map unconstrained `u` to constrained `theta`."""
        ...

    def inverse(self, theta: Array) -> Array:
        """Map constrained `theta` back to unconstrained `u`."""
        ...

    def log_det_jac(self, u: Array) -> Array:
        """`log |det ∂θ/∂u|`, summed over the array."""
        ...


def _inverse_softplus(theta: Array) -> Array:
    """Stable inverse of `softplus`: `u = theta + log(-expm1(-theta))`, valid for `theta > 0`."""
    return theta + jnp.log(-jnp.expm1(-theta))


def _simplex_forward(u: Float[Array, " n_minus_1"]) -> Float[Array, " n"]:  # noqa: F722
    z = jnp.concatenate([jnp.zeros((1,), dtype=u.dtype), u])
    return jax.nn.softmax(z)


def _simplex_inverse(theta: Float[Array, " n"]) -> Float[Array, " n_minus_1"]:  # noqa: F722
    return jnp.log(theta[1:]) - jnp.log(theta[0])


def _simplex_log_det_jac(u: Float[Array, " n_minus_1"]) -> Array:  # noqa: F722
    def forward_free(u_: Array) -> Array:
        return _simplex_forward(u_)[1:]

    jac = jax.jacfwd(forward_free)(u)
    _, logabsdet = jnp.linalg.slogdet(jac)
    return jnp.asarray(logabsdet)


class IdentityBijector(eqx.Module):
    """`real` domain: identity."""

    domain: str = eqx.field(static=True, default="real")

    def forward(self, u: Array) -> Array:
        """Identity."""
        return u

    def inverse(self, theta: Array) -> Array:
        """Identity."""
        return theta

    def log_det_jac(self, u: Array) -> Array:
        """Zero (identity has unit Jacobian)."""
        return jnp.zeros(())


class PositiveBijector(eqx.Module):
    """`positive` domain: `softplus(u)` (default) or `exp(u)` (`variant="exp"`)."""

    variant: Literal["softplus", "exp"] = eqx.field(static=True, default="softplus")
    domain: str = eqx.field(static=True, default="positive")

    def forward(self, u: Array) -> Array:
        """`softplus(u)` or `exp(u)`."""
        if self.variant == "exp":
            return jnp.exp(u)
        return jax.nn.softplus(u)

    def inverse(self, theta: Array) -> Array:
        """Inverse of `softplus`/`exp`."""
        if self.variant == "exp":
            return jnp.log(theta)
        return _inverse_softplus(theta)

    def log_det_jac(self, u: Array) -> Array:
        """`sum(log_sigmoid(u))` (softplus) or `sum(u)` (exp)."""
        if self.variant == "exp":
            return jnp.sum(u)
        return jnp.sum(jax.nn.log_sigmoid(u))


class IntervalBijector(eqx.Module):
    """`interval[lo, hi]` domain: `lo + (hi - lo) * sigmoid(u)`."""

    lo: float = eqx.field(static=True)
    hi: float = eqx.field(static=True)
    domain: str = eqx.field(static=True, default="interval")

    def forward(self, u: Array) -> Array:
        """`lo + (hi - lo) * sigmoid(u)`."""
        return self.lo + (self.hi - self.lo) * jax.nn.sigmoid(u)

    def inverse(self, theta: Array) -> Array:
        """Logit of the rescaled `theta`."""
        p = (theta - self.lo) / (self.hi - self.lo)
        return jnp.log(p) - jnp.log1p(-p)

    def log_det_jac(self, u: Array) -> Array:
        """`sum(log(hi - lo) + log_sigmoid(u) + log_sigmoid(-u))`."""
        width = jnp.log(jnp.asarray(self.hi - self.lo))
        return jnp.sum(width + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))


class NuGt2Bijector(eqx.Module):
    """`nu_gt_2` domain: `2 + softplus(u)` (Student-t degrees of freedom)."""

    domain: str = eqx.field(static=True, default="nu_gt_2")

    def forward(self, u: Array) -> Array:
        """`2 + softplus(u)`."""
        return 2.0 + jax.nn.softplus(u)

    def inverse(self, theta: Array) -> Array:
        """Inverse of `softplus(theta - 2)`."""
        return _inverse_softplus(theta - 2.0)

    def log_det_jac(self, u: Array) -> Array:
        """`sum(log_sigmoid(u))` (the `+2` shift doesn't affect the Jacobian)."""
        return jnp.sum(jax.nn.log_sigmoid(u))


class SimplexRefBijector(eqx.Module):
    """`simplex_ref` domain: `softmax(concat([0], u))`, reference category first."""

    domain: str = eqx.field(static=True, default="simplex_ref")

    def forward(self, u: Float[Array, " n_minus_1"]) -> Float[Array, " n"]:  # noqa: F722
        """`softmax(concat([0], u))`."""
        return _simplex_forward(u)

    def inverse(self, theta: Float[Array, " n"]) -> Float[Array, " n_minus_1"]:  # noqa: F722
        """`log(theta[1:]) - log(theta[0])`."""
        return _simplex_inverse(theta)

    def log_det_jac(self, u: Float[Array, " n_minus_1"]) -> Array:  # noqa: F722
        """`slogdet` of the free-coordinate Jacobian (autodiff, cheap: called once per objective)."""
        return _simplex_log_det_jac(u)


class PmfBijector(eqx.Module):
    """`pmf` domain (nonparametric over `D_max`): same construction as `simplex_ref`."""

    domain: str = eqx.field(static=True, default="pmf")

    def forward(self, u: Float[Array, " n_minus_1"]) -> Float[Array, " n"]:  # noqa: F722
        """`softmax(concat([0], u))`."""
        return _simplex_forward(u)

    def inverse(self, theta: Float[Array, " n"]) -> Float[Array, " n_minus_1"]:  # noqa: F722
        """`log(theta[1:]) - log(theta[0])`."""
        return _simplex_inverse(theta)

    def log_det_jac(self, u: Float[Array, " n_minus_1"]) -> Array:  # noqa: F722
        """`slogdet` of the free-coordinate Jacobian (autodiff, cheap: called once per objective)."""
        return _simplex_log_det_jac(u)


class IncreasingBijector(eqx.Module):
    """`increasing` domain: an ordering constraint over a vector.

    `theta_0 = base.forward(u_0)`, `theta_i = theta_{i-1} + softplus(u_i)` for `i >= 1`.
    """

    base: Bijector
    domain: str = eqx.field(static=True, default="increasing")

    def forward(self, u: Float[Array, " n"]) -> Float[Array, " n"]:  # noqa: F722
        """Cumulative-softplus construction, strictly increasing in `theta`."""
        theta0 = self.base.forward(u[0])
        steps = jax.nn.softplus(u[1:])
        return jnp.concatenate([theta0[None], theta0 + jnp.cumsum(steps)])

    def inverse(self, theta: Float[Array, " n"]) -> Float[Array, " n"]:  # noqa: F722
        """Undo the cumulative-softplus construction."""
        u0 = self.base.inverse(theta[0])
        diffs = jnp.diff(theta)
        return jnp.concatenate([u0[None], _inverse_softplus(diffs)])

    def log_det_jac(self, u: Float[Array, " n"]) -> Array:  # noqa: F722
        """`base.log_det_jac(u[0]) + sum(log_sigmoid(u[1:]))` (triangular Jacobian)."""
        return self.base.log_det_jac(u[0]) + jnp.sum(jax.nn.log_sigmoid(u[1:]))
