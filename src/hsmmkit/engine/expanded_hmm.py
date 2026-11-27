"""`ExpandedHMMEngine`: log-space forward recursion over the expanded state space (E4-T03a).

Concept refs: §4.7.2 (forward), Appendix C, §6.3. Index convention (CLAUDE.md §3.7): code index `t`
(0-based) is math step `t+1`; `A_t` moves from code index `t-1` to `t`, covariate row `t` is the
lagged value usable at step `t`.

The recursion is one `jax.lax.scan` over `t = 1..T-1` whose body builds `log A_t` from
`space.transition_fn` -- `(T, S, S)` is never materialized. `spec` is a static jit argument,
`space` and `data` are pytrees.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from hsmmkit.engine.base import Posterior
from hsmmkit.engine.constraints import apply_transition_mask, boundary_code, check_feasible, label_log_mask
from hsmmkit.statespace.builders import initial_log_rho
from hsmmkit.utils.errors import NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace


def _safe_logsumexp(x: Array, axis: int) -> Array:
    """`logsumexp` that returns `-inf` with a zero (not NaN) gradient on an all-`-inf` slice.

    A phase-`k > 1` state has `alpha = -inf` until the chain has had time to reach it, so its
    incoming column can be entirely `-inf` (CLAUDE.md §5).
    """
    m = jnp.max(x, axis=axis, keepdims=True)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    total = jnp.sum(jnp.exp(x - m_safe), axis=axis)
    total_safe = jnp.where(total > 0, total, 1.0)
    out = jnp.squeeze(m_safe, axis=axis) + jnp.log(total_safe)
    # NaN inputs must propagate (invariant 5), not collapse to -inf.
    return jnp.where(total > 0, out, jnp.where(jnp.isnan(total), jnp.nan, -jnp.inf))


def _block(x: Array | None, t: int) -> Array:
    """A covariate block, or `(T, 0)` zeros when absent (CLAUDE.md §3.7)."""
    return jnp.zeros((t, 0)) if x is None else x


_Xs = tuple[Array, Array, Array, Array, Array]  # (ell, z_pi, z_d, dt, boundary code), sliced from step 1


def _log_a(
    space: StateSpace, theta: Params, z_pi_t: Array, z_d_t: Array, dt_t: Array, code_t: Array, active: bool
) -> Array:
    """`log A_t`; with boundary constraints (`active`, static) exits are kept / dropped (E12-T06)."""
    log_a = space.transition_fn(theta, z_pi_t, z_d_t, dt_t)[1]
    if not active:
        return log_a
    regime_of = space.index.regime_of
    return apply_transition_mask(log_a, regime_of[:, None] != regime_of[None, :], code_t)


def _regime_log_emis(theta: Params, spec: ModelSpec, data: PreparedDataset, dt: Array | None = None) -> Array:
    """`(T, N)` per-regime emission log-densities; masked steps contribute 0 (shared with E4-T06a).

    `dt` overrides `data.dt` (the explicit-duration engine uses `nominal_dt` everywhere).
    """
    dt = data.dt if dt is None else dt
    z_e = _block(data.cov_emission, data.observations.shape[0])
    return regime_log_emis(theta, spec, data.observations, data.mask, dt, z_e)


def regime_log_emis(theta: Params, spec: ModelSpec, y: Array, mask: Array, dt: Array, z_e: Array) -> Array:
    """Array form of `_regime_log_emis`: `y, mask (T, M)`, `dt (T,)`, `z_e (T, K_e)` -> `(T, N)`.

    With a `ProductEmission` (`M > 1`) the mask acts per series: an unobserved series contributes 0 and the
    observed ones keep their density. Single-series (bare family) models drop the step as before. The
    online filter (E15-T01) calls this with `T = 1` so both paths run the same code.
    """
    # Masked observations contribute zero log-density; sanitize `y` too so a NaN print cannot
    # leak NaN gradients through the untaken branch (double `where`).
    y_safe = jnp.where(mask, y, 0.0)
    if hasattr(spec.emission, "log_prob_masked"):  # `ProductEmission`, `MixtureFamily`
        # Multi-series model (E9-T01): each series is masked on its own, the others keep their density.
        masked: Array = spec.emission.log_prob_masked(y_safe, mask, dt, theta, z_e)
        return masked
    observed = jnp.all(mask, axis=-1)  # (T,)
    if spec.emission.init_carry() is not None:
        # Recurrent emission state (E9-T03, ADR-0022): the family threads its carry through a scan.
        log_emis = spec.emission.log_prob_scan(y_safe, mask, dt, theta, z_e)  # type: ignore[attr-defined]
    else:
        log_emis = spec.emission.log_prob(y_safe, dt, theta, z_e)  # (T, N)
    return jnp.where(observed[:, None], log_emis, 0.0)


def _expanded_log_emis(theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
    """`(T, S)` emission table of the expanded states (see `expanded_log_emis`)."""
    z_e = _block(data.cov_emission, data.observations.shape[0])
    return expanded_log_emis(theta, spec, space, data.observations, data.mask, data.dt, z_e, data.labels)


def expanded_log_emis(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    y: Array,
    mask: Array,
    dt: Array,
    z_e: Array,
    labels: Array | None = None,
) -> Array:
    """`(T, S)` emission table of the expanded states from plain arrays.

    Segment memory: the per-regime table gathered by `regime_of`. Time-step memory (`memory_mode: steps`): the
    emission is evaluated per tuple `(s_{t-p}, ..., s_t)`; masked steps contribute 0.
    """
    if spec.memory_mode != "steps":
        ell = jnp.take(regime_log_emis(theta, spec, y, mask, dt, z_e), space.index.regime_of, axis=1)
    else:
        tuples = jnp.concatenate([space.index.memory_of, space.index.regime_of[:, None]], axis=1)
        ell = spec.emission.log_prob_tuples(jnp.where(mask, y, 0.0), dt, theta, z_e, tuples)  # type: ignore[attr-defined]
        ell = jnp.where(jnp.all(mask, axis=-1)[:, None], ell, 0.0)
    if labels is not None:  # supervision: a label forbids the other regimes at that step (E12-T06)
        ell = jnp.where(label_log_mask(labels, space.index.regime_of), ell, -jnp.inf)
    return ell


def _emission_inputs(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array, Array, _Xs]:
    """`(log_rho, ell_0, ell, xs)`: initial log-distribution, step-0 emissions, and the scan inputs.

    `ell` is the `(T, S)` expanded emission table; `xs = (ell, z_pi, z_d, dt)` is sliced from step 1
    and shared by the forward, backward and Viterbi scans.
    """
    t_len = data.observations.shape[0]
    z_pi, z_d = _block(data.cov_transition, t_len), _block(data.cov_duration, t_len)
    ell = _expanded_log_emis(theta, spec, space, data)  # (T, S)
    log_rho = initial_log_rho(spec, space, theta, data.nominal_dt)
    code = boundary_code(data.boundaries, t_len)
    code_arr = jnp.zeros((t_len,), dtype=jnp.int8) if code is None else jnp.asarray(code)
    return log_rho, ell[0], ell, (ell[1:], z_pi[1:], z_d[1:], data.dt[1:], code_arr[1:])


def forward_step(
    space: StateSpace,
    theta: Params,
    active: bool,
    alpha_prev: Array,
    ell_t: Array,
    z_pi_t: Array,
    z_d_t: Array,
    dt_t: Array,
    code_t: Array,
) -> Array:
    """One forward step `log alpha_t = ell_t + logsumexp_i(log alpha_{t-1}[i] + log A_t[i, :])`.

    The single body of the offline scan and of `online.OnlineFilter.update`, so that both run the same
    floating-point operations in the same order (NFR-15, bitwise).
    """
    log_a = _log_a(space, theta, z_pi_t, z_d_t, dt_t, code_t, active)
    return ell_t + _safe_logsumexp(alpha_prev[:, None] + log_a, axis=0)


def _forward_impl(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array, _Xs]:
    """`(log_alpha, loglik, xs)`; `xs = (ell, z_pi, z_d, dt)` sliced from step 1, shared with backward."""
    log_rho, ell_0, _, xs = _emission_inputs(theta, spec, space, data)
    alpha_0 = log_rho + ell_0

    active = data.boundaries is not None

    def step(alpha_prev: Array, xs_t: _Xs) -> tuple[Array, Array]:
        alpha_t = forward_step(space, theta, active, alpha_prev, *xs_t)
        return alpha_t, alpha_t

    _, rest = jax.lax.scan(step, alpha_0, xs)
    log_alpha = jnp.concatenate([alpha_0[None, :], rest], axis=0)
    return log_alpha, _safe_logsumexp(log_alpha[-1], axis=0), xs


@partial(jax.jit, static_argnames=("spec",))
def _forward_core(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array]:
    log_alpha, loglik, _ = _forward_impl(theta, spec, space, data)
    return log_alpha, loglik


@partial(jax.jit, static_argnames=("spec",))
def _forward_backward_core(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array, Array, Array]:
    """`(log_alpha, log_beta, smoothed, loglik)`; `A_{t+1}` is rebuilt inside the reverse scan."""
    log_alpha, loglik, xs = _forward_impl(theta, spec, space, data)
    s = log_alpha.shape[1]

    active = data.boundaries is not None

    def step(beta_next: Array, xs_next: _Xs) -> tuple[Array, Array]:
        ell_next, z_pi_next, z_d_next, dt_next, code_next = xs_next
        log_a = _log_a(space, theta, z_pi_next, z_d_next, dt_next, code_next, active)
        beta_t = _safe_logsumexp(log_a + (ell_next + beta_next)[None, :], axis=1)
        return beta_t, beta_t

    beta_last = jnp.zeros((s,))
    _, rest = jax.lax.scan(step, beta_last, xs, reverse=True)  # rest[t] = beta_t, t = 0..T-2
    log_beta = jnp.concatenate([rest, beta_last[None, :]], axis=0)

    log_gamma = log_alpha + log_beta
    norm = _safe_logsumexp(log_gamma, axis=1)[:, None]
    finite = jnp.isfinite(log_gamma)
    smoothed = jnp.where(finite, jnp.exp(jnp.where(finite, log_gamma - norm, 0.0)), 0.0)
    return log_alpha, log_beta, smoothed, loglik


def _check_finite(loglik: Array) -> None:
    """Raise `NumericalError` on a non-finite loglik; skipped under tracing (grad/jit callers)."""
    if isinstance(loglik, jax.core.Tracer):
        return
    if not bool(jnp.isfinite(loglik)):
        raise NumericalError("non-finite log-likelihood from the forward recursion", path="loglik")


def _check_posterior(log_alpha: Array, log_beta: Array, smoothed: Array, loglik: Array) -> None:
    """Public-boundary guard: no NaN in the log arrays (`-inf` is structural), all else finite."""
    if isinstance(loglik, jax.core.Tracer):
        return
    _check_finite(loglik)
    for name, arr in (("log_alpha", log_alpha), ("log_beta", log_beta)):
        if bool(jnp.any(jnp.isnan(arr))):
            raise NumericalError(f"NaN in {name} from the forward-backward recursion", path=name)
    if not bool(jnp.all(jnp.isfinite(smoothed))):
        raise NumericalError("non-finite smoothed posterior", path="smoothed")


@dataclass(frozen=True)
class ExpandedHMMEngine:
    """HMM inference on the phase/memory-expanded state space (`spec.engine == "expanded"`).

    `parallel_scan` (`estimation.parallel_scan`, off by default) computes the smoothing pass of
    `forward_backward` with `jax.lax.associative_scan` (`engine/parallel.py`): `O(log T)` depth, `(T, S, S)`
    memory, used only while `T * S^2 <= 5e7` (otherwise a warning and the sequential scan).
    """

    parallel_scan: bool = False

    def forward(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> tuple[Array, Array]:
        """`(log_alpha (T, S), loglik)`; `loglik` is a 0-d `Array`.

        Raises:
            NumericalError: if `loglik` is not finite (checked outside jit, only when called with
                concrete arrays).
        """
        log_alpha, loglik = _forward_core(theta, spec, space, data)
        check_feasible(loglik, data, log_alpha)
        _check_finite(loglik)
        return log_alpha, loglik

    def loglik(self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
        """`log p(y_1:T)`, differentiable w.r.t. `theta`."""
        return self.forward(theta, spec, space, data)[1]

    def forward_backward(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> Posterior:
        """Forward then backward pass (sharing the emission table) as a `Posterior`.

        Raises:
            NumericalError: on a non-finite loglik/posterior (checked outside jit, only when called
                with concrete arrays).
        """
        log_alpha, log_beta, smoothed, loglik = self._smooth(theta, spec, space, data)
        check_feasible(loglik, data, log_alpha)
        _check_posterior(log_alpha, log_beta, smoothed, loglik)
        return Posterior(
            log_alpha=log_alpha, log_beta=log_beta, smoothed=smoothed, loglik=loglik, index=space.index
        )

    def viterbi(self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
        """The most likely expanded-state path `(T,)` (int); see `engine.viterbi.viterbi_path`."""
        from hsmmkit.engine.viterbi import viterbi_path

        return viterbi_path(theta, spec, space, data)

    def _smooth(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> tuple[Array, Array, Array, Array]:
        """`(log_alpha, log_beta, smoothed, loglik)`: sequential scan, or the parallel one if enabled."""
        if self.parallel_scan:
            from hsmmkit.engine import parallel

            cells = data.observations.shape[0] * space.n_expanded**2
            if cells <= parallel.MAX_CELLS:
                par: tuple[Array, Array, Array, Array] = parallel.parallel_forward_backward(
                    theta, spec, space, data
                )
                return par
            warnings.warn(
                f"parallel_scan needs T*S^2 = {cells:.3g} > {parallel.MAX_CELLS:.0e} cells; "
                "using the sequential scan",
                UserWarning,
                stacklevel=3,
            )
        seq: tuple[Array, Array, Array, Array] = _forward_backward_core(theta, spec, space, data)
        return seq

    def approx(self) -> None:
        """No approximate-inference plug-in for the expanded engine (FR-X01 hook)."""
        return None
