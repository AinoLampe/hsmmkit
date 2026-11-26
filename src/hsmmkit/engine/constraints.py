"""Supervision as engine masks: label and boundary constraints (E12-T06, ADR-0019).

Full or partial supervision is expressed as dynamic masks that the unchanged engines multiply into the joint
likelihood; rows are **not** renormalized, so a constrained model's likelihood is
`p(y_{1:T}, constraints | theta)`. Two kinds of constraints exist:

* **labels** `(T, N)` bool (`PreparedDataset.labels`): regime `j` is allowed at step `t` iff `labels[t, j]`;
  an unlabelled step is all `True`, a fully labelled step has one `True`;
* **boundaries** `must_switch_at t` (a regime change between steps `t-1` and `t`, `1 <= t <= T-1`) and
  `must_not_switch_at t`, kept as an int8 code per step (`CODE_SWITCH`, `CODE_STAY`, 0 = free).

Expanded engine: the emission table gets `-inf` where a label forbids the regime; the transition `A_t` keeps
only the exit entries (a regime change) under `must_switch_at t` and drops exactly those under
`must_not_switch_at t`. Explicit engine: a segment `(s, j, d)` is feasible iff the label run of regime `j`
from `s` is at least `d`, no `must_switch` step lies strictly inside it and it does not end just before a
`must_not_switch` step; the constraints are folded into the per-start duration table, so the recursions are
unchanged. Unconstrained data take the original code paths (bitwise identical results).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.utils.errors import ConstraintError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.contracts import BoundaryConstraints, PreparedDataset

__all__ = [
    "CODE_STAY",
    "CODE_SWITCH",
    "apply_transition_mask",
    "boundary_code",
    "check_feasible",
    "explicit_constrain",
    "label_log_mask",
]

CODE_SWITCH = 1
CODE_STAY = 2
_NEG_INF = -jnp.inf


def boundary_code(boundaries: BoundaryConstraints | None, t_len: int) -> np.ndarray | None:
    """`(T,)` int8 codes of a sequence's boundaries (`None` without boundaries)."""
    if boundaries is None or not (boundaries.must_switch_at or boundaries.must_not_switch_at):
        return None
    code = np.zeros(t_len, dtype=np.int8)
    code[list(boundaries.must_switch_at)] = CODE_SWITCH
    code[list(boundaries.must_not_switch_at)] = CODE_STAY
    return code


def label_log_mask(labels: Array, regime_of: Array) -> Array:
    """`(T, S)` bool: expanded state `s` is allowed at step `t` iff `labels[t, regime_of[s]]`."""
    return jnp.take(labels, regime_of, axis=1)


def apply_transition_mask(log_a: Array, change: Array, code_t: Array) -> Array:
    """`A_t` under the boundary code of step `t`: only exits (`must_switch`) / no exit (`must_not_switch`).

    `change (S, S)` is `True` where the regime changes (an exit transition).
    """
    switch = jnp.where(change, log_a, _NEG_INF)
    stay = jnp.where(change, _NEG_INF, log_a)
    return jnp.where(code_t == CODE_SWITCH, switch, jnp.where(code_t == CODE_STAY, stay, log_a))


def explicit_constrain(
    log_pmf: Array, allowed: Array | None, code: Array | None, t_len: int
) -> tuple[Array, Array, Array]:
    """Fold the constraints into the explicit engine's tables.

    Args:
        log_pmf: `(N, D)` or `(T, N, D)` duration log-pmf per segment start.
        allowed: `(T, N)` bool labels or `None`.
        code: `(T,)` boundary codes or `None`.
        t_len: `T`.

    Returns:
        `(log_pmf (T, N, D), term_ok (T, N), log_surv (T, N, D))`: segment `(s, j, d)` has `-inf` mass when
        infeasible; `term_ok[s, j]` says that a right-censored final segment of regime `j` from `s` (to step
        `T-1`) is feasible; `log_surv[s, j, d-1] = log P(duration >= d)` of the *unconstrained* pmf (the
        mass of
        longer segments still counts for a segment that is cut off), `-inf` where `d` exceeds the feasible
        length `min(label run, distance to the next must-switch step)`.
    """
    n, d_max = log_pmf.shape[-2], log_pmf.shape[-1]
    table = jnp.broadcast_to(log_pmf, (t_len, n, d_max)) if log_pmf.ndim == 2 else log_pmf
    starts = jnp.arange(t_len)
    ds = jnp.arange(1, d_max + 1)
    ok = jnp.ones((t_len, n, d_max), dtype=bool)
    reach = jnp.ones((t_len, n, d_max), dtype=bool)  # a segment can still be running after `d` steps
    term_ok = jnp.ones((t_len, n), dtype=bool)
    if allowed is not None:
        run = _label_run(allowed)  # (T, N): consecutive allowed steps from s
        ok = ok & (ds[None, None, :] <= run[:, :, None])
        reach = reach & (ds[None, None, :] <= run[:, :, None])
        term_ok = term_ok & ((t_len - starts)[:, None] <= run)
    if code is not None:
        switch = code == CODE_SWITCH
        # `nxt[s]`: first must-switch step after `s` (`T` if none); a segment from `s` ends before it
        nxt = _next_switch(switch)
        ok = ok & (ds[None, None, :] <= (nxt - starts)[:, None, None])
        reach = reach & (ds[None, None, :] <= (nxt - starts)[:, None, None])
        term_ok = term_ok & ((nxt >= t_len)[:, None])
        stay = jnp.concatenate([code == CODE_STAY, jnp.zeros((d_max + 1,), dtype=bool)])
        ends_before_stay = stay[
            starts[:, None] + ds[None, :]
        ]  # a segment ending at s+d-1 meets a no-switch step
        ok = ok & ~ends_before_stay[:, None, :]
    surv = jax.lax.cumlogsumexp(table, axis=2, reverse=True)
    return jnp.where(ok, table, _NEG_INF), term_ok, jnp.where(reach, surv, _NEG_INF)


def _label_run(allowed: Array) -> Array:
    """`(T, N)` int: number of consecutive steps from `s` on at which regime `j` is allowed."""

    def body(nxt: Array, ok_s: Array) -> tuple[Array, Array]:
        run = jnp.where(ok_s, nxt + 1, 0)
        return run, run

    _, runs = jax.lax.scan(body, jnp.zeros((allowed.shape[1],), dtype=jnp.int32), allowed, reverse=True)
    return runs


def _next_switch(switch: Array) -> Array:
    """`(T,)` int: the first `t > s` with `switch[t]`, else `T`."""
    t_len = switch.shape[0]

    def body(nxt: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        sw_next, s = x
        cur = jnp.where(sw_next, s + 1, nxt)
        return cur, cur

    sw_shift = jnp.concatenate([switch[1:], jnp.zeros((1,), dtype=bool)])
    _, out = jax.lax.scan(
        body,
        jnp.asarray(t_len, dtype=jnp.int32),
        (sw_shift, jnp.arange(t_len, dtype=jnp.int32)),
        reverse=True,
    )
    return out


def check_feasible(
    loglik: Array, data: PreparedDataset | None, log_alpha: Array | None, *, sequence: str | None = None
) -> None:
    """Raise `ConstraintError` if a constrained model's likelihood is exactly zero (outside jit only).

    `data` decides whether constraints exist (labels that are not all `True`, or boundaries); `log_alpha`
    (`(T, S)` expanded, or `None`) locates the first step at which every state has probability zero.
    """
    if isinstance(loglik, jax.core.Tracer) or data is None:
        return
    constrained = data.boundaries is not None or (data.labels is not None and not bool(jnp.all(data.labels)))
    if not constrained or bool(jnp.isfinite(loglik)):
        return
    step: int | None = None
    if log_alpha is not None:
        dead = np.flatnonzero(~np.asarray(jnp.any(jnp.isfinite(log_alpha), axis=-1)))
        step = int(dead[0]) if dead.size else None
    raise ConstraintError(
        "the label / boundary constraints admit no state path (zero likelihood)",
        path="data.labels",
        step=step,
        sequence=sequence,
    )
