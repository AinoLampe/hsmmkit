"""`TransitionModel`: jump matrix, multinomial-logit covariate link, generator mode (E3-T05).

`MemoryLayout` lives here per CLAUDE.md §3.1 ("E3-T05 left `covariates.py` open" -- no such module
is created in 0.1.0). Concept refs: §4.3 Geometric/Generator, §4.4 Transitions, §4.5, Appendix E.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from regimekit.model.params import DirichletPrior, NormalPrior, Params
from regimekit.model.params.spec import Mode, ParamSpec
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.config.schema import TransitionCfg


@dataclass(frozen=True, eq=False)
class MemoryLayout:
    """Maps each admissible memory tuple (row `r`) to the regime whose exit it models.

    `tuples (R, order)`: admissible memory tuples (order 1: `R = N`, tuple `(i,)`; order 2:
    `R = N(N-1)`, tuples `(i, j)` with `i != j`; adjacent entries always differ). `current_of (R,)`
    is each tuple's last element (the regime whose exit is being modelled). `target_mask (R, N)`
    is `True` where `target != current` (`bool`).
    """

    n_states: int
    order: int
    tuples: np.ndarray
    current_of: np.ndarray
    target_mask: np.ndarray

    def __eq__(self, other: object) -> bool:
        """Structural equality, safe for numpy array fields.

        Needed once `StateSpace` (E4-T02a) stores a `MemoryLayout` as `eqx.field(static=True)`
        (CLAUDE.md §5's numpy-unhashable gotcha).
        """
        if not isinstance(other, MemoryLayout):
            return NotImplemented
        return (
            self.n_states == other.n_states
            and self.order == other.order
            and np.array_equal(self.tuples, other.tuples)
            and np.array_equal(self.current_of, other.current_of)
            and np.array_equal(self.target_mask, other.target_mask)
        )

    def __hash__(self) -> int:
        """Content-based hash, consistent with `__eq__`."""
        return hash((self.n_states, self.order, self.tuples.tobytes(), self.current_of.tobytes()))


def build_memory_layout(n_states: int, order: int) -> MemoryLayout:
    """Enumerate every length-`order` sequence of regimes with no two adjacent entries equal."""
    if order < 1:
        raise ConfigError(f"transition.order must be >= 1, got {order}", path="model.transition.order")
    tuples: list[tuple[int, ...]] = [(i,) for i in range(n_states)]
    for _ in range(order - 1):
        tuples = [(*t, j) for t in tuples for j in range(n_states) if j != t[-1]]
    arr = np.asarray(tuples, dtype=np.int64)
    current_of = arr[:, -1]
    target_mask = current_of[:, None] != np.arange(n_states)[None, :]
    return MemoryLayout(
        n_states=n_states, order=order, tuples=arr, current_of=current_of, target_mask=target_mask
    )


def _target_order(current_of: np.ndarray, n_states: int) -> np.ndarray:
    """`(R, N-1)`: for each row, the targets `j != current`, ascending (`[:, 0]` is the reference)."""
    return np.stack([np.array([j for j in range(n_states) if j != c], dtype=np.int64) for c in current_of])


def _target_onehot(target_idx: np.ndarray, n_states: int) -> np.ndarray:
    """`(R, N-1, N)`: one-hot scatter matrix from target-order position to the full state axis."""
    return np.asarray(np.eye(n_states, dtype=np.float64)[target_idx])


class TransitionModel(Protocol):
    """Structural type for the jump/generator transition model."""

    mode: Literal["jump", "generator"]

    def param_specs(self, n_states: int, order: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """The `ParamSpec`s this model contributes for a model of `n_states` regimes/`order` memory."""
        ...

    def jump_probs(self, theta: Params, z_pi: Array | None, memory: MemoryLayout) -> Array:
        """`(T, R, N)`: row `r` sums to 1 with exactly zero on `memory.current_of[r]`."""
        ...

    def jump_probs_t(self, theta: Params, z_pi_t: Array | None, memory: MemoryLayout) -> Array:
        """Per-step wrapper of `jump_probs` (`(R, N)`), for use inside a `jax.lax.scan` body."""
        ...


def _cov_idx_from_names(names: list[str], cov_names: tuple[str, ...], path: str) -> tuple[int, ...]:
    try:
        return tuple(cov_names.index(c) for c in names)
    except ValueError as exc:
        raise ConfigError(str(exc), path=path) from exc


@dataclass(frozen=True)
class JumpTransition:
    """Multinomial-logit jump matrix: rows are memory tuples, reference target is the smallest `j != i`.

    No covariates: `transition.pi_rows (R, N-1)` is a `simplex_ref`-domain spec (the row itself,
    reference-dropped bijector -- `N-2` free entries per row, matching Appendix B). With
    covariates: `transition.a (R, N-2)` / `transition.b (R, N-2, K_pi)` are logits for the
    non-reference targets, softmax-combined with an implicit zero logit at the reference.
    """

    mode: Literal["jump", "generator"] = "jump"
    cov_idx: tuple[int, ...] = ()
    dirichlet_alpha: float | None = None
    """Set by `from_config` when covariates are configured with a Dirichlet prior: `ParamManager`'s
    per-spec prior (`prior.log_prob(theta[name])`) can't express "Dirichlet on `pi` at `z = 0`" since
    `pi` isn't `theta["transition.a"/"transition.b"]` directly; `log_prior_extra` computes it instead
    (a hook for `estimation/objective.py`, E5-T01, to add to the objective's `log_prior` term)."""

    def param_specs(self, n_states: int, order: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """Default (config-free) spec: `transition.pi_rows`, free/`simplex_ref`."""
        del cov_names
        layout = build_memory_layout(n_states, order)
        r = layout.tuples.shape[0]
        shape = (r, n_states - 1)
        return (
            ParamSpec(
                name="transition.pi_rows",
                shape=shape,
                domain="simplex_ref",
                mode="free",
                init=np.full(shape, 1.0 / (n_states - 1)),
            ),
        )

    @classmethod
    def from_config(
        cls, cfg: TransitionCfg, n_states: int, order: int, cov_names: tuple[str, ...]
    ) -> tuple[JumpTransition, tuple[ParamSpec, ...]]:
        """Map `model.transition` config to a configured `JumpTransition` and its `ParamSpec`s."""
        cov_idx = _cov_idx_from_names(cfg.covariates, cov_names, "model.transition.covariates")
        layout = build_memory_layout(n_states, order)
        r = layout.tuples.shape[0]
        k_pi = len(cov_idx)
        prior_cfg = cfg.prior if (cfg.prior is not None and cfg.prior.type == "dirichlet") else None
        if k_pi == 0:
            family = cls(mode=cfg.mode, cov_idx=cov_idx)
            shape = (r, n_states - 1)
            prior = DirichletPrior(alpha=prior_cfg.alpha or 1.0) if prior_cfg is not None else None
            mode: Mode = "prior" if prior is not None else "free"
            spec = ParamSpec(
                name="transition.pi_rows",
                shape=shape,
                domain="simplex_ref",
                mode=mode,
                init=np.full(shape, 1.0 / (n_states - 1)),
                prior=prior,
            )
            return family, (spec,)
        dirichlet_alpha = (prior_cfg.alpha or 1.0) if prior_cfg is not None else None
        family = cls(mode=cfg.mode, cov_idx=cov_idx, dirichlet_alpha=dirichlet_alpha)
        coef_prior = NormalPrior(sd=cfg.coef_prior_sd)
        a_shape = (r, n_states - 2)
        b_shape = (r, n_states - 2, k_pi)
        a_spec = ParamSpec(
            name="transition.a", shape=a_shape, domain="real", mode="free", init=np.zeros(a_shape)
        )
        b_spec = ParamSpec(
            name="transition.b",
            shape=b_shape,
            domain="real",
            mode="free",
            init=np.zeros(b_shape),
            prior=coef_prior,
        )
        return family, (a_spec, b_spec)

    def log_prior_extra(self, theta: Params, memory: MemoryLayout) -> Array:
        """`DirichletPrior(dirichlet_alpha).log_prob(pi at z=0)`, or `0.` when not configured.

        Not wired into `ParamManager.log_prior` (which only evaluates a spec's prior on its own
        `theta[name]`); callers building the full objective (`estimation/objective.py`, E5-T01) add
        this term explicitly when `dirichlet_alpha is not None`.
        """
        if self.dirichlet_alpha is None:
            return jnp.zeros(())
        n_states = memory.n_states
        z0 = jnp.zeros((len(self.cov_idx),))
        pi_full = self.jump_probs_t(theta, z0, memory)  # (R, N)
        target_idx = jnp.asarray(_target_order(memory.current_of, n_states))  # (R, N-1)
        pi_targets = jnp.take_along_axis(pi_full, target_idx, axis=1)  # (R, N-1)
        return DirichletPrior(alpha=self.dirichlet_alpha).log_prob(pi_targets)

    def jump_probs(self, theta: Params, z_pi: Array | None, memory: MemoryLayout) -> Array:
        """`(T, R, N)` jump probabilities."""
        n_states = memory.n_states
        r = memory.current_of.shape[0]
        target_idx = _target_order(memory.current_of, n_states)
        onehot = jnp.asarray(_target_onehot(target_idx, n_states))  # (R, N-1, N)

        if "transition.pi_rows" in theta:
            t = z_pi.shape[0] if z_pi is not None else 1
            pi = jnp.broadcast_to(theta["transition.pi_rows"][None, :, :], (t, r, n_states - 1))
        else:
            a, b = theta["transition.a"], theta["transition.b"]
            t = z_pi.shape[0] if z_pi is not None else 1
            z = (
                z_pi[:, jnp.asarray(self.cov_idx)]
                if (z_pi is not None and self.cov_idx)
                else jnp.zeros((t, 0))
            )
            logits_nonref = a[None, :, :] + jnp.einsum("rok,tk->tro", b, z)  # (T, R, N-2)
            logits = jnp.concatenate([jnp.zeros((t, r, 1)), logits_nonref], axis=-1)  # (T, R, N-1)
            pi = jax.nn.softmax(logits, axis=-1)

        return jnp.einsum("tro,ron->trn", pi, onehot)

    def jump_probs_t(self, theta: Params, z_pi_t: Array | None, memory: MemoryLayout) -> Array:
        """`(R, N)` jump probabilities at a single step."""
        n_states = memory.n_states
        r = memory.current_of.shape[0]
        target_idx = _target_order(memory.current_of, n_states)
        onehot = jnp.asarray(_target_onehot(target_idx, n_states))  # (R, N-1, N)

        if "transition.pi_rows" in theta:
            pi = theta["transition.pi_rows"]  # (R, N-1)
        else:
            a, b = theta["transition.a"], theta["transition.b"]
            z = (
                z_pi_t[jnp.asarray(self.cov_idx)]
                if (z_pi_t is not None and self.cov_idx)
                else jnp.zeros((0,))
            )
            logits_nonref = a + b @ z  # (R, N-2)
            logits = jnp.concatenate([jnp.zeros((r, 1)), logits_nonref], axis=-1)
            pi = jax.nn.softmax(logits, axis=-1)

        return jnp.einsum("ro,ron->rn", pi, onehot)


def transition_matrix_generator(theta: Params, lam: Array, dt: Array) -> Array:
    """`(T, N, N)`: the CTMC transition matrix `expm(Q * dt_t)` for the geometric/HMM special case.

    `Q_ij = lambda_i * pi_ij` (`i != j`), `Q_ii = -lambda_i`; `pi` is `JumpTransition.jump_probs`
    at `order = 1` (each memory row is a single current regime), with no covariates in scope here
    (HMM-only generator mode, per the ticket).
    """
    n_states = lam.shape[0]
    memory = build_memory_layout(n_states, order=1)
    pi = JumpTransition().jump_probs(theta, None, memory)[0]  # (N, N)
    q_off = pi * lam[:, None]
    q = q_off - jnp.diag(jnp.sum(q_off, axis=1))
    return jnp.asarray(jax.vmap(lambda dt_t: jax.scipy.linalg.expm(q * dt_t))(dt))
