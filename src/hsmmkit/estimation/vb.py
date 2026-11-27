"""`VBEstimator`: mean-field variational Bayes, exact structured posterior over the state path (E13-T05a).

Concept refs: Part II §14.8, Appendix F (conjugates); Beal (2003). The posterior factorizes
`q(theta) prod_b q(s^(b))`; given `q(theta)` the optimal `q(s)` is the exact forward-backward posterior of the
chain whose log-weights are the *expected log-parameters* `E_q[log theta]`:

* emissions: `E_q[log f_i(y_t)]` (`EmissionFamily.expected_log_prob`: digamma / Wishart forms);
* phase advance: `E[log a_k]` and `E[log (1 - a_k)]` for `a_k ~ Beta` (geometric / negative-binomial
  durations at a constant step), jumps: `E[log pi_rj] = psi(alpha_rj) - psi(sum alpha_r)`;
* the weights are sub-normalized, so the forward normalizer `log Z~` is not a likelihood, and the statistics
  of the pass (`ExpandedEStats`, E13-T01a) are the expected counts under `q(s)`.

One iteration is: E-step on `E_q[log theta]` -> `ELBO = log Z~ - sum KL(q || prior)` ->
`q <- posterior(prior, counts)`.
The ELBO of the successive `q` is non-decreasing (coordinate ascent). Each run starts from a point estimate of
an initialization method (one E-step at that point gives the first `q`); restarts follow MAP; the run with
the best final ELBO wins. `FitResult.params` are the posterior means (constrained scale), `objective` is
`-ELBO`, `loglik` the log-likelihood at the posterior means, and `diagnostics["vb"]` holds `elbo_trace`
and `q_params`.

Supported: expanded engine, `initial_state: uniform`, `use_priors: true`. Each block (emission, phase advance,
jump rows) is treated as *conjugate* when its family has a conjugate prior, no covariate link, a constant step
and only free, untied, unordered parameters: Gaussian diagonal / full, Poisson, Bernoulli, Categorical
emissions,
geometric / negative-binomial durations, jump rows (also under a topology without absorbing regimes). Every
other block is a *point* block (VB-EM hybrid, E13-T05b): its parameters stay point estimates, updated by the
ECM
conditional step of E13-T02b on the current expected counts, its log-weights in the E-step are its log-density
at the point estimate, and its contribution to the ELBO is `log p(theta) + log|J|` at the point estimate (the
bound of a partially collapsed objective; it is still non-decreasing).

After convergence a regime whose expected occupancy `sum gamma / sum T` is below `estimation.vb.prune_below`
(default 0.01) is reported in `FitResult.pruned_states` and `diagnostics["vb"]["occupancy"]` (the model keeps
its structure; `decode` marks the pruned regimes) with a warning that suggests a smaller `n_states`.
`FitResult.std_errors` holds the posterior standard deviations of the conjugate blocks (constrained scale,
delta method for the transformed parameters) and `diagnostics["vb"]["treatment"]` maps each component to
`"conjugate"` or `"point"`.
Priors are the weakly informative defaults of
`docs/math/estimation.md` (Gaussian priors scaled to the data); the config `prior` entries are not read.
When `params` are renumbered by ascending scale (no identifiability guard), `occupancy`, `pruned_states`,
`std_errors` and `q_params` are renumbered with them (`q_params["jump"]` is dropped for order > 1).
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from hsmmkit.config.schema import EmCfg
from hsmmkit.engine.estep import _estep_batch
from hsmmkit.estimation import ecm
from hsmmkit.estimation.em import EMEstimator, Plan, _final_loglik
from hsmmkit.estimation.mstep import transition_mstep
from hsmmkit.model.conjugate import BetaConjugate, beta_kl, dirichlet_kl
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EstimationSection
    from hsmmkit.data.batch import PaddedBatch, SequenceBatch
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.engine.estep import ExpandedEStats
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

__all__ = ["Treatment", "VBEmission", "VBEstimator", "check_vb_supported", "run_vb", "vb_treatment"]

ELBO_SLACK = 1e-8  # decrease tolerated before debug mode raises (Appendix G)


@dataclass(frozen=True)
class Treatment:
    """Which blocks are conjugate (`True`) and which keep point estimates (`False`)."""

    emission: bool
    duration: bool
    transition: bool

    def table(self) -> dict[str, str]:
        """`{component: "conjugate" | "point"}`."""
        return {
            "emission": "conjugate" if self.emission else "point",
            "duration": "conjugate" if self.duration else "point",
            "transition": "conjugate" if self.transition else "point",
        }


@dataclass(frozen=True)
class VBEmission:
    """Emission wrapper: `E_q[log f]` from `theta["vb.q.<key>"]` (conjugate) or `log f(theta)` (point)."""

    family: Any
    keys: tuple[str, ...]
    conjugate: bool = True
    name: str = "vb"

    def init_carry(self) -> None:
        """No recurrent state."""
        return None

    def log_prob(self, y: Array, dt: Array, theta: Params, z: Array | None) -> Array:
        """`(T, N)` expected log-density under `q`, or the log-density at the point estimate."""
        if not self.conjugate:
            return jnp.asarray(self.family.log_prob(y, dt, theta, z))
        q = {k: theta[f"vb.q.{k}"] for k in self.keys}
        out = self.family.expected_log_prob(y, dt, q, z)
        assert out is not None
        return jnp.asarray(out)


def _plain(specs: tuple[Any, ...]) -> bool:
    """Every parameter free, untied and unordered (what a conjugate update can honour)."""
    return all(
        s.mode not in ("fixed", "tied", "bounded", "prior") and s.tie_group is None and s.constraint == "none"
        for s in specs
    )


def _block_specs(spec: ModelSpec, prefix: str) -> tuple[Any, ...]:
    return tuple(s for s in spec.param_specs if s.name.startswith(prefix))


def vb_treatment(spec: ModelSpec, pb: PaddedBatch) -> Treatment:
    """Decide per block: conjugate update or point estimate (VB-EM hybrid)."""
    dt = np.asarray(pb.dt)[np.asarray(pb.valid)]
    regular = bool(dt.size == 0 or np.allclose(dt, dt[0], rtol=1e-9, atol=0.0))
    prior = spec.emission.conjugate_prior()
    emission = (
        prior is not None
        and hasattr(prior, "kl")
        and hasattr(prior, "mean_params")
        and int(pb.cov_emission.shape[-1]) == 0
        and _plain(_block_specs(spec, "emission."))
        and not (
            int(pb.observations.shape[-1]) > 1
            and not getattr(spec.emission, "joint", False)
            and any(s.name == "emission.scale" and len(s.shape) < 2 for s in spec.param_specs)
        )
    )
    duration = (
        hasattr(spec.duration, "from_advance_prob")
        and not bool(getattr(spec.duration, "cov_idx", ()))
        and regular
        and _plain(_block_specs(spec, "duration."))
    )
    transition = (
        not bool(getattr(spec.transition, "cov_idx", ()))
        and not any(s.name == "transition.b" for s in spec.param_specs)
        and _plain(_block_specs(spec, "transition."))
    )
    return Treatment(bool(emission), bool(duration), bool(transition))


def check_vb_supported(spec: ModelSpec, pb: PaddedBatch) -> Treatment:
    """Raise `ConfigError` for the structures VB does not handle; return the block treatment."""
    if spec.engine != "expanded":
        raise ConfigError("VB supports the expanded engine only", path="model.engine")
    if spec.memory_mode != "segments" or spec.duration_conditioning != "current":
        raise ConfigError("VB needs segment memory with duration_conditioning 'current'", path="model")
    if spec.initial != "uniform":
        raise ConfigError("VB supports initial_state: uniform only", path="model.initial_state")
    if spec.duration_dependent_jumps or spec.left_censored:
        raise ConfigError("VB does not support duration-dependent jumps or left-censoring", path="model")
    topology = getattr(spec.transition, "topology", None)
    if topology is not None and topology.absorbing:
        raise ConfigError("VB does not support absorbing regimes", path="model.topology")
    if (
        hasattr(spec.emission, "blocks")
        or hasattr(spec.emission, "log_prob_masked")
        or spec.emission.init_carry() is not None
    ):
        raise ConfigError(
            "VB does not support product / mixture / recurrent emissions", path="model.emission"
        )
    return vb_treatment(spec, pb)


def _data_prior(prior: Any, pb: PaddedBatch) -> Any:  # noqa: ANN401
    """The conjugate prior centred on the data scale (Gaussian families); others unchanged."""
    if prior is None or not hasattr(prior, "with_data_scale"):
        return prior
    obs = np.asarray(pb.observations)
    keep = np.asarray(pb.valid & jnp.all(pb.mask, axis=-1))
    y = obs[keep]
    if y.shape[0] < 2:
        return prior
    if hasattr(prior, "w0"):  # Normal-Wishart: data covariance
        cov = np.atleast_2d(np.cov(y.T)) + 1e-9 * np.eye(y.shape[1])
        return prior.with_data_scale(jnp.asarray(y.mean(axis=0)), jnp.asarray(cov))
    return prior.with_data_scale(float(y.mean()), float(max(y.var(), 1e-12)))


class _Context:
    """The jitted E-step / ELBO / update of one fit (the static structure is closed over)."""

    def __init__(self, spec: ModelSpec, space: StateSpace, pb: PaddedBatch, treat: Treatment) -> None:
        self.spec, self.space, self.treat = spec, space, treat
        self.prior: Any = _data_prior(spec.emission.conjugate_prior(), pb) if treat.emission else None
        self.beta_prior = BetaConjugate(1.0, 1.0)
        self.alpha0 = 1.0
        valid_dt = np.asarray(pb.dt)[np.asarray(pb.valid)]
        self.dt = float(valid_dt[0]) if valid_dt.size else float(pb.nominal_dt)
        self.mask = jnp.asarray(space.layout.target_mask)
        self.keep_per_t = (not treat.duration) or (not treat.transition)
        keys = self._q_keys(spec, pb) if treat.emission else ()
        emission = VBEmission(spec.emission, keys, treat.emission)
        self.spec_vb = dataclasses.replace(spec, emission=emission)  # type: ignore[arg-type]
        self.space_vb = dataclasses.replace(space, transition_fn=self._transition_fn(spec, space, treat))
        self.step = jax.jit(self._step)
        self.init = jax.jit(self._init)

    # --- structure -----------------------------------------------------------------------------

    def _q_keys(self, spec: ModelSpec, pb: PaddedBatch) -> tuple[str, ...]:
        n = spec.n_states
        zeros = SimpleNamespace(gamma_regime=jnp.zeros((*pb.observations.shape[:2], n)))
        return tuple(sorted(self.prior.update(self._emission_stats(pb, zeros))))

    @staticmethod
    def _transition_fn(spec: ModelSpec, space: StateSpace, treat: Treatment) -> Any:  # noqa: ANN401
        s = space.n_expanded
        diag = jnp.arange(s)
        row_of_state = space.index.row_of_state
        valid_next = space.next_phase_state >= 0
        safe_next = jnp.where(valid_next, space.next_phase_state, 0)
        valid_exit = (space.exit_targets >= 0) & space.is_last_phase[:, None]
        safe_exit = jnp.where(valid_exit, space.exit_targets, 0)
        rate_index_of = space.rate_index_of
        tiny = 1e-300

        def fn(theta: Params, z_pi_t: Array | None, z_d_t: Array | None, dt_t: Array) -> tuple[Array, Array]:
            """`log A`: expected log-weights (conjugate blocks), log-probabilities (point blocks)."""
            if treat.duration:
                log_stay = theta["vb.log_stay"][rate_index_of]
                log_adv = theta["vb.log_adv"][rate_index_of]
            else:
                a = spec.duration.advance_prob_t(theta, dt_t, z_d_t)[rate_index_of]
                log_stay = jnp.log(jnp.maximum(1.0 - a, tiny))
                log_adv = jnp.log(jnp.maximum(a, tiny))
            if treat.transition:
                log_pi = theta["vb.log_pi"][row_of_state]  # (S, N)
            else:
                pi = spec.transition.jump_probs_t(theta, z_pi_t, space.layout)[row_of_state]
                log_pi = jnp.where(pi > 0, jnp.log(jnp.where(pi > 0, pi, 1.0)), -jnp.inf)
            la = jnp.full((s, s), -jnp.inf)
            la = la.at[diag, diag].max(log_stay)
            la = la.at[diag, safe_next].max(jnp.where(valid_next, log_adv, -jnp.inf))
            la = la.at[diag[:, None], safe_exit].max(
                jnp.where(valid_exit, log_adv[:, None] + log_pi, -jnp.inf)
            )
            return jnp.exp(la), la

        return fn

    # --- statistics and q ----------------------------------------------------------------------

    def _emission_stats(self, pb: PaddedBatch, stats: Any) -> Any:  # noqa: ANN401
        m = pb.observations.shape[-1]
        observed = pb.valid & jnp.all(pb.mask, axis=-1)
        gamma = (stats.gamma_regime * observed[..., None]).reshape(-1, stats.gamma_regime.shape[-1])
        out = self.spec.emission.sufficient_stats(
            pb.observations.reshape(-1, m), pb.dt.reshape(-1), gamma, None
        )
        assert out is not None
        return out

    def _update(self, stats: ExpandedEStats, pb: PaddedBatch) -> dict[str, Any]:
        q: dict[str, Any] = {}
        if self.treat.emission:
            q["emission"] = self.prior.update(self._emission_stats(pb, stats))
        if self.treat.duration:
            q["dur"] = {"a": self.beta_prior.a0 + stats.advance, "b": self.beta_prior.b0 + stats.stay}
        if self.treat.transition:
            q["jump"] = self.alpha0 + jnp.where(self.mask, stats.jump, 0.0)
        return q

    def _theta(self, q: dict[str, Any], point: Params) -> Params:
        psi = jsp.digamma
        theta: Params = dict(point)
        if self.treat.duration:
            a, b = q["dur"]["a"], q["dur"]["b"]
            theta["vb.log_stay"] = psi(b) - psi(a + b)
            theta["vb.log_adv"] = psi(a) - psi(a + b)
        if self.treat.transition:
            alpha = q["jump"]
            total = jnp.sum(jnp.where(self.mask, alpha, 0.0), axis=-1, keepdims=True)
            theta["vb.log_pi"] = jnp.where(
                self.mask, psi(jnp.where(self.mask, alpha, 1.0)) - psi(total), -jnp.inf
            )
        if self.treat.emission:
            for k, v in q["emission"].items():
                theta[f"vb.q.{k}"] = v
        return theta

    def _kl(self, q: dict[str, Any]) -> Array:
        total = jnp.zeros(())
        if self.treat.emission:
            total = total + self.prior.kl(q["emission"])
        if self.treat.duration:
            total = total + jnp.sum(
                beta_kl(q["dur"]["a"], q["dur"]["b"], self.beta_prior.a0, self.beta_prior.b0)
            )
        if self.treat.transition:
            total = total + jnp.sum(dirichlet_kl(q["jump"], self.alpha0, self.mask))
        return total

    def _point_prefixes(self) -> list[str]:
        t = self.treat
        return [
            p
            for p, conj in (
                ("emission.", t.emission),
                ("duration.", t.duration),
                ("transition.", t.transition),
            )
            if not conj
        ]

    def _log_prior_point(self, point: Params) -> Array:
        """`log p(theta) + log|J|` of the point blocks (the term ECM maximizes with `use_priors`)."""
        total = jnp.zeros(())
        for prefix in self._point_prefixes():
            specs = tuple(s for s in self.spec.param_specs if s.name.startswith(prefix))
            if not specs:
                continue
            manager = ParamManager(specs)
            if manager.n_free == 0:
                continue
            total = total + manager.log_prior(point) + manager.log_det_jac(manager.flatten(point))
        return total

    def _ecm(self, point: Params, stats: ExpandedEStats, pb: PaddedBatch) -> Params:
        """One ECM conditional step of every point block on the current expected counts."""
        out = dict(point)
        spec = self.spec
        for prefix in self._point_prefixes():
            if prefix == "emission.":
                q_fn = partial(ecm.q_emission, spec=spec, pb=pb, stats=stats)
            elif prefix == "duration.":
                q_fn = partial(ecm.q_duration, spec=spec, pb=pb, stats=stats)
            else:
                q_fn = partial(ecm.q_transition, spec=spec, space=self.space, pb=pb, stats=stats)
            out = ecm.ecm_block(out, spec, prefix, q_fn, use_priors=True, inner_iter=5)
        return out

    def _init(self, theta0: Params, pb: PaddedBatch) -> dict[str, Any]:
        stats = _estep_batch(theta0, self.spec, self.space, pb, jnp.asarray(1.0), False, self.keep_per_t)
        return self._update(stats, pb)

    def _step(
        self, q: dict[str, Any], point: Params, pb: PaddedBatch
    ) -> tuple[dict[str, Any], Params, Array, Array]:
        theta = self._theta(q, point)
        stats = _estep_batch(theta, self.spec_vb, self.space_vb, pb, jnp.asarray(1.0), False, self.keep_per_t)
        log_z = jnp.sum(stats.loglik)
        elbo = log_z - self._kl(q) + self._log_prior_point(point)
        occupancy = jnp.sum(stats.gamma_regime, axis=(0, 1)) / jnp.maximum(jnp.sum(pb.lengths), 1)
        return self._update(stats, pb), self._ecm(point, stats, pb), elbo, occupancy

    # --- posterior means -----------------------------------------------------------------------

    def means(self, q: dict[str, Any], theta: Params, pb: PaddedBatch) -> Params:
        """The constrained parameters at the posterior means (point blocks: the point estimates)."""
        out = dict(theta)
        if self.treat.emission:
            for name, value in self.prior.mean_params(q["emission"]).items():
                out[name] = jnp.reshape(value, jnp.shape(theta[name]))
        if self.treat.duration:
            a = q["dur"]["a"] / (q["dur"]["a"] + q["dur"]["b"])
            for name, value in self.spec.duration.from_advance_prob(a, self.dt).items():  # type: ignore[attr-defined]
                out[name] = jnp.reshape(value, jnp.shape(theta[name]))
        if self.treat.transition:
            view = SimpleNamespace(jump=q["jump"])
            out = transition_mstep(self.spec, self.space, out, view, False)  # type: ignore[arg-type]
        return out

    def posterior_sd(self, q: dict[str, Any], theta: Params) -> dict[str, np.ndarray]:
        """Posterior standard deviations of the conjugate blocks on the constrained scale (delta method)."""
        out: dict[str, np.ndarray] = {}
        if self.treat.emission and hasattr(self.prior, "sd_params"):
            for name, value in self.prior.sd_params(q["emission"]).items():
                out[name] = np.asarray(jnp.reshape(value, jnp.shape(theta[name])))
        if self.treat.duration and "duration.rate" in theta:
            a, b = q["dur"]["a"], q["dur"]["b"]
            mean = a / (a + b)
            sd_a = jnp.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
            out["duration.rate"] = np.asarray(
                jnp.reshape(sd_a / ((1.0 - mean) * self.dt), jnp.shape(theta["duration.rate"]))
            )
        return out


def run_vb(
    theta0: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    cfg: EstimationSection,
    *,
    debug: bool | None = None,
) -> tuple[Params, list[float], bool, int, dict[str, Any]]:
    """One VB run from the point estimate `theta0`.

    Returns `(theta at the posterior means, ELBO trace, converged, n_iter, extras)` with the extras
    `elbo_trace`, `q_params`, `treatment`, `occupancy`, `pruned_states`, `std_errors`.

    Raises:
        ConfigError: unsupported structure or `use_priors: false`.
        NumericalError: a non-finite ELBO, or (debug mode) an ELBO decrease beyond `1e-8`.
    """
    if not cfg.use_priors:
        raise ConfigError("VB needs priors (estimation.use_priors: true)", path="estimation.use_priors")
    treat = check_vb_supported(spec, pb)
    check = bool(debug) if debug is not None else os.environ.get("HSMMKIT_DEBUG") == "1"
    ctx = _Context(spec, space, pb, treat)
    q = ctx.init(theta0, pb)
    point = dict(theta0)
    trace: list[float] = []
    converged = False
    q_eval, point_eval, occupancy = q, point, np.zeros(spec.n_states)
    n_iter = 0
    for n_iter in range(1, cfg.max_iter + 1):
        q_next, point_next, elbo_a, occ = ctx.step(q, point, pb)
        elbo = float(elbo_a)
        if not np.isfinite(elbo):
            raise NumericalError("non-finite ELBO during VB", path="elbo")
        if check and trace and elbo < trace[-1] - (ELBO_SLACK + 1e-13 * abs(trace[-1])):
            raise NumericalError(
                f"ELBO decreased: {trace[-1]!r} -> {elbo!r} at iteration {n_iter}", path="elbo"
            )
        trace.append(elbo)
        q_eval, point_eval, occupancy = q, point, np.asarray(occ)
        if len(trace) > 1 and abs(elbo - trace[-2]) <= cfg.tol_f * max(abs(elbo), 1.0):
            converged = True
            break
        q, point = q_next, point_next
    theta = ctx.means(q_eval, point_eval, pb)
    shapes = {s.name: s.shape for s in spec.param_specs}  # init strategies may return `(N,)` for `(N, 1)`
    theta = {k: (jnp.reshape(v, shapes[k]) if k in shapes else v) for k, v in theta.items()}
    eps = cfg.vb.prune_below if cfg.vb is not None else 0.01
    pruned = tuple(int(i) for i in np.flatnonzero(occupancy < eps))
    if pruned:
        warnings.warn(
            f"VB: regimes {list(pruned)} have expected occupancy below {eps} and are pruned from the report; "
            "consider refitting with a smaller model.n_states",
            stacklevel=2,
        )
    extras = {
        "elbo_trace": list(trace),
        "q_params": jax.tree_util.tree_map(np.asarray, q_eval),
        "treatment": treat.table(),
        "occupancy": occupancy,
        "pruned_states": pruned,
        "std_errors": ctx.posterior_sd(q_eval, theta),
    }
    return theta, trace, converged, n_iter, extras


class VBEstimator(EMEstimator):
    """Registered estimator `"vb"`: mean-field VB with MAP's multi-start (best final ELBO wins)."""

    name = "vb"
    diagnostics_key = "vb"
    score_from_trace = True
    trace_key = "elbo_trace"

    def _section(self, cfg: EstimationSection) -> EmCfg:
        return EmCfg()

    def _check(self, spec: ModelSpec, data: PreparedDataset | SequenceBatch) -> None:
        del data
        if spec.engine != "expanded":
            raise ConfigError("VB supports the expanded engine only", path="model.engine")

    def _plan(self, spec: ModelSpec, pb: PaddedBatch) -> Plan:
        treat = check_vb_supported(spec, pb)
        return tuple((name, kind) for name, kind in treat.table().items())

    def _run(
        self,
        theta0: Params,
        spec: ModelSpec,
        space: StateSpace,
        pb: PaddedBatch,
        cfg: EstimationSection,
        plan: Plan,
    ) -> Any:  # noqa: ANN401
        del plan
        return run_vb(theta0, spec, space, pb, cfg)

    def _final(self, theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch) -> float:
        return _final_loglik(theta, spec, space, pb)


REGISTRY.register("estimator", VBEstimator.name, VBEstimator(), provenance=provenance_of(__name__))
