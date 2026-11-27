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
an initialization strategy (one E-step at that point gives the first `q`); restarts follow MAP; the run with
the best final ELBO wins. `FitResult.params` are the posterior means (constrained scale), `objective` is
`-ELBO`, `loglik` the log-likelihood at the posterior means, and `diagnostics["vb"]` holds `elbo_trace`
and `q_params`.

Supported: expanded engine, `initial_state: uniform`, constant step, no covariate links, conjugate emissions
(Gaussian diagonal / full, Poisson, Bernoulli, Categorical), geometric / negative-binomial durations, jump
rows (also under a topology without absorbing regimes), `use_priors: true`. Everything else raises a
`ConfigError` ("VB-EM hybrid available from E13-T05b"). Priors are the weakly informative defaults of
`docs/math/estimation.md` (Gaussian priors scaled to the data); the config `prior` entries are not read.
`diagnostics["vb"]["q_params"]` keep the numbering of the run: when `diagnostics["vb"]["relabelled"]` is
true the fitted `params` were renumbered by ascending scale and `q_params` were not.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from hsmmkit.config.schema import EmCfg
from hsmmkit.engine.estep import _estep_batch
from hsmmkit.estimation.em import EMEstimator, Plan, _final_loglik
from hsmmkit.estimation.mstep import transition_mstep
from hsmmkit.model.conjugate import BetaConjugate, beta_kl, dirichlet_kl
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

__all__ = ["VBEmission", "VBEstimator", "check_vb_supported", "run_vb"]

ELBO_SLACK = 1e-8  # decrease tolerated before debug mode raises (Appendix G)
_HYBRID = "VB-EM hybrid available from E13-T05b"


@dataclass(frozen=True)
class VBEmission:
    """Emission wrapper whose `log_prob` is `E_q[log f]` read from `theta["vb.q.<key>"]`."""

    family: Any
    keys: tuple[str, ...]
    name: str = "vb"

    def init_carry(self) -> None:
        """No recurrent state."""
        return None

    def log_prob(self, y: Array, dt: Array, theta: Params, z: Array | None) -> Array:
        """`(T, N)` expected log-density under `q`."""
        q = {k: theta[f"vb.q.{k}"] for k in self.keys}
        out = self.family.expected_log_prob(y, dt, q, z)
        assert out is not None
        return jnp.asarray(out)


def check_vb_supported(spec: ModelSpec, pb: PaddedBatch) -> None:
    """Raise `ConfigError` for the structures VB (fully conjugate, E13-T05a) does not handle."""
    if spec.engine != "expanded":
        raise ConfigError(f"VB supports the expanded engine only ({_HYBRID})", path="model.engine")
    if spec.memory_mode != "segments" or spec.duration_conditioning != "current":
        raise ConfigError("VB needs segment memory with duration_conditioning 'current'", path="model")
    if spec.initial != "uniform":
        raise ConfigError(f"VB supports initial_state: uniform only ({_HYBRID})", path="model.initial_state")
    if spec.duration_dependent_jumps or spec.left_censored:
        raise ConfigError("VB does not support duration-dependent jumps or left-censoring", path="model")
    topology = getattr(spec.transition, "topology", None)
    if topology is not None and topology.absorbing:
        raise ConfigError("VB does not support absorbing regimes", path="model.topology")
    covariates = (pb.cov_emission, pb.cov_duration, pb.cov_transition)
    if any(int(c.shape[-1]) > 0 for c in covariates) or bool(getattr(spec.transition, "cov_idx", ())):
        raise ConfigError(f"VB has no covariate links ({_HYBRID})", path="model")
    if bool(getattr(spec.duration, "cov_idx", ())) or not hasattr(spec.duration, "from_advance_prob"):
        raise ConfigError(
            f"duration {type(spec.duration).__name__} is not conjugate ({_HYBRID})", path="model.duration"
        )
    dt = np.asarray(pb.dt)[np.asarray(pb.valid)]
    if dt.size and not np.allclose(dt, dt[0], rtol=1e-9, atol=0.0):
        raise ConfigError(f"VB needs a constant step ({_HYBRID})", path="data.clock")
    if hasattr(spec.emission, "blocks") or hasattr(spec.emission, "log_prob_masked"):
        raise ConfigError(
            f"VB does not support product / mixture emissions ({_HYBRID})", path="model.emission"
        )
    if int(pb.observations.shape[-1]) > 1 and not getattr(spec.emission, "joint", False):
        scale_spec = next((s for s in spec.param_specs if s.name == "emission.scale"), None)
        if scale_spec is not None and len(scale_spec.shape) < 2:
            raise ConfigError(
                "VB fits one scale per series; per_series: false is not supported",
                path="model.emission.per_series",
            )
    prior = spec.emission.conjugate_prior()
    if prior is None or not hasattr(prior, "kl") or not hasattr(prior, "mean_params"):
        raise ConfigError(
            f"emission {type(spec.emission).__name__} has no conjugate prior with a KL term ({_HYBRID})",
            path="model.emission",
        )
    bad = [
        s.name
        for s in spec.param_specs
        if s.mode in ("fixed", "tied", "bounded", "prior")
        or s.tie_group is not None
        or s.constraint != "none"
    ]
    if bad:
        raise ConfigError(
            f"VB updates every parameter freely; fixed / bounded / prior / tied / ordered parameters "
            f"are not supported: {bad}",
            path="model",
        )


def _data_prior(prior: Any, pb: PaddedBatch) -> Any:  # noqa: ANN401
    """The conjugate prior centred on the data scale (Gaussian families); others unchanged."""
    if not hasattr(prior, "with_data_scale"):
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

    def __init__(self, spec: ModelSpec, space: StateSpace, pb: PaddedBatch) -> None:
        self.spec, self.space = spec, space
        self.prior = _data_prior(spec.emission.conjugate_prior(), pb)
        self.beta_prior = BetaConjugate(1.0, 1.0)
        self.alpha0 = 1.0
        valid_dt = np.asarray(pb.dt)[np.asarray(pb.valid)]
        self.dt = float(valid_dt[0]) if valid_dt.size else float(pb.nominal_dt)
        self.layout = space.layout
        self.mask = jnp.asarray(space.layout.target_mask)
        keys = self._q_keys(spec, pb)
        self.spec_vb = dataclasses.replace(spec, emission=VBEmission(spec.emission, keys))  # type: ignore[arg-type]
        self.space_vb = dataclasses.replace(space, transition_fn=self._transition_fn(spec, space))
        self.step = jax.jit(self._step)
        self.init = jax.jit(self._init)

    # --- structure -----------------------------------------------------------------------------

    def _q_keys(self, spec: ModelSpec, pb: PaddedBatch) -> tuple[str, ...]:
        n = spec.n_states
        stats = self._emission_stats(
            pb, SimpleNamespace(gamma_regime=jnp.zeros((*pb.observations.shape[:2], n)))
        )
        return tuple(sorted(self.prior.update(stats)))

    @staticmethod
    def _transition_fn(spec: ModelSpec, space: StateSpace) -> Any:  # noqa: ANN401
        s = space.n_expanded
        diag = jnp.arange(s)
        row_of_state = space.index.row_of_state
        valid_next = space.next_phase_state >= 0
        safe_next = jnp.where(valid_next, space.next_phase_state, 0)
        valid_exit = (space.exit_targets >= 0) & space.is_last_phase[:, None]
        safe_exit = jnp.where(valid_exit, space.exit_targets, 0)
        rate_index_of = space.rate_index_of
        del spec

        def fn(theta: Params, z_pi_t: Array | None, z_d_t: Array | None, dt_t: Array) -> tuple[Array, Array]:
            """`log A` from the expected log-weights `vb.log_stay` / `vb.log_adv` / `vb.log_pi`."""
            del z_pi_t, z_d_t, dt_t
            log_stay = theta["vb.log_stay"][rate_index_of]
            log_adv = theta["vb.log_adv"][rate_index_of]
            log_pi = theta["vb.log_pi"][row_of_state]  # (S, N)
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
        q_e = self.prior.update(self._emission_stats(pb, stats))
        a = self.beta_prior.a0 + stats.advance
        b = self.beta_prior.b0 + stats.stay
        alpha = self.alpha0 + jnp.where(self.mask, stats.jump, 0.0)
        return {"emission": q_e, "dur": {"a": a, "b": b}, "jump": alpha}

    def _theta(self, q: dict[str, Any]) -> Params:
        psi = jsp.digamma
        a, b = q["dur"]["a"], q["dur"]["b"]
        alpha = q["jump"]
        total = jnp.sum(jnp.where(self.mask, alpha, 0.0), axis=-1, keepdims=True)
        log_pi = jnp.where(self.mask, psi(jnp.where(self.mask, alpha, 1.0)) - psi(total), -jnp.inf)
        theta: Params = {
            "vb.log_stay": psi(b) - psi(a + b),
            "vb.log_adv": psi(a) - psi(a + b),
            "vb.log_pi": log_pi,
        }
        for k, v in q["emission"].items():
            theta[f"vb.q.{k}"] = v
        return theta

    def _kl(self, q: dict[str, Any]) -> Array:
        kl_e = self.prior.kl(q["emission"])
        kl_d = jnp.sum(beta_kl(q["dur"]["a"], q["dur"]["b"], self.beta_prior.a0, self.beta_prior.b0))
        kl_j = jnp.sum(dirichlet_kl(q["jump"], self.alpha0, self.mask))
        total: Array = kl_e + kl_d + kl_j
        return total

    def _init(self, theta0: Params, pb: PaddedBatch) -> dict[str, Any]:
        stats = _estep_batch(theta0, self.spec, self.space, pb, jnp.asarray(1.0), False, False)
        return self._update(stats, pb)

    def _step(self, q: dict[str, Any], pb: PaddedBatch) -> tuple[dict[str, Any], Array, Array]:
        stats = _estep_batch(self._theta(q), self.spec_vb, self.space_vb, pb, jnp.asarray(1.0), False, False)
        log_z = jnp.sum(stats.loglik)
        return self._update(stats, pb), log_z - self._kl(q), log_z

    # --- posterior means -----------------------------------------------------------------------

    def means(self, q: dict[str, Any], theta: Params, pb: PaddedBatch) -> Params:
        """The constrained parameters at the posterior means."""
        out = dict(theta)
        for name, value in self.prior.mean_params(q["emission"]).items():
            out[name] = jnp.reshape(value, jnp.shape(theta[name]))
        a = q["dur"]["a"] / (q["dur"]["a"] + q["dur"]["b"])
        for name, value in self.spec.duration.from_advance_prob(a, self.dt).items():  # type: ignore[attr-defined]
            out[name] = jnp.reshape(value, jnp.shape(theta[name]))
        view = SimpleNamespace(jump=q["jump"])
        return transition_mstep(self.spec, self.space, out, view, False)  # type: ignore[arg-type]


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

    Returns `(theta at the posterior means, ELBO trace, converged, n_iter, {"elbo_trace", "q_params"})`.

    Raises:
        ConfigError: unsupported structure or `use_priors: false`.
        NumericalError: a non-finite ELBO, or (debug mode) an ELBO decrease beyond `1e-8`.
    """
    if not cfg.use_priors:
        raise ConfigError("VB needs priors (estimation.use_priors: true)", path="estimation.use_priors")
    check_vb_supported(spec, pb)
    check = bool(debug) if debug is not None else os.environ.get("HSMMKIT_DEBUG") == "1"
    ctx = _Context(spec, space, pb)
    q = ctx.init(theta0, pb)
    trace: list[float] = []
    converged = False
    q_eval = q
    n_iter = 0
    for n_iter in range(1, cfg.max_iter + 1):
        q_next, elbo_a, _ = ctx.step(q, pb)
        elbo = float(elbo_a)
        if not np.isfinite(elbo):
            raise NumericalError("non-finite ELBO during VB", path="elbo")
        if check and trace and elbo < trace[-1] - (ELBO_SLACK + 1e-13 * abs(trace[-1])):
            raise NumericalError(
                f"ELBO decreased: {trace[-1]!r} -> {elbo!r} at iteration {n_iter}", path="elbo"
            )
        trace.append(elbo)
        q_eval = q
        if len(trace) > 1 and abs(elbo - trace[-2]) <= cfg.tol_f * max(abs(elbo), 1.0):
            converged = True
            break
        q = q_next
    theta = ctx.means(q_eval, theta0, pb)
    extras = {
        "elbo_trace": list(trace),
        "q_params": jax.tree_util.tree_map(
            np.asarray, {"emission": q_eval["emission"], "dur": q_eval["dur"], "jump": q_eval["jump"]}
        ),
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
            raise ConfigError(f"VB supports the expanded engine only ({_HYBRID})", path="model.engine")

    def _plan(self, spec: ModelSpec, pb: PaddedBatch) -> Plan:
        check_vb_supported(spec, pb)
        return (("emission", "vb-conjugate"), ("duration", "vb-conjugate"), ("transition", "vb-conjugate"))

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
