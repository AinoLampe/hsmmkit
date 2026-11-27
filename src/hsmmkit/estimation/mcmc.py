"""`MCMCEstimator`: NUTS on the marginal posterior of the parameters (E13-T06a, ADR-0024).

Concept refs: Part II §14.9 (a), §15.5 (`bayes`), §19 #15. The states are marginalized by the forward
algorithm, so the target is
`log_joint(u) = sum_b loglik_b(theta(u)) + log p(theta(u)) + log|d theta / d u|`, the negated MAP
objective of `estimation.objective` (`use_priors` is required). NumPyro's NUTS samples `u`
(`potential_fn(u) = -log_joint(u)`); every chain starts from the MAP optimum plus Gaussian jitter. Draws are
mapped back to the constrained `theta` and returned as ArviZ posterior draws (`FitResult.posterior_samples`).

Free parameters without a prior make the posterior improper: with `estimation.mcmc.default_priors: true` the
weakly informative defaults of `docs/math/estimation.md` are attached (real: `Normal(0, 10 max(1, |init|))`,
positive: `LogNormal(mean log init, 3)`, `nu > 2`: `LogNormal(log 10, 1.5)`, packed Cholesky:
`LogNormal` on the diagonal); otherwise they are a `ConfigError`. Without an identifiability guard the
posterior can be multimodal (label switching): a warning is issued; relabelling is E13-T06b. Same seed,
backend and library versions give
identical draws; the versions are recorded in `diagnostics["mcmc"]["versions"]`.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.config.schema import McmcCfg
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.data.contracts import PreparedDataset
from hsmmkit.engine.batch import BatchedEngine, batched_loglik
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.estimation.map import fit as map_fit
from hsmmkit.estimation.objective import _objective, make_objective
from hsmmkit.estimation.results import FitResult
from hsmmkit.interop.arviz import summary_table, to_inference_data
from hsmmkit.model.lattice import has_identifiability_guard
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.model.params.priors import CholeskyDiagPrior, LogNormalPrior, NormalPrior
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.statespace.builders import build_expanded, build_explicit
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.jax_setup import setup as jax_setup
from hsmmkit.utils.optional import optional_import
from hsmmkit.utils.rng import PURPOSE_MCMC, key_for
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EstimationSection
    from hsmmkit.model.params import Params
    from hsmmkit.model.params.spec import ParamSpec
    from hsmmkit.model.spec import ModelSpec

__all__ = ["MCMCEstimator", "improper_parameters", "with_default_priors"]

_IMPROPER = ("real", "positive", "nu_gt_2", "cholesky", "increasing", "increasing_positive")
R_HAT_LIMIT = 1.05


def _modes(spec: ParamSpec) -> tuple[str, ...]:
    lead = spec.shape[0] if spec.shape else 1
    return (spec.mode,) * lead if isinstance(spec.mode, str) else tuple(spec.mode)


def _default_prior(spec: ParamSpec) -> Any:  # noqa: ANN401
    """The weakly informative prior of a free entry of `spec`'s domain (`None`: no default)."""
    init = np.asarray(spec.init if spec.init is not None else spec.value if spec.value is not None else 1.0)
    if spec.domain in ("real", "increasing"):
        return NormalPrior(sd=10.0 * max(1.0, float(np.max(np.abs(init)))), mean=0.0)
    if spec.domain in ("positive", "increasing_positive"):
        return LogNormalPrior(mean_log=float(np.mean(np.log(np.maximum(init, 1e-12)))), sd=3.0)
    if spec.domain == "nu_gt_2":
        return LogNormalPrior(mean_log=float(np.log(10.0)), sd=1.5)
    if spec.domain == "cholesky":
        return CholeskyDiagPrior(LogNormalPrior(mean_log=0.0, sd=3.0))
    return None


def improper_parameters(spec: ModelSpec) -> list[str]:
    """Names of the specs with free entries of an unbounded domain and no prior (an improper posterior)."""
    out = []
    for s in spec.param_specs:
        if s.domain in _IMPROPER and "free" in _modes(s) and s.tie_group is None:
            out.append(s.name)
    return out


def with_default_priors(spec: ModelSpec) -> ModelSpec:
    """`spec` with the default prior on every spec that has free entries of an unbounded domain."""
    new: list[ParamSpec] = []
    for s in spec.param_specs:
        modes = _modes(s)
        if s.domain in _IMPROPER and "free" in modes and "prior" not in modes and s.tie_group is None:
            prior = _default_prior(s)
            if prior is not None:
                prior_modes: Any = (
                    "prior"
                    if (s.constraint != "none" or all(m == "free" for m in modes))
                    else tuple("prior" if m == "free" else m for m in modes)
                )
                new.append(dataclasses.replace(s, mode=prior_modes, prior=prior))
                continue
        new.append(s)
    return dataclasses.replace(spec, param_specs=tuple(new))


def _versions() -> dict[str, str]:
    import numpyro  # type: ignore[import-untyped]

    az = optional_import("arviz", extra="bayes")
    return {"numpyro": numpyro.__version__, "arviz": str(az.__version__), "jax": jax.__version__}


class MCMCEstimator:
    """Registered estimator `"mcmc"`: NUTS (NumPyro) on `log_joint(u)` with ArviZ output."""

    name = "mcmc"

    def fit(
        self,
        spec: ModelSpec,
        data: PreparedDataset | SequenceBatch,
        cfg: EstimationSection,
        seed: int,
        warm_start: Array | None = None,
    ) -> FitResult:
        """Sample the posterior; `FitResult(estimator="mcmc")` with posterior means as `params`.

        `posterior_samples` holds the ArviZ draws, `diagnostics["mcmc"]` the R-hat / ESS tables, the number
        of divergences, the sampler settings and the library versions; `std_errors` are the posterior SDs.

        Raises:
            ImportError: the `bayes` extra (NumPyro, ArviZ) is not installed.
            ConfigError: `use_priors: false`, the Gibbs sampler (E13-T06b) or improper parameters without
                `default_priors`.
            NumericalError: a non-finite potential at the start.
        """
        optional_import("numpyro", extra="bayes")
        optional_import("arviz", extra="bayes")
        infer = optional_import("numpyro.infer", extra="bayes")
        m_cfg = cfg.mcmc if cfg.mcmc is not None else McmcCfg()
        if not cfg.use_priors:
            raise ConfigError("MCMC needs priors (estimation.use_priors: true)", path="estimation.use_priors")
        if m_cfg.sampler != "nuts":
            raise ConfigError(
                f"sampler {m_cfg.sampler!r} is available from E13-T06b", path="estimation.mcmc.sampler"
            )
        jax_setup(cfg.device)
        if m_cfg.default_priors:
            spec = with_default_priors(spec)
        bad = improper_parameters(spec)
        if bad:
            raise ConfigError(
                f"free parameters without a prior make the posterior improper: {bad}; "
                "set estimation.mcmc.default_priors: true or give them priors "
                "(a spec that mixes free and prior entries keeps its own prior)",
                path="estimation.mcmc.default_priors",
            )
        batch = data if isinstance(data, SequenceBatch) else SequenceBatch.of(data)
        first = batch.sequences[0]
        if not has_identifiability_guard(spec):
            warnings.warn(
                "no identifiability guard (no fixed or ordered emission location/scale spec): the posterior "
                "can be multimodal by label switching; post-hoc relabelling comes with E13-T06b",
                UserWarning,
                stacklevel=2,
            )
        manager = ParamManager(spec.param_specs)
        space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
        base = (
            ExplicitDurationEngine(chunk_size=cfg.chunk_size)
            if spec.engine == "explicit"
            else ExpandedHMMEngine(parallel_scan=cfg.parallel_scan)
        )
        engine = BatchedEngine(base)
        pb = batch.padded()
        objective = make_objective(spec, space, engine, manager, pb, use_priors=True)

        # --- starting point: the MAP optimum (the MAP restarts of the configuration) -----------------
        map_cfg = cfg.model_copy(update={"method": "map"})
        start = map_fit(spec, data, map_cfg, seed, warm_start)
        u_map = jnp.asarray(manager.flatten(start.params))
        if not bool(jnp.isfinite(objective.value(u_map))):
            raise NumericalError("non-finite log-joint at the MAP start", path="estimation.mcmc")

        def potential(u: Array) -> Array:
            return _objective(u, space, pb, spec, engine, manager, True)  # type: ignore[arg-type]

        n = int(u_map.shape[0])
        chains = m_cfg.chains
        keys = [key_for(seed, PURPOSE_MCMC, c) for c in range(chains)]
        starts = jnp.stack(
            [u_map + m_cfg.jitter * jax.random.normal(jax.random.fold_in(k, 1), (n,)) for k in keys]
        )
        extra = ("potential_energy", "num_steps", "diverging", "energy")
        kernel = infer.NUTS(
            potential_fn=potential,
            target_accept_prob=m_cfg.target_accept,
            max_tree_depth=m_cfg.max_tree_depth,
        )
        draws: list[Any] = []
        stats: list[dict[str, Any]] = []
        if m_cfg.chain_method == "vectorized" and chains > 1:
            mcmc = infer.MCMC(
                kernel,
                num_warmup=m_cfg.warmup,
                num_samples=m_cfg.draws,
                num_chains=chains,
                chain_method="vectorized",
                progress_bar=False,
            )
            mcmc.run(keys[0], init_params=starts, extra_fields=extra)
            flat = np.asarray(mcmc.get_samples(group_by_chain=True))
            fields = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
            draws = [flat[c] for c in range(chains)]
            stats = [{k: v[c] for k, v in fields.items()} for c in range(chains)]
        else:
            for c in range(chains):
                mcmc = infer.MCMC(
                    kernel,
                    num_warmup=m_cfg.warmup,
                    num_samples=m_cfg.draws,
                    num_chains=1,
                    progress_bar=False,
                )
                mcmc.run(keys[c], init_params=starts[c], extra_fields=extra)
                draws.append(np.asarray(mcmc.get_samples()))
                stats.append({k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()})
        u_draws = jnp.asarray(np.stack(draws))  # (chain, draw, n)

        # --- constrained draws, per-sequence log-likelihood, ArviZ ----------------------------------
        theta_draws = jax.jit(jax.vmap(jax.vmap(manager.unflatten)))(u_draws)
        posterior = {k: np.asarray(v) for k, v in theta_draws.items()}
        sample_stats = {
            "diverging": np.stack([s["diverging"] for s in stats]).astype(bool),
            "tree_depth": np.log2(np.stack([s["num_steps"] for s in stats]) + 1.0).round().astype(int),
            "energy": np.stack([s["energy"] for s in stats]),
            "potential_energy": np.stack([s["potential_energy"] for s in stats]),
        }
        log_lik = None
        if m_cfg.log_likelihood:

            @jax.jit
            def per_sequence(u: Array) -> Array:
                return batched_loglik(base, manager.unflatten(u), spec, space, pb)

            flat_u = u_draws.reshape(-1, n)
            ll = jax.lax.map(per_sequence, flat_u)[:, : len(batch.sequences)]
            log_lik = np.asarray(ll).reshape(chains, m_cfg.draws, -1)
        versions = _versions()
        idata = to_inference_data(
            posterior,
            sample_stats,
            log_lik,
            sequence_ids=list(batch.ids),
            attrs={"seed": seed, "sampler": "nuts", **versions},
        )
        table = summary_table(idata)
        r_hat = {k: float(v) for k, v in table["r_hat"].items()}
        ess_bulk = {k: float(v) for k, v in table["ess_bulk"].items()}
        ess_tail = {k: float(v) for k, v in table["ess_tail"].items()}
        divergences = int(sample_stats["diverging"].sum())

        # --- the FitResult -----------------------------------------------------------------------
        mean_theta: Params = {k: jnp.asarray(v.mean(axis=(0, 1))) for k, v in posterior.items()}
        u_mean = manager.flatten(mean_theta)
        loglik = float(objective.loglik(u_mean))
        sd = {k: jnp.asarray(v.reshape(-1, *v.shape[2:]).std(axis=0)) for k, v in posterior.items()}
        finite_r_hat = [v for v in r_hat.values() if np.isfinite(v)]
        max_r_hat = max(finite_r_hat) if finite_r_hat else float("nan")  # one chain: R-hat is undefined
        diagnostics: dict[str, Any] = {
            "mcmc": {
                "r_hat": r_hat,
                "ess_bulk": ess_bulk,
                "ess_tail": ess_tail,
                "divergences": divergences,
                "max_r_hat": max_r_hat,
                "min_ess_bulk": float(np.nanmin(list(ess_bulk.values()))) if ess_bulk else float("nan"),
                "sampler": "nuts",
                "chains": chains,
                "warmup": m_cfg.warmup,
                "draws": m_cfg.draws,
                "target_accept": m_cfg.target_accept,
                "max_tree_depth": m_cfg.max_tree_depth,
                "default_priors": m_cfg.default_priors,
                "versions": versions,
            },
            "per_sequence": [
                {"id": sid, "group": seq.group, "T": int(seq.observations.shape[0])}
                for sid, seq in zip(batch.ids, batch.sequences, strict=True)
            ],
        }
        if divergences:
            warnings.warn(f"MCMC: {divergences} divergent transitions", UserWarning, stacklevel=2)
        return FitResult(
            estimator="mcmc",
            params=mean_theta,
            uparams=u_mean,
            objective=float(objective.value(u_mean)),
            loglik=loglik,
            restarts=start.restarts,
            converged=bool(max_r_hat <= R_HAT_LIMIT),
            n_iter=m_cfg.draws,
            std_errors=sd,
            diagnostics=diagnostics,
            spec=spec,
            prep_stats=first.prep_stats,
            config_resolved={},
            posterior_samples=idata,
        )


REGISTRY.register("estimator", MCMCEstimator.name, MCMCEstimator(), provenance=provenance_of(__name__))
