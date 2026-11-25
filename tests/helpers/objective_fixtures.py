"""Model assembly for objective tests (not collected): family x engine x covariates combinations.

`make_case` builds a `ModelSpec` + `StateSpace` + `ParamManager` + dataset by hand from the families'
default parameter specs plus coefficient specs (there is no config -> model assembly yet, see
`model/build.py` in CLAUDE.md §3.3).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from hsmmkit.data.contracts import PreparedDataset
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.model.duration.gamma import GammaDuration
from hsmmkit.model.duration.geometric import GeometricDuration
from hsmmkit.model.duration.negbin import NegBinDuration
from hsmmkit.model.emission.gaussian import GaussianFamily, _coef_spec
from hsmmkit.model.emission.student_t import StudentTFamily
from hsmmkit.model.params import ParamManager, ParamSpec
from hsmmkit.model.params.priors import DirichletPrior, LogNormalPrior, NormalPrior
from hsmmkit.model.spec import ModelSpec
from hsmmkit.model.transition import JumpTransition, build_memory_layout
from hsmmkit.statespace.builders import build_expanded, build_explicit
from hsmmkit.statespace.index import StateSpace
from tests.helpers.engine_fixtures import DT, make_dataset


@dataclass
class Case:
    """A complete objective problem: structure, layout, data and a starting point."""

    spec: ModelSpec
    space: StateSpace
    engine: ExpandedHMMEngine | ExplicitDurationEngine
    manager: ParamManager
    data: PreparedDataset
    u0: jnp.ndarray


def _with_mode(spec: ParamSpec, prior: object) -> ParamSpec:
    """`spec` with every state in `prior` mode under `prior`."""
    return dataclasses.replace(spec, mode="prior", prior=prior)  # type: ignore[arg-type]


def make_case(
    emission: str,
    duration: str,
    order: int,
    *,
    cov_tr: bool = False,
    cov_du: bool = False,
    cov_em: bool = False,
    t_len: int = 40,
    seed: int = 0,
    priors: bool = False,
    dirichlet_alpha: float | None = None,
) -> Case:
    """Build a case; `emission` in gbm/student_t, `duration` in geometric/negbin/gamma (explicit engine).

    Transition covariates need `N >= 3` (with `N = 2` the only target has no free logit), so those cases
    use three regimes; all others use two.
    """
    n = 3 if cov_tr else 2
    rng = np.random.default_rng(seed)
    explicit = duration == "gamma"
    if explicit and order != 1:
        raise ValueError("the explicit engine supports order 1 only")

    cov_e = (0,) if cov_em else ()
    cov_d = (0,) if cov_du else ()
    emis = (
        GaussianFamily(scale_by_dt=True, ito_correction=True, mean_idx=cov_e, scale_idx=cov_e)
        if emission == "gbm"
        else StudentTFamily(scale_by_dt=True, loc_idx=cov_e, logscale_idx=cov_e)
    )
    specs: list[ParamSpec] = list(emis.param_specs(n, 1, ()))
    scale_name = "emission.scale" if emission == "gbm" else "emission.s"
    if priors:
        specs = [
            _with_mode(s, LogNormalPrior(mean_log=float(np.log(0.2)), sd=1.0)) if s.name == scale_name else s
            for s in specs
        ]
    specs = [
        dataclasses.replace(s, init=np.linspace(0.15, 0.4, n)[:, None]) if s.name == scale_name else s
        for s in specs
    ]
    if cov_em:
        specs += [_coef_spec("emission.gamma", n, 1), _coef_spec("emission.delta", n, 1)]

    if duration == "geometric":
        dur = GeometricDuration(cov_idx=cov_d)
    elif duration == "negbin":
        dur = NegBinDuration(r=tuple(1 + i % 2 for i in range(n)), cov_idx=cov_d)
    else:
        dur = GammaDuration(cov_idx=cov_d)
    dur_specs = list(dur.param_specs(n, ()))
    for i, s in enumerate(dur_specs):
        if s.name == "duration.rate":
            init = np.linspace(12.0, 25.0, n)
            dur_specs[i] = (
                _with_mode(s, LogNormalPrior(mean_log=float(np.log(15.0)), sd=1.0)) if priors else s
            )
            dur_specs[i] = dataclasses.replace(dur_specs[i], init=init)
    specs += dur_specs
    if cov_du:
        from hsmmkit.model.duration.base import coef_spec

        specs.append(coef_spec("duration.c", n, 1))

    layout = build_memory_layout(n, order)
    r_rows = layout.tuples.shape[0]
    trans = JumpTransition(cov_idx=(0,) if cov_tr else (), dirichlet_alpha=dirichlet_alpha)
    if cov_tr:
        specs.append(
            ParamSpec("transition.a", (r_rows, n - 2), "real", "free", init=np.zeros((r_rows, n - 2)))
        )
        specs.append(
            ParamSpec(
                "transition.b",
                (r_rows, n - 2, 1),
                "real",
                "prior" if priors else "free",
                init=np.zeros((r_rows, n - 2, 1)),
                prior=NormalPrior(sd=1.0) if priors else None,
            )
        )
    else:
        pi = trans.param_specs(n, order, ())[0]
        specs.append(_with_mode(pi, DirichletPrior(alpha=1.5)) if priors else pi)

    spec = ModelSpec(
        n_states=n,
        n_series=1,
        emission=emis,
        duration=dur,
        transition=trans,
        order=order,
        duration_conditioning="current",
        engine="explicit" if explicit else "expanded",
        d_max=12 if explicit else 10,
        initial="uniform",
        param_specs=tuple(specs),
    )
    space = build_explicit(spec) if explicit else build_expanded(spec)
    engine = ExplicitDurationEngine() if explicit else ExpandedHMMEngine()
    manager = ParamManager(spec.param_specs)

    y = rng.normal(0.0, 0.01, size=(t_len, 1))
    data = make_dataset(
        y,
        np.full(t_len, DT),
        z_pi=rng.normal(size=(t_len, 1)) if cov_tr else None,
        z_d=rng.normal(size=(t_len, 1)) if cov_du else None,
    )
    if cov_em:
        data = dataclasses.replace(data, cov_emission=jnp.asarray(rng.normal(size=(t_len, 1))))
    u0 = manager.init_u()
    u0 = u0 + jnp.asarray(rng.normal(0.0, 0.15, size=u0.shape))
    return Case(spec, space, engine, manager, data, u0)
