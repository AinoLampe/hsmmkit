"""E13-T01a AC3: the E-step never materializes `(T, S, S)` and stays within `5 (T N + S^2)` floats."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import jaxprs_in_params

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.estep import _estep_batch
from tests.helpers.engine_fixtures import make_scenario

T, N, R = 20_000, 30, 2  # S = N R = 60


def _scenario() -> Any:  # noqa: ANN401
    sc = make_scenario(N, [R] * N, 1, T, seed=0)
    assert sc.space.n_expanded == 60
    return sc


def _avals(jaxpr: Any) -> list[tuple[int, ...]]:  # noqa: ANN401
    """Every array shape in a (nested) jaxpr."""
    shapes: list[tuple[int, ...]] = []

    def walk(jp: Any) -> None:  # noqa: ANN401
        for var in [*jp.invars, *jp.outvars, *jp.constvars]:
            shapes.append(tuple(getattr(var.aval, "shape", ())))
        for eqn in jp.eqns:
            for var in [*eqn.invars, *eqn.outvars]:
                shapes.append(tuple(getattr(getattr(var, "aval", None), "shape", ())))
            for sub in jaxprs_in_params(eqn.params):
                walk(sub)

    walk(jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr)
    return shapes


def test_e13_t01a_no_t_s_s_array_in_the_program() -> None:
    """The jaxpr (scan bodies included) holds no array with two trailing `S` dimensions and a time axis."""
    sc = make_scenario(3, [2, 3, 2], 1, 40, n_cov_pi=2, n_cov_d=2, seed=0)  # S = 7, T = 40
    pb = SequenceBatch.of(sc.data).padded()
    s, t_max = sc.space.n_expanded, int(pb.observations.shape[1])
    jaxpr = jax.make_jaxpr(
        lambda th: _estep_batch.__wrapped__(th, sc.spec, sc.space, pb, jnp.asarray(1.0), False)
    )(sc.theta)
    shapes = _avals(jaxpr)
    assert any(sh[-2:] == (s, s) for sh in shapes)  # the per-step (S, S) matrix exists...
    assert not [
        sh for sh in shapes if len(sh) >= 3 and sh[-2:] == (s, s) and t_max in sh[:-2]
    ]  # never over time
    assert not [sh for sh in shapes if t_max in sh and s in sh and int(np.prod(sh)) >= t_max * s * s]


def test_e13_t01a_peak_memory_stays_below_the_budget() -> None:
    """Compiled memory analysis at `T = 20 000`, `S = 60`: below `5 (T N + S^2)` floats of 8 bytes."""
    sc = _scenario()
    pb = SequenceBatch.of(sc.data).padded()
    compiled = _estep_batch.lower(sc.theta, sc.spec, sc.space, pb, jnp.asarray(1.0), False).compile()
    analysis = compiled.memory_analysis()
    assert analysis is not None
    used = analysis.temp_size_in_bytes + analysis.output_size_in_bytes
    budget = 5 * (T * N + 60 * 60) * 8
    print(f"E13-T01a E-step memory: temp+output {used / 1e6:.1f} MB, budget {budget / 1e6:.1f} MB")
    assert used <= budget
