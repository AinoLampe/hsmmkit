"""E5-T03: `fit` orchestration (`estimation/map.py`) -- best-of-restarts, labelling, steps clock."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from regimekit.config.schema import EstimationSection
from regimekit.estimation import fit
from regimekit.estimation.init import lattice_jitter, random, rollvol_kmeans
from regimekit.estimation.objective import make_objective
from regimekit.estimation.optimizers import RunRecord
from regimekit.utils.rng import key_for
from tests.helpers.objective_fixtures import make_case


def _cfg(**overrides: object) -> EstimationSection:
    return EstimationSection(**overrides)  # type: ignore[arg-type]


def test_e5_t03_best_of_restarts_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: a restart with an injected lower objective wins; the `restarts` table has `cfg.restarts` rows."""
    case = make_case("gbm", "negbin", 1, t_len=100, seed=5)
    restarts = 4
    cfg = _cfg(optimizer="lbfgs", max_iter=10, restarts=restarts, init=["random"])

    objectives = [10.0, -50.0, 20.0, 5.0]  # restart 1 is the injected winner
    calls: list[int] = []

    def fake_minimize(objective: object, u0: object, cfg_: object, log: object) -> RunRecord:
        i = len(calls)
        calls.append(i)
        return RunRecord(
            objective=objectives[i],
            loglik=-objectives[i],
            n_iter=5,
            converged=True,
            wall_s=0.01,
            reason="tol_f",
            u=u0,  # type: ignore[arg-type]
        )

    monkeypatch.setattr("regimekit.estimation.map.minimize", fake_minimize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = fit(case.spec, case.data, cfg, seed=1)

    assert result.objective == -50.0
    # `result.loglik` is recomputed fresh via `engine.loglik` on the winning theta, not trusted from the
    # (here faked) `RunRecord.loglik` -- only `restarts["loglik"]` reports each run's own record.
    assert result.restarts.loc[1, "loglik"] == 50.0
    assert len(result.restarts) == restarts
    assert list(result.restarts.columns) == [
        "restart",
        "init_strategy",
        "objective",
        "loglik",
        "n_iter",
        "converged",
        "wall_s",
    ]
    assert result.restarts["restart"].tolist() == [0, 1, 2, 3]
    assert result.restarts.loc[1, "objective"] == -50.0


def test_e5_t03_no_guard_warns_and_sorts_by_sigma() -> None:
    """AC5: without an ordering constraint or fixed emission params, a warning fires and states sort."""
    case = make_case("gbm", "negbin", 1, t_len=80, seed=6)
    cfg = _cfg(optimizer="lbfgs", max_iter=5, restarts=1, init=["random"])

    with pytest.warns(UserWarning, match="identifiability guard"):
        result = fit(case.spec, case.data, cfg, seed=2)

    sigma = np.asarray(result.params["emission.sigma"])
    assert np.all(np.diff(sigma) >= 0)


def test_e5_t03_estimator_is_map_and_import_works() -> None:
    """AC6 (part): `FitResult.estimator == "map"`; `from regimekit.estimation import fit` works."""
    from regimekit.estimation import fit as reexported_fit

    assert reexported_fit is fit

    case = make_case("gbm", "geometric", 1, t_len=60, seed=7)
    cfg = _cfg(optimizer="lbfgs", max_iter=5, restarts=1, init=["random"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = fit(case.spec, case.data, cfg, seed=3)

    assert result.estimator == "map"


def test_e5_t03_empty_init_list_defaults_to_random_for_every_restart() -> None:
    """`cfg.init == []` (the schema default) falls back to `["random"]`, not zero restarts."""
    case = make_case("gbm", "geometric", 1, t_len=60, seed=10)
    restarts = 3
    cfg = _cfg(optimizer="lbfgs", max_iter=5, restarts=restarts, init=[])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = fit(case.spec, case.data, cfg, seed=4)

    assert len(result.restarts) == restarts
    assert set(result.restarts["init_strategy"]) == {"random"}


@pytest.mark.parametrize(
    ("emission", "duration"),
    [("gbm", "negbin"), ("gbm", "geometric")],
)
def test_e5_t03_every_strategy_finite_under_steps_clock(emission: str, duration: str) -> None:
    """AC6 (part): every init strategy gives a finite objective on the same data under `clock: steps`."""
    case = make_case(emission, duration, 1, t_len=120, seed=8)
    steps_data = dataclasses.replace(case.data, dt=case.data.dt * 0 + 1.0, clock="steps", nominal_dt=1.0)
    objective = make_objective(case.spec, case.space, case.engine, case.manager, steps_data, use_priors=False)
    key = key_for(0, 1)

    for theta in (
        rollvol_kmeans(steps_data, case.spec, key),
        lattice_jitter(case.spec, key),
        random(steps_data, case.spec, key),
    ):
        u = case.manager.flatten(theta)
        assert np.isfinite(float(objective.value(u)))
