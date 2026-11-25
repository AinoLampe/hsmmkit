"""Emission-family contract suite (E11-T01): the checks every emission family must pass.

Usage from any package (built-in or external)::

    from hsmmkit.testing import family_contract
    from hsmmkit.testing.family_contract import FamilyCase, pytest_generate_tests, test_family_contract

    family_contract.register_case(
        FamilyCase(name="my-family", factory=..., theta_factory=..., data_factory=...)
    )

`pytest_generate_tests` + `test_family_contract` (imported into the test module) yield
`test_family_contract[<case>-<check>]`; the checks are also plain functions (`CHECKS`), which raise
`AssertionError` with a readable message, so a failing family can be inspected without pytest.

All tolerances are those of Part II Appendix G / Part I Appendix D (`LOGPDF_ATOL`, `MOMENT_SE`,
`ROUNDTRIP_ATOL`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.model.params.manager import ParamManager
from hsmmkit.utils.errors import DataError

__all__ = [
    "CASES",
    "CHECKS",
    "FamilyCase",
    "check_family",
    "pytest_generate_tests",
    "register_case",
    "test_family_contract",
]

LOGPDF_ATOL = 1e-10  # distribution log-pdf vs scipy (Part I Appendix D)
ROUNDTRIP_ATOL = 1e-12  # theta = theta(u(theta))
MOMENT_SE = 3.0  # simulation moments within 3 SE (Part I Appendix D)
N_SAMPLES = 20_000
N_STATES = 2
T_DATA = 40


@dataclass(frozen=True)
class FamilyCase:
    """One family under test.

    Attributes:
        name: id of the case in the test names.
        factory: `() -> family` (covariate-free).
        theta_factory: `(key, N, M, K) -> theta`, the parameter dict of `N` regimes, `M` series, `K`
            emission covariates (`K = 0`: covariate-free).
        data_factory: `(key, T, M) -> (y, dt, z)` valid for the family's support (`z` may be `None`).
        scipy_logpdf: optional `(y, dt, theta, z) -> (T, N)` reference density.
        covariate_factory: optional `() -> family` whose covariate links read column 0 of `z` (check 5).
        coefficient_names: names of the covariate-coefficient parameters (zeroed by check 5).
        clocks: clocks the family must evaluate under (extensions add theirs).
        n_series: series dimension `M` used by the checks.
        tolerance: overrides `{"logpdf": ..., "roundtrip": ...}`, only with a justification in the case.
    """

    name: str
    factory: Callable[[], Any]
    theta_factory: Callable[[Any, int, int, int], dict[str, jax.Array]]
    data_factory: Callable[[Any, int, int], tuple[jax.Array, jax.Array, jax.Array | None]]
    scipy_logpdf: Callable[..., np.ndarray] | None = None
    covariate_factory: Callable[[], Any] | None = None
    coefficient_names: tuple[str, ...] = ("emission.gamma", "emission.delta")
    clocks: tuple[str, ...] = ("steps", "calendar")
    n_series: int = 2
    tolerance: dict[str, float] = field(default_factory=dict)


CASES: dict[str, FamilyCase] = {}


def register_case(case: FamilyCase) -> FamilyCase:
    """Add `case` to the suite (names are unique)."""
    if case.name in CASES:
        raise ValueError(f"family case '{case.name}' is already registered")
    CASES[case.name] = case
    return case


def _data(case: FamilyCase, n: int = T_DATA) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    return case.data_factory(jax.random.key(1), n, case.n_series)


def _theta(case: FamilyCase, k: int = 0) -> dict[str, jax.Array]:
    return case.theta_factory(jax.random.key(2), N_STATES, case.n_series, k)


# --- the checks ------------------------------------------------------------------------------


def check_log_prob(case: FamilyCase) -> None:
    """1. `log_prob` is `(T, N)` and finite on valid data, and equals `scipy_logpdf` within 1e-10."""
    fam, theta = case.factory(), _theta(case)
    y, dt, z = _data(case)
    lp = np.asarray(fam.log_prob(y, dt, theta, z))
    assert lp.shape == (y.shape[0], N_STATES), f"{case.name}: log_prob shape {lp.shape}, expected (T, N)"
    assert np.all(np.isfinite(lp)), f"{case.name}: log_prob is not finite on valid data"
    if case.scipy_logpdf is not None:
        ref = np.asarray(case.scipy_logpdf(y, dt, theta, z))
        atol = case.tolerance.get("logpdf", LOGPDF_ATOL)
        worst = float(np.max(np.abs(lp - ref)))
        assert worst <= atol, (
            f"{case.name}: log_prob differs from the reference by {worst:.3e} (> {atol:.0e}); "
            "is the normalizing constant right?"
        )


def _valid_support(support: str, s: np.ndarray) -> bool:
    if support == "real":
        return bool(np.all(np.isfinite(s)))
    ints = bool(np.all(np.abs(s - np.rint(s)) == 0))
    if support == "count":
        return ints and bool(np.all(s >= 0))
    if support == "binary":
        return bool(np.all(np.isin(s, (0, 1))))
    return ints and bool(np.all(s >= 0))  # categorical


def check_sample(case: FamilyCase) -> None:
    """2. `sample` is support-valid; mean and variance are within 3 SE of `moments()` when it is analytic."""
    fam, theta = case.factory(), _theta(case)
    _, dt0, z0 = _data(case, 1)
    n = N_SAMPLES
    dt = jnp.full((n,), float(dt0[0]))
    z = None if z0 is None else jnp.tile(z0[:1], (n, 1))
    for i in range(N_STATES):
        regime = jnp.full((n,), i, dtype=jnp.int32)
        s = np.asarray(fam.sample(jax.random.key(10 + i), regime, dt, theta, z))
        assert s.shape == (n, case.n_series), f"{case.name}: sample shape {s.shape}, expected (n, M)"
        assert _valid_support(fam.support, s), f"{case.name}: sample leaves the '{fam.support}' support"
        moments = fam.moments(regime, dt, theta, z)
        if moments is None:
            continue
        mean, var = (np.asarray(m)[0] for m in moments)
        emp_mean, emp_var = s.mean(axis=0), s.var(axis=0)
        for m in range(case.n_series):
            if not np.isfinite(var[m]):
                continue
            se_mean = np.sqrt(var[m] / n)
            assert abs(emp_mean[m] - mean[m]) <= MOMENT_SE * se_mean, (
                f"{case.name}: regime {i}, series {m}: sample mean {emp_mean[m]:.5g} vs moments "
                f"{mean[m]:.5g} (3 SE = {MOMENT_SE * se_mean:.2g})"
            )
            m4 = np.mean((s[:, m] - emp_mean[m]) ** 2 * (s[:, m] - emp_mean[m]) ** 2)
            se_var = np.sqrt(max(m4 - emp_var[m] ** 2, 0.0) / n)
            assert abs(emp_var[m] - var[m]) <= MOMENT_SE * se_var, (
                f"{case.name}: regime {i}, series {m}: sample variance {emp_var[m]:.5g} vs moments "
                f"{var[m]:.5g} (3 SE = {MOMENT_SE * se_var:.2g})"
            )


def check_dt_scaling(case: FamilyCase) -> None:
    """3. `scale_by_dt`: mean and variance are proportional to `dt`; otherwise `log_prob` ignores `dt`."""
    fam, theta = case.factory(), _theta(case)
    y, dt, z = _data(case)
    if not fam.scale_by_dt:
        a = np.asarray(fam.log_prob(y, dt, theta, z))
        b = np.asarray(fam.log_prob(y, 2.0 * dt, theta, z))
        assert np.max(np.abs(a - b)) <= LOGPDF_ATOL, (
            f"{case.name}: log_prob depends on dt without scale_by_dt"
        )
        return
    regime = jnp.zeros((dt.shape[0],), dtype=jnp.int32)
    m1 = fam.moments(regime, dt, theta, z)
    m2 = fam.moments(regime, 2.0 * dt, theta, z)
    if m1 is None or m2 is None:
        pytest.skip(f"{case.name}: moments are not analytic")
    np.testing.assert_allclose(np.asarray(m2[0]), 2.0 * np.asarray(m1[0]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(m2[1]), 2.0 * np.asarray(m1[1]), rtol=1e-10, atol=1e-12)


def check_support_validation(case: FamilyCase) -> None:
    """4. The preparer rejects data outside the family's support (`real` accepts every float)."""
    from hsmmkit.config.schema import DataSection
    from hsmmkit.data.contracts import ObservationSet
    from hsmmkit.data.prepare import Preparer

    support = case.factory().support
    if support == "real":
        pytest.skip(f"{case.name}: every finite float is in the real support")
    bad = {"count": -1.0, "binary": 2.0, "categorical": -1.0}[support]
    values = np.array([[0.0], [1.0], [bad], [1.0], [0.0]])
    obs = ObservationSet(("y",), None, values, "memory", "")
    section = DataSection.model_validate(
        {"source": {"kind": "numpy", "observation_columns": ["y"]}, "clock": "steps"}
    )
    train_end: Any = 5  # time-less data: a step index
    with pytest.raises(DataError):
        Preparer(section, support=support).fit(obs, [], train_end).transform(obs, [])


def check_covariate_links(case: FamilyCase) -> None:
    """5. Zero coefficients reproduce the covariate-free density bitwise; non-zero ones change it."""
    if case.covariate_factory is None:
        pytest.skip(f"{case.name}: no covariate links")
    plain, linked = case.factory(), case.covariate_factory()
    y, dt, z = case.data_factory(jax.random.key(1), T_DATA, case.n_series)
    assert z is not None, f"{case.name}: data_factory must return z for the covariate case"
    theta = case.theta_factory(jax.random.key(2), N_STATES, case.n_series, z.shape[1])
    base = {k: v for k, v in theta.items() if k not in case.coefficient_names}
    zeroed = {k: (jnp.zeros_like(v) if k in case.coefficient_names else v) for k, v in theta.items()}
    free = np.asarray(plain.log_prob(y, dt, base, None))
    assert np.array_equal(np.asarray(linked.log_prob(y, dt, zeroed, z)), free), (
        f"{case.name}: zero coefficients do not reproduce the covariate-free density bitwise"
    )
    assert not np.array_equal(np.asarray(linked.log_prob(y, dt, theta, z)), free), (
        f"{case.name}: non-zero coefficients do not change the density"
    )


def check_param_roundtrip(case: FamilyCase) -> None:
    """6. `ParamManager` round trip `theta = theta(u(theta))` within 1e-12; unique names; jit, grad finite."""
    fam = case.factory()
    specs = fam.param_specs(N_STATES, case.n_series, ())
    names = [s.name for s in specs]
    assert len(set(names)) == len(names), f"{case.name}: duplicate parameter names {names}"
    mgr = ParamManager(specs)
    theta0 = mgr.unflatten(mgr.init_u())
    theta1 = mgr.unflatten(mgr.flatten(theta0))
    atol = case.tolerance.get("roundtrip", ROUNDTRIP_ATOL)
    for name in names:
        worst = float(np.max(np.abs(np.asarray(theta0[name]) - np.asarray(theta1[name]))))
        assert worst <= atol, f"{case.name}: {name} does not round-trip ({worst:.2e} > {atol:.0e})"
    y, dt, z = _data(case)

    @jax.jit
    def total(u: jax.Array) -> jax.Array:
        return jnp.sum(fam.log_prob(y, dt, mgr.unflatten(u), z))

    u0 = mgr.init_u()
    assert bool(jnp.isfinite(total(u0))), f"{case.name}: jitted log_prob is not finite at the initial values"
    g = jax.grad(total)(u0)
    assert bool(jnp.all(jnp.isfinite(g))), f"{case.name}: the gradient of log_prob is not finite"


def check_optional_methods(case: FamilyCase) -> None:
    """7. Optional hooks: `m_step` maximizes the expected complete-data log-likelihood; priors fit."""
    fam = case.factory()
    y, dt, z = _data(case)
    gamma = jnp.asarray(np.random.default_rng(0).dirichlet(np.ones(N_STATES), size=y.shape[0]))
    stats = fam.sufficient_stats(y, dt, gamma, z)
    theta_hat = fam.m_step(stats) if stats is not None else None
    if theta_hat is not None:
        base = _theta(case)
        theta_hat = {**base, **theta_hat}

        def q(th: dict[str, jax.Array]) -> jax.Array:
            return jnp.sum(gamma * fam.log_prob(y, dt, th, z))

        grad = jax.grad(q)(theta_hat)
        worst = max(float(jnp.max(jnp.abs(grad[k]))) for k in theta_hat if k in grad)
        scale = max(1.0, abs(float(q(theta_hat))))
        assert worst / scale <= 1e-8, f"{case.name}: m_step is not a stationary point (|grad| = {worst:.2e})"
    prior = fam.conjugate_prior()
    if prior is not None:
        assert hasattr(prior, "update"), f"{case.name}: conjugate_prior() must provide update(stats)"
    else:
        pytest.skip(f"{case.name}: no optional estimation hooks")


def check_clocks(case: FamilyCase) -> None:
    """8. The density evaluates under every clock of `case.clocks`, without branching on clock names."""
    from hsmmkit.testing.clock_contract import fixture_times
    from hsmmkit.utils.clock import get_clock, parse_clock_options

    fam, theta = case.factory(), _theta(case)
    times = fixture_times()
    n = times.shape[0] - 1
    for name in case.clocks:
        clock = get_clock(name)
        dt = jnp.asarray(clock.dt(times, n, parse_clock_options(name, None)))
        y, _, z = case.data_factory(jax.random.key(1), n, case.n_series)
        lp = np.asarray(fam.log_prob(y, dt, theta, z))
        assert lp.shape == (n, N_STATES), f"{case.name}: clock '{name}': log_prob shape {lp.shape}"
        assert np.all(np.isfinite(lp)), f"{case.name}: clock '{name}': log_prob is not finite"


CHECKS: dict[str, Callable[[FamilyCase], None]] = {
    "log_prob": check_log_prob,
    "sample": check_sample,
    "dt_scaling": check_dt_scaling,
    "support_validation": check_support_validation,
    "covariate_links": check_covariate_links,
    "param_roundtrip": check_param_roundtrip,
    "optional_methods": check_optional_methods,
    "clocks": check_clocks,
}


def check_family(case: FamilyCase, check: str | None = None) -> None:
    """Run one check (or all) on `case`; raises `AssertionError` with a readable message on failure."""
    for name, fn in CHECKS.items():
        if check in (None, name):
            fn(case)


# --- pytest integration ------------------------------------------------------------------------


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize `test_family_contract` over every registered case and check (`<case>-<check>`)."""
    if "family_case" in metafunc.fixturenames and "family_check" in metafunc.fixturenames:
        params = [(c, k) for c in CASES.values() for k in CHECKS]
        ids = [f"{c.name}-{k}" for c, k in params]
        metafunc.parametrize(("family_case", "family_check"), params, ids=ids)


def test_family_contract(family_case: FamilyCase, family_check: str) -> None:
    """One check of the contract on one family case."""
    CHECKS[family_check](family_case)
