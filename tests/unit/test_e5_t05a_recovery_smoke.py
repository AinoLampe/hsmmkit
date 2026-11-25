"""E5-T05a: fast structural tests for `estimation.recovery` (default suite, not `-m slow`).

Tiny `t_len` so these run in the default suite and keep `estimation`'s coverage >= 85% (CLAUDE.md
§3.2) without paying for the nightly suite's `T=5000` fits; they check the harness's plumbing
(shapes, check bookkeeping, the empirical-SE fallback), not tolerance recovery.
"""

from __future__ import annotations

import json
import os

import pytest

from regimekit.estimation.recovery import (
    SCENARIOS,
    ParamCheck,
    RecoveryReport,
    SuiteResult,
    _default_t_len,
    run_recovery_suite,
    run_scenario,
)

_SMOKE_T = 120


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_e5_t05a_recovery_run_scenario_well_formed(name: str) -> None:
    """Every scenario simulates, fits and produces a non-empty, well-typed `RecoveryReport`."""
    report = run_scenario(name, seed=0, t_len=_SMOKE_T)
    assert isinstance(report, RecoveryReport)
    assert report.scenario == name
    assert report.t_len == _SMOKE_T
    assert len(report.checks) > 0
    for check in report.checks:
        assert isinstance(check, ParamCheck)
        assert check.line()  # must not raise, must be non-empty
    # sigma/mean-sojourn/jump checks never need an SE, so they are always evaluated.
    always_evaluated = [c for c in report.checks if "sigma" in c.name or "mean_sojourn" in c.name]
    assert always_evaluated
    assert all(c.evaluated for c in always_evaluated)


def test_e5_t05a_recovery_mu_fixed_scenario_has_no_mu_checks() -> None:
    """Scenario 6 (`fixed_drift`): `emission.mean` is fixed, so no `mu` tolerance check is emitted."""
    report = run_scenario("fixed_drift", seed=0, t_len=_SMOKE_T)
    assert not any(c.name.startswith("emission.mean") for c in report.checks)


def test_e5_t05a_recovery_empirical_se_fills_in_missing_hessian() -> None:
    """`explicit_gamma`'s Hessian is unavailable (no JVP for `igamma`'s shape derivative).

    `run_recovery_suite` must still evaluate its `mu` checks via the across-seed empirical spread.
    """
    suite = run_recovery_suite("explicit_gamma", seeds=(0, 1), t_len=_SMOKE_T)
    assert len(suite.reports) == 2
    mu_checks = [c for r in suite.reports for c in r.checks if c.name.startswith("emission.mean")]
    assert mu_checks
    assert all(c.evaluated for c in mu_checks)
    assert all(c.se is not None for c in mu_checks)
    # n_pass/passed/failures() must not raise regardless of outcome.
    assert suite.n_pass == sum(r.passed for r in suite.reports)
    suite.failures()


def test_e5_t05a_recovery_t_len_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: `REGIMEKIT_RECOVERY_T` controls the default `t_len` (nightly-suite sizing knob)."""
    monkeypatch.setenv("REGIMEKIT_RECOVERY_T", "777")
    assert _default_t_len() == 777
    monkeypatch.delenv("REGIMEKIT_RECOVERY_T", raising=False)
    assert _default_t_len() == 20000


def test_e5_t05a_recovery_t_len_env_var_used_when_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_scenario(t_len=None)` reads the env var (kept tiny here to stay in the default suite)."""
    monkeypatch.setenv("REGIMEKIT_RECOVERY_T", str(_SMOKE_T))
    report = run_scenario("gbm_negbin", seed=0, t_len=None)
    assert report.t_len == _SMOKE_T
    assert os.environ["REGIMEKIT_RECOVERY_T"] == str(_SMOKE_T)


def _report(seed: int, failing: set[str], skip_mu: bool = False) -> RecoveryReport:
    """A fabricated report with one check per category; the categories in `failing` fail."""
    names = {
        "sigma": "emission.scale[0]",
        "mu": "emission.mean[0]",
        "mean_sojourn": "mean_sojourn[0]",
        "jump": "pi[0->1](r=0)",
        "coefficient": "transition.b[0]",
    }
    checks = tuple(
        ParamCheck(n, 1.0, 1.0, None, "rule", not (cat == "mu" and skip_mu), cat not in failing)
        for cat, n in names.items()
    )
    return RecoveryReport("fake", seed, 100, checks, None)  # type: ignore[arg-type]


def test_e5_t05a_recovery_check_categories() -> None:
    """Check names map to the Appendix D tolerance categories."""
    cats = {c.name: c.category for c in _report(0, set()).checks}
    assert cats == {
        "emission.scale[0]": "sigma",
        "emission.mean[0]": "mu",
        "mean_sojourn[0]": "mean_sojourn",
        "pi[0->1](r=0)": "jump",
        "transition.b[0]": "coefficient",
    }


def test_e5_t05a_recovery_each_category_needs_four_of_five_seeds_on_its_own() -> None:
    """Different categories failing in different seeds still pass; one category failing twice fails."""
    scattered = [
        _report(0, {"sigma"}),
        _report(1, {"mu"}),
        _report(2, {"mean_sojourn"}),
        _report(3, set()),
        _report(4, set()),
    ]
    ok = SuiteResult("fake", tuple(scattered), n_pass=sum(r.passed for r in scattered))
    assert ok.n_pass == 2  # the stricter all-checks-in-one-seed count would fail
    assert ok.passed
    assert ok.categories["sigma"] == (4, 5)

    twice = [_report(0, {"mean_sojourn"}), _report(1, {"mean_sojourn"})] + [
        _report(s, set()) for s in (2, 3, 4)
    ]
    bad = SuiteResult("fake", tuple(twice), n_pass=3)
    assert not bad.passed
    assert bad.categories["mean_sojourn"] == (3, 5)
    message = bad.failures()
    assert "mean_sojourn: 3/5" in message
    assert "seed 0" in message
    assert "sigma" not in message


def test_e5_t05a_recovery_unevaluated_categories_are_not_counted() -> None:
    """A category with no evaluated check in a seed (e.g. `mu` without a SE) is not in its denominator."""
    reports = [_report(s, set(), skip_mu=(s < 2)) for s in range(5)]
    suite = SuiteResult("fake", tuple(reports), n_pass=5)
    assert suite.categories["mu"] == (3, 3)
    assert suite.passed


def test_e5_t05a_recovery_order2_default_t_is_scaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """`order2` runs at 2.5x the default `T` (an explicit `t_len` is never scaled)."""
    assert SCENARIOS["order2"].t_scale == 2.5
    assert all(s.t_scale == 1.0 for n, s in SCENARIOS.items() if n != "order2")
    monkeypatch.setenv("REGIMEKIT_RECOVERY_T", "60")
    assert run_scenario("order2", seed=0).t_len == 150
    assert run_scenario("order2", seed=0, t_len=_SMOKE_T).t_len == _SMOKE_T


def test_e5_t05a_recovery_report_only_categories_do_not_decide_the_outcome() -> None:
    """`explicit_gamma` reports mean sojourn without gating on it; summary() lists it as report-only."""
    assert SCENARIOS["explicit_gamma"].report_only == ("mean_sojourn",)
    assert all(s.report_only == () for n, s in SCENARIOS.items() if n != "explicit_gamma")

    reports = [_report(s, {"mean_sojourn"} if s < 3 else set()) for s in range(5)]
    gating = SuiteResult("fake", tuple(reports), n_pass=2)
    assert not gating.passed  # mean_sojourn: 2/5 seeds
    only = SuiteResult("fake", tuple(reports), n_pass=2, report_only=("mean_sojourn",))
    assert only.passed
    assert only.failures() == "no failing category"
    assert only.categories["mean_sojourn"] == (2, 5)  # still reported

    # a gating category still fails the suite
    bad = [_report(s, {"sigma"}) for s in range(2)] + [_report(s, set()) for s in range(2, 5)]
    assert not SuiteResult("fake", tuple(bad), n_pass=3, report_only=("mean_sojourn",)).passed

    summary = only.summary()
    assert summary["categories"]["mean_sojourn"] == {  # type: ignore[index]
        "passing_seeds": 2,
        "evaluated_seeds": 5,
        "report_only": True,
    }
    assert summary["passed"] is True
    assert len(summary["checks"]) == 25  # type: ignore[arg-type]
    json.dumps(summary)  # serializable
