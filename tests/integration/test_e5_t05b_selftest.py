"""E5-T05b: `regimekit selftest` -- the exit code rests on finite, bitwise-reproducible fits only.

The default-suite tests run the real command at a tiny size (`T=300`, one restart); the ticket's own size
(`T=1500`, two restarts, under two minutes) is the `slow` test at the bottom.
"""

from __future__ import annotations

import dataclasses
import time

import jax.numpy as jnp
import pytest
from click.testing import Result
from typer.testing import CliRunner

from regimekit.estimation import selftest as selftest_mod
from regimekit.estimation.recovery import RecoveryReport, run_scenario
from regimekit.finance.cli import app

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

_runner = CliRunner()


def _invoke(*args: str) -> Result:
    return _runner.invoke(app, ["selftest", *args])


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the quick run so the default suite stays fast."""
    monkeypatch.setattr(selftest_mod, "QUICK_T", 300)
    monkeypatch.setattr(selftest_mod, "QUICK_RESTARTS", 1)
    return


def test_e5_t05b_selftest_exits_0_and_prints_backend_table_and_determinism(tiny: None) -> None:
    """AC1 (tiny size) + AC3: exit 0 although the T=300 recovery rows miss; backend and determinism lines."""
    result = _invoke()
    assert result.exit_code == 0, result.output
    out = result.output
    assert "backend: jax " in out
    assert "x64 on" in out
    assert "scenario gbm_negbin" in out
    assert "emission.scale[0]" in out
    assert "determinism (bitwise u over two fits): identical" in out
    assert "selftest passed" in out


def test_e5_t05b_non_finite_fit_exits_4_and_prints_the_failing_rows(
    tiny: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2: a non-finite estimate gives exit 4 and the failing row on stderr (tolerance rows never gate)."""
    real = selftest_mod.run_scenario

    def _poisoned(*args: object, **kwargs: object) -> RecoveryReport:
        report = real(*args, **kwargs)  # type: ignore[arg-type]
        checks = list(report.checks)
        checks[0] = dataclasses.replace(checks[0], estimate=float("nan"))
        return dataclasses.replace(report, checks=tuple(checks))

    monkeypatch.setattr(selftest_mod, "run_scenario", _poisoned)
    result = _invoke()
    assert result.exit_code == 4, result.output
    assert "all fitted values finite: NO" in result.output
    assert "selftest FAILED" in result.output
    assert "estimate=nan" in result.output  # the failing row, from the NumericalError message


def test_e5_t05b_non_reproducible_fit_exits_4(tiny: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second fit whose `u` differs in the last bit is reported as DIFFERS and fails the run."""
    real = selftest_mod.run_scenario
    calls = {"n": 0}

    def _drifting(*args: object, **kwargs: object) -> RecoveryReport:
        report = real(*args, **kwargs)  # type: ignore[arg-type]
        calls["n"] += 1
        if calls["n"] == 2:
            u = jnp.nextafter(report.fit.uparams, jnp.inf)
            report = dataclasses.replace(report, fit=dataclasses.replace(report.fit, uparams=u))
        return report

    monkeypatch.setattr(selftest_mod, "run_scenario", _drifting)
    result = _invoke()
    assert result.exit_code == 4, result.output
    assert "determinism (bitwise u over two fits): DIFFERS" in result.output
    assert "non-reproducible fit" in result.output


def test_e5_t05b_full_runs_every_scenario(tiny: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--full` visits all six scenarios, one seed (stubbed fit: the plumbing, not the estimator)."""
    base = run_scenario("gbm_negbin", 0, 300, restarts=1)
    seen: list[str] = []

    def _stub(name: str, *args: object, **kwargs: object) -> RecoveryReport:
        seen.append(name)
        return dataclasses.replace(base, scenario=name)

    monkeypatch.setattr(selftest_mod, "run_scenario", _stub)
    result = _invoke("--full")
    assert result.exit_code == 0, result.output
    assert len(set(seen)) == 6
    assert result.output.count("\nscenario ") == 6


def test_e5_t05b_seed_and_device_globals_reach_the_fit(tiny: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The global `--seed` and `--device` options are forwarded to the scenario runs."""
    base = run_scenario("gbm_negbin", 0, 300, restarts=1)
    got: list[tuple[int, str]] = []

    def _stub(name: str, seed: int, *args: object, device: str = "cpu", **kwargs: object) -> RecoveryReport:
        got.append((seed, device))
        return base

    monkeypatch.setattr(selftest_mod, "run_scenario", _stub)
    result = _runner.invoke(app, ["--seed", "7", "--device", "cpu", "selftest"])
    assert result.exit_code == 0, result.output
    assert got == [(7, "cpu"), (7, "cpu")]
    assert "seed 7" in result.output


@pytest.mark.slow
def test_e5_t05b_quick_selftest_at_the_ticket_size_finishes_under_two_minutes() -> None:
    """AC1: `T=1500`, 2 restarts, 1 seed: exit 0 in < 120 s (the statistical rows are informational)."""
    start = time.perf_counter()
    result = _invoke()
    elapsed = time.perf_counter() - start
    assert result.exit_code == 0, result.output
    assert "T=1500, restarts=2" in result.output
    assert elapsed < 120.0, f"selftest took {elapsed:.0f} s"
