"""`hsmmkit selftest`: a fast check that the installed package estimates on this machine and backend.

The exit decision rests on non-statistical checks only (owner decision for E5-T05b): every fit is finite,
and fitting the same data twice gives a bitwise-identical `u`. At the quick size (`T=1500`, about 15-20 regime
visits per path) the Appendix D recovery tolerances cannot be met reliably even by an oracle that reads the
true path, so the truth/estimate/tolerance rows are reported for information and never gate the exit code.
The tolerances themselves are checked by the recovery suite (`estimation.recovery`) at an attainable size.

Pure functions: no file I/O and no printing; `cli.app` formats the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from hsmmkit.estimation.recovery import SCENARIOS, ParamCheck, RecoveryReport, run_scenario

QUICK_SCENARIO = "gbm_negbin"
QUICK_T = 1500
QUICK_RESTARTS = 2


@dataclass(frozen=True)
class SelftestRun:
    """One scenario: the first fit's report plus the two gating checks."""

    report: RecoveryReport
    finite: bool
    deterministic: bool

    @property
    def passed(self) -> bool:
        """The gate for this scenario: finite fit and a bitwise-reproducible `u`."""
        return self.finite and self.deterministic

    def non_finite_rows(self) -> list[ParamCheck]:
        """Rows whose estimate is not finite (the failing rows when `finite` is false)."""
        return [c for c in self.report.checks if not np.isfinite(c.estimate)]


@dataclass(frozen=True)
class SelftestResult:
    """All scenarios of one `selftest` invocation and the backend they ran on."""

    runs: tuple[SelftestRun, ...]
    seed: int
    t_len: int
    restarts: int
    backend: str

    @property
    def passed(self) -> bool:
        """True when every scenario passed its gate."""
        return all(r.passed for r in self.runs)


def backend_line() -> str:
    """`jax <version>, device <device>, x64 <on|off>` for the current process."""
    x64 = "on" if jax.config.jax_enable_x64 else "off"
    return f"jax {jax.__version__}, device {jax.devices()[0]}, x64 {x64}"


def _is_finite(report: RecoveryReport) -> bool:
    fit = report.fit
    return bool(
        np.isfinite(fit.loglik)
        and np.isfinite(fit.objective)
        and np.all(np.isfinite(np.asarray(fit.uparams)))
        and all(np.isfinite(c.estimate) for c in report.checks)
    )


def run_selftest(*, seed: int = 0, full: bool = False, device: str = "cpu") -> SelftestResult:
    """Fit the quick scenario (`full`: all six scenarios) twice at `T=1500`, `restarts=2`, one seed.

    The second fit of identical data and seed must reproduce `u` bitwise (determinism, overview §4.7).
    `QUICK_T` / `QUICK_RESTARTS` are read at call time.

    Args:
        seed: Simulation and fit seed.
        full: Run every recovery scenario instead of only `gbm_negbin`.
        device: `cpu`, `gpu` or `auto`.

    Returns:
        A `SelftestResult`; `passed` is the exit decision.
    """
    names = list(SCENARIOS) if full else [QUICK_SCENARIO]
    runs = []
    for name in names:
        first = run_scenario(name, seed, QUICK_T, restarts=QUICK_RESTARTS, device=device)
        second = run_scenario(name, seed, QUICK_T, restarts=QUICK_RESTARTS, device=device)
        same = bool(np.array_equal(np.asarray(first.fit.uparams), np.asarray(second.fit.uparams)))
        runs.append(SelftestRun(first, _is_finite(first), same))
    return SelftestResult(tuple(runs), seed, QUICK_T, QUICK_RESTARTS, backend_line())
