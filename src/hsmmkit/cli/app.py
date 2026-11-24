"""Typer CLI entry point: thin wrappers over `regimekit.project.Project` (all pass `write=True`)."""

from __future__ import annotations

import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import typer

from regimekit.config.yaml_io import export_json_schema, load_config
from regimekit.data.contracts import PreparedDataset
from regimekit.estimation.selftest import SelftestResult
from regimekit.persistence.results import read_prepared
from regimekit.project import FinanceConfig, Project, apply_overrides, check_simulate_args
from regimekit.utils.errors import EXIT_CODES, DataError, NumericalError, RegimekitError
from regimekit.utils.jax_setup import setup as jax_setup
from regimekit.utils.logging import configure_logging, get_logger

app = typer.Typer(name="regimekit")


@dataclass
class GlobalOptions:
    """CLI-wide options set on the `regimekit` app itself, read by subcommands."""

    out: Path | None = None
    offline: bool = False
    device: str | None = None
    seed: int | None = None
    log_level: str | None = None


_global_options = GlobalOptions()


@app.callback()
def _main_callback(
    out: Path | None = typer.Option(None, "--out", help="Output directory (overrides project.output_dir)."),
    offline: bool = typer.Option(False, "--offline", help="Do not access the network."),
    device: str | None = typer.Option(None, "--device", help="cpu|gpu|auto (overrides estimation.device)"),
    seed: int | None = typer.Option(None, "--seed", help="Overrides project.seed."),
    log_level: str | None = typer.Option(None, "--log-level", help="Overrides project.log_level."),
) -> None:
    """regimekit: JAX-based Hidden Semi-Markov Model regime detection and forecasting."""
    global _global_options
    _global_options = GlobalOptions(out=out, offline=offline, device=device, seed=seed, log_level=log_level)


def _global_overrides(options: GlobalOptions) -> dict[str, object]:
    """The config overrides the global `--seed` / `--device` / `--log-level` options stand for."""
    overrides: dict[str, object] = {}
    if options.seed is not None:
        overrides["project.seed"] = options.seed
    if options.device is not None:
        overrides["estimation.device"] = options.device
    if options.log_level is not None:
        overrides["project.log_level"] = options.log_level
    return overrides


@contextmanager
def _exit_on_error() -> Iterator[None]:
    """Map a `RegimekitError` to its message on stderr and its exit code (overview section 4.4)."""
    try:
        yield
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc


def _project(config_path: Path, *, logging: bool = True) -> Project:
    """Load `config_path`, apply the global options, configure logging, and wrap it in a `Project`."""
    config = cast(FinanceConfig, load_config(config_path, FinanceConfig))
    apply_overrides(config, _global_overrides(_global_options))
    if logging:
        configure_logging(config.project.log_level)
    jax_setup(config.estimation.device)  # --device takes effect before any JAX use
    return Project(
        config,
        config_dir=config_path.resolve().parent,
        out_dir=_global_options.out,
        offline=_global_options.offline,
    )


def _load_prepared(project: Project) -> PreparedDataset:
    """Read `prepared.parquet` from the output directory, running `fetch` first when it is missing."""
    prepared_path = project.out_dir / "prepared.parquet"
    if not prepared_path.exists():
        if project.offline:
            raise DataError(f"no prepared dataset at {prepared_path} and --offline forbids fetching one")
        project.fetch(write=True)
    return read_prepared(prepared_path)


@app.command("validate-config")
def validate_config_cmd(config_path: Path) -> None:
    """Validate `config_path`, print a summary, and write the resolved config alongside it."""
    with _exit_on_error():
        lines = _project(config_path, logging=False).validate(write=True)
    for line in lines:
        typer.echo(line)


@app.command("config-schema")
def config_schema_cmd(out_path: Path) -> None:
    """Write the JSON-Schema of the configuration to `out_path`."""
    export_json_schema(out_path, FinanceConfig)


@app.command("fetch")
def fetch_cmd(config_path: Path) -> None:
    """Fetch data through the cache, prepare it, and write the dataset plus a data-quality report."""
    with _exit_on_error():
        _project(config_path).fetch(write=True)


@app.command("fit")
def fit_cmd(config_path: Path) -> None:
    """Fetch, fit, and write `<name>.rkmodel`, `fit_report.json` and `config.resolved.yaml`."""
    with _exit_on_error():
        project = _project(config_path)
        data = project.fetch(write=True)
        project.fit(data, write=True)


@app.command("decode")
def decode_cmd(
    config_path: Path,
    model: Path = typer.Option(..., "--model", help="Model bundle directory (`<name>.rkmodel`)."),
    augmented: bool = typer.Option(False, "--augmented", help="Also write augmented.parquet."),
) -> None:
    """Decode a fitted model against `config_path`'s data: filtered/smoothed/Viterbi/segments/IC."""
    with _exit_on_error():
        project = _project(config_path)
        data = _load_prepared(project)
        project.decode(Project.load(model), data, write=True, augmented=augmented)


@app.command("forecast")
def forecast_cmd(
    config_path: Path,
    model: Path = typer.Option(..., "--model", help="Model bundle directory (`<name>.rkmodel`)."),
    horizon: int | None = typer.Option(None, "--horizon", help="Steps ahead (default forecast.horizon)."),
    n_sim: int | None = typer.Option(None, "--n-sim", help="Simulated paths (default forecast.n_sim)."),
    scenario: Path | None = typer.Option(None, "--scenario", help="Scenario CSV of future covariates."),
    samples: bool = typer.Option(False, "--samples", help="Also write forecast_samples.parquet."),
) -> None:
    """Monte-Carlo forecast from the end of `config_path`'s data: regime probabilities, quantiles, moments."""
    with _exit_on_error():
        project = _project(config_path)
        data = _load_prepared(project)
        project.forecast(
            Project.load(model),
            data,
            horizon=horizon,
            n_sim=n_sim,
            scenario=scenario,
            samples=samples,
            write=True,
        )


@app.command("simulate")
def simulate_cmd(
    config_path: Path,
    params: str = typer.Option("config", "--params", help="'config' (init values) or 'model'."),
    model: Path | None = typer.Option(None, "--model", help="Model directory for --params model."),
    steps: int | None = typer.Option(None, "--steps", help="Steps per path (default: dataset length)."),
    n_paths: int = typer.Option(1, "--n-paths", help="Number of simulated paths."),
) -> None:
    """Simulate regime/return paths from the configured (or a fitted) model."""
    with _exit_on_error():
        project = _project(config_path)
        check_simulate_args(params, steps, n_paths)  # before any fetch: the dataset may need the network
        data = _load_prepared(project)
        project.simulate(params, model=model, steps=steps, n_paths=n_paths, data=data, write=True)


@app.command("backtest")
def backtest_cmd(config_path: Path) -> None:
    """Walk-forward backtest: metrics, folds, strategy trace and `report.md` in the output directory."""
    with _exit_on_error():
        _project(config_path).backtest(write=True)


@app.command("compare")
def compare_cmd(
    configs: list[Path] = typer.Argument(..., help="Two or more configuration files."),
    allow_mixed: bool = typer.Option(
        False, "--allow-mixed", help="Compare runs on different data snapshots."
    ),
    metric: list[str] = typer.Option([], "--metric", help="Only these metrics (repeatable)."),
    reuse: bool = typer.Option(
        False, "--reuse", help="Use a config's stored backtest instead of re-running."
    ),
) -> None:
    """Backtest several configs and tabulate them (`compare.parquet`, `compare.md` in the output dir)."""
    with _exit_on_error():
        configure_logging(_global_options.log_level or "INFO")
        Project.compare(
            configs,
            allow_mixed=allow_mixed,
            reuse=reuse,
            metrics=metric or None,
            offline=_global_options.offline,
            overrides=_global_overrides(_global_options),
            out_dir=_global_options.out or Path("runs") / "compare",
            write=True,
        )


def _selftest_lines(result: SelftestResult) -> list[str]:
    """The printed report: backend, per-scenario information table, determinism, verdict."""
    lines = [
        f"backend: {result.backend}",
        f"selftest: seed {result.seed}, T={result.t_len}, restarts={result.restarts}",
    ]
    for run in result.runs:
        report = run.report
        lines.append(f"scenario {report.scenario}  (statistical rows are informational at this size)")
        lines.append(f"  {'parameter':<28}{'truth':>10}{'estimate':>12}  {'tolerance':<30}within")
        for c in report.checks:
            within = "yes" if c.passed else ("no" if c.evaluated else "n/a")
            lines.append(f"  {c.name:<28}{c.truth:>10.4g}{c.estimate:>12.4g}  {c.rule:<30}{within}")
        lines.append(f"  all fitted values finite: {'yes' if run.finite else 'NO'}")
        lines.append(
            f"  determinism (bitwise u over two fits): {'identical' if run.deterministic else 'DIFFERS'}"
        )
    lines.append("selftest " + ("passed" if result.passed else "FAILED"))
    return lines


@app.command("selftest")
def selftest_cmd(
    full: bool = typer.Option(
        False, "--full/--quick", help="--quick (default): scenario gbm_negbin; --full: all six scenarios."
    ),
) -> None:
    """Fit simulated data twice on this backend; exit 4 unless the fits are finite and reproducible."""
    with _exit_on_error():
        configure_logging(_global_options.log_level or "INFO")
        result = Project.selftest(
            seed=_global_options.seed if _global_options.seed is not None else 0,
            full=full,
            device=_global_options.device or "cpu",
        )
        for line in _selftest_lines(result):
            typer.echo(line)
        if not result.passed:
            failing = [f"{r.report.scenario}: {c.line()}" for r in result.runs for c in r.non_finite_rows()]
            bad = [r.report.scenario for r in result.runs if not r.passed]
            detail = "; ".join(failing) if failing else "non-reproducible fit"
            raise NumericalError(f"selftest failed for {', '.join(bad)} ({detail})")


def main() -> None:
    """Run the CLI: `RegimekitError`s map to their exit codes, anything else to 1 (traceback at DEBUG)."""
    try:
        app()
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        sys.exit(EXIT_CODES.get(type(exc), 1))
    except Exception as exc:
        get_logger("regimekit.cli").debug("unexpected_error", traceback=traceback.format_exc())
        typer.echo(f"unexpected error: {exc!r} (rerun with --log-level DEBUG for the traceback)", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
