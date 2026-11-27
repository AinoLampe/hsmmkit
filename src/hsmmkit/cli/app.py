"""The CLI app factory (E10-T07b): `build_app` makes a Typer app over a `Project` class and its config class.

The core ships a complete domain-neutral application (`hsmmkit`): `validate-config`, `config-schema`,
`fit`, `decode`, `forecast`, `simulate`, `backtest`, `compare` and `selftest`, all taking their data from
`data.source`. An application builds its own app with extra commands and registers its own exit codes (see
`regimekit`, an example extension).
"""

from __future__ import annotations

import importlib
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

from hsmmkit.config.schema import ProjectConfig
from hsmmkit.config.yaml_io import export_json_schema, load_config
from hsmmkit.data.contracts import PreparedDataset
from hsmmkit.estimation.selftest import SelftestResult
from hsmmkit.persistence.results import read_prepared
from hsmmkit.presets import get_preset, list_presets, write_preset
from hsmmkit.project import Project, apply_overrides, check_simulate_args
from hsmmkit.utils.errors import (
    ConfigError,
    ConstraintError,
    DataError,
    HsmmkitError,
    NumericalError,
    PersistenceError,
)
from hsmmkit.utils.jax_setup import setup as jax_setup
from hsmmkit.utils.logging import configure_logging, get_logger

EXIT_CODES: dict[type[BaseException], int] = {
    ConfigError: 2,
    DataError: 3,
    ConstraintError: 3,  # Part II overview §5.4: data / constraint
    NumericalError: 4,
    PersistenceError: 1,  # CLAUDE.md §3.8 (the ticket text says 2; the existing behaviour is kept)
}
"""Exit codes by exception class (others exit with 1); applications add theirs with `register_exit_code`."""


def register_exit_code(exc: type[BaseException], code: int) -> None:
    """Map `exc` (and its subclasses, unless they have their own entry) to the exit `code`."""
    EXIT_CODES[exc] = code


def exit_code_of(exc: BaseException) -> int:
    """The exit code of `exc`: the entry of its most specific registered class, else 1."""
    for cls in type(exc).__mro__:
        if cls in EXIT_CODES:
            return EXIT_CODES[cls]
    return 1


@dataclass
class GlobalOptions:
    """The options of the app callback, read by the subcommands."""

    out: Path | None = None
    offline: bool = False
    device: str | None = None
    seed: int | None = None
    log_level: str | None = None


@dataclass
class CliContext:
    """What the commands of one app share: its classes and the parsed global options."""

    name: str
    config_cls: type[ProjectConfig]
    project_cls: type[Project]
    options: GlobalOptions = field(default_factory=GlobalOptions)

    def overrides(self) -> dict[str, object]:
        """The config overrides the global `--seed` / `--device` / `--log-level` options stand for."""
        found: dict[str, object] = {}
        if self.options.seed is not None:
            found["project.seed"] = self.options.seed
        if self.options.device is not None:
            found["estimation.device"] = self.options.device
        if self.options.log_level is not None:
            found["project.log_level"] = self.options.log_level
        return found

    def project(self, config_path: Path, *, logging: bool = True) -> Project:
        """Load `config_path`, apply the global options, set up logging and wrap it in the project class."""
        config = load_config(config_path, self.config_cls)
        apply_overrides(config, self.overrides())
        if logging:
            configure_logging(config.project.log_level)
        jax_setup(config.estimation.device)  # --device takes effect before any JAX use
        return self.project_cls(
            config,
            config_dir=config_path.resolve().parent,
            out_dir=self.options.out,
            offline=self.options.offline,
        )

    def load_prepared(self, project: Project) -> PreparedDataset:
        """Read `prepared.parquet` from the output directory, loading (and writing) the data when missing."""
        prepared_path = project.out_dir / "prepared.parquet"
        if not prepared_path.exists():
            if project.offline and project.needs_network():
                raise DataError(f"no prepared dataset at {prepared_path} and --offline forbids fetching one")
            project.load_data(write=True)
        return read_prepared(prepared_path)


class CliApp(typer.Typer):
    """A Typer app that carries its `CliContext` (extra commands read it)."""

    context: CliContext


@contextmanager
def exit_on_error() -> Iterator[None]:
    """Map a `HsmmkitError` to its message on stderr and its exit code (overview section 4.4)."""
    try:
        yield
    except HsmmkitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=exit_code_of(exc)) from exc


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


def _import_config_class(dotted: str) -> type[ProjectConfig]:
    """The `ProjectConfig` subclass at `module.Class`.

    Raises:
        ConfigError: the module or class does not exist, or is not a `ProjectConfig` subclass.
    """
    module, _, cls_name = dotted.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module), cls_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ConfigError(f"cannot import config class '{dotted}': {exc}", path="--config-class") from exc
    if not (isinstance(cls, type) and issubclass(cls, ProjectConfig)):
        raise ConfigError(f"'{dotted}' is not a ProjectConfig subclass", path="--config-class")
    return cls


def build_app(
    *,
    name: str,
    config_cls: type[ProjectConfig],
    project_cls: type[Project],
    extra_commands: Sequence[Callable[[typer.Typer], None]] = (),
) -> CliApp:
    """A CLI over `project_cls` / `config_cls`; `extra_commands` register further commands on the app.

    Each extra command is a function of the Typer app (a `CliApp` with its `context`).
    """
    app = CliApp(name=name)
    ctx = CliContext(name=name, config_cls=config_cls, project_cls=project_cls)
    app.context = ctx

    @app.callback()
    def _main_callback(
        out: Path | None = typer.Option(
            None, "--out", help="Output directory (overrides project.output_dir)."
        ),
        offline: bool = typer.Option(
            False, "--offline", help="Do not access the network (ignored by sources that have none)."
        ),
        device: str | None = typer.Option(
            None, "--device", help="cpu|gpu|auto (overrides estimation.device)"
        ),
        seed: int | None = typer.Option(None, "--seed", help="Overrides project.seed."),
        log_level: str | None = typer.Option(None, "--log-level", help="Overrides project.log_level."),
    ) -> None:
        """Hidden semi-Markov model regime detection and forecasting."""
        ctx.options = GlobalOptions(out=out, offline=offline, device=device, seed=seed, log_level=log_level)

    @app.command("validate-config")
    def validate_config_cmd(config_path: Path) -> None:
        """Validate `config_path`, print a summary, and write the resolved config alongside it."""
        with exit_on_error():
            lines = ctx.project(config_path, logging=False).validate(write=True)
        for line in lines:
            typer.echo(line)

    @app.command("config-schema")
    def config_schema_cmd(
        out_path: Path,
        config_class: Annotated[
            str | None, typer.Option("--config-class", help="dotted path of a ProjectConfig subclass")
        ] = None,
    ) -> None:
        """Write the JSON-Schema of the configuration (or of `--config-class`) to `out_path`."""
        with exit_on_error():
            export_json_schema(out_path, _import_config_class(config_class) if config_class else config_cls)

    presets_app = typer.Typer(help="Preset configurations with synthetic data.", no_args_is_help=True)
    app.add_typer(presets_app, name="presets")

    @presets_app.command("list")
    def presets_list_cmd() -> None:
        """List the registered presets."""
        for preset in list_presets():
            typer.echo(f"{preset.name}\t{preset.description}")

    @presets_app.command("show")
    def presets_show_cmd(name: str) -> None:
        """Print the YAML of preset NAME."""
        with exit_on_error():
            typer.echo(get_preset(name).yaml, nl=False)

    @presets_app.command("write")
    def presets_write_cmd(name: str, directory: Path, seed: int = 0) -> None:
        """Write preset NAME (YAML and, if it has one, its synthetic data files) into DIRECTORY."""
        with exit_on_error():
            for path in write_preset(name, directory, seed=seed):
                typer.echo(str(path))

    @app.command("fit")
    def fit_cmd(config_path: Path) -> None:
        """Load the data, fit, and write the model bundle, `fit_report.json` and `config.resolved.yaml`."""
        with exit_on_error():
            project = ctx.project(config_path)
            data = project.load_data(write=True)
            project.fit(data, write=True)

    @app.command("decode")
    def decode_cmd(
        config_path: Path,
        model: Path = typer.Option(..., "--model", help="Model bundle directory."),
        augmented: bool = typer.Option(False, "--augmented", help="Also write augmented.parquet."),
    ) -> None:
        """Decode a fitted model against `config_path`'s data: filtered/smoothed/Viterbi/segments/IC."""
        with exit_on_error():
            project = ctx.project(config_path)
            data = ctx.load_prepared(project)
            project.decode(project_cls.load(model), data, write=True, augmented=augmented)

    @app.command("forecast")
    def forecast_cmd(
        config_path: Path,
        model: Path = typer.Option(..., "--model", help="Model bundle directory."),
        horizon: int | None = typer.Option(None, "--horizon", help="Steps ahead (default forecast.horizon)."),
        n_sim: int | None = typer.Option(None, "--n-sim", help="Simulated paths (default forecast.n_sim)."),
        scenario: Path | None = typer.Option(None, "--scenario", help="Scenario CSV of future covariates."),
        samples: bool = typer.Option(False, "--samples", help="Also write forecast_samples.parquet."),
    ) -> None:
        """Monte-Carlo forecast from the end of `config_path`'s data: regime probabilities, quantiles."""
        with exit_on_error():
            project = ctx.project(config_path)
            data = ctx.load_prepared(project)
            project.forecast(
                project_cls.load(model),
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
        """Simulate regime/observation paths from the configured (or a fitted) model."""
        with exit_on_error():
            project = ctx.project(config_path)
            check_simulate_args(params, steps, n_paths)  # before any data is loaded: it may need the network
            data = ctx.load_prepared(project)
            project.simulate(params, model=model, steps=steps, n_paths=n_paths, data=data, write=True)

    @app.command("backtest")
    def backtest_cmd(config_path: Path) -> None:
        """Walk-forward backtest: metrics, folds, the hooks' frames and `report.md` in the output dir."""
        with exit_on_error():
            ctx.project(config_path).backtest(write=True)

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
        with exit_on_error():
            configure_logging(ctx.options.log_level or "INFO")
            project_cls.compare(
                configs,
                allow_mixed=allow_mixed,
                reuse=reuse,
                metrics=metric or None,
                offline=ctx.options.offline,
                overrides=ctx.overrides(),
                out_dir=ctx.options.out or Path("runs") / "compare",
                write=True,
            )

    @app.command("selftest")
    def selftest_cmd(
        full: bool = typer.Option(
            False, "--full/--quick", help="--quick (default): one scenario; --full: all six scenarios."
        ),
    ) -> None:
        """Fit simulated data twice on this backend; exit 4 unless the fits are finite and reproducible."""
        with exit_on_error():
            configure_logging(ctx.options.log_level or "INFO")
            result: SelftestResult = project_cls.selftest(
                seed=ctx.options.seed if ctx.options.seed is not None else 0,
                full=full,
                device=ctx.options.device or "cpu",
            )
            for line in _selftest_lines(result):
                typer.echo(line)
            if not result.passed:
                failing = [
                    f"{r.report.scenario}: {c.line()}" for r in result.runs for c in r.non_finite_rows()
                ]
                bad = [r.report.scenario for r in result.runs if not r.passed]
                detail = "; ".join(failing) if failing else "non-reproducible fit"
                raise NumericalError(f"selftest failed for {', '.join(bad)} ({detail})")

    from hsmmkit.cli.stream import add_stream_command

    add_stream_command(app, exit_on_error)
    for add in extra_commands:
        add(app)
    return app


def run(app: typer.Typer) -> None:
    """Run `app`: `HsmmkitError`s map to their exit codes, anything else to 1 (traceback at DEBUG)."""
    try:
        app()
    except HsmmkitError as exc:
        typer.echo(str(exc), err=True)
        sys.exit(exit_code_of(exc))
    except Exception as exc:
        get_logger("hsmmkit.cli").debug("unexpected_error", traceback=traceback.format_exc())
        typer.echo(f"unexpected error: {exc!r} (rerun with --log-level DEBUG for the traceback)", err=True)
        sys.exit(1)


app = build_app(name="hsmmkit", config_cls=ProjectConfig, project_cls=Project)


def main() -> None:
    """Entry point of the core console script `hsmmkit`."""
    run(app)


if __name__ == "__main__":  # pragma: no cover
    main()
