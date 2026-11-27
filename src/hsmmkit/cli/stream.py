"""`hsmmkit stream`: CSV rows from stdin or a (growing) file -> JSONL regime probabilities and alert events.

Concept refs: Part II §14.11; Part I §5.9. One line per processed row on stdout (`t`, `time`, `filtered`,
`smoothed_lag`, `events`); diagnostics go to stderr so a pipe stays clean. `--state` resumes from, and
checkpoints into, a state directory (`online.checkpoint_every` rows, and on exit or SIGINT / SIGTERM).
"""

from __future__ import annotations

import csv
import json
import signal
import sys
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import typer

from hsmmkit.utils.errors import ConfigError, DataError

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from hsmmkit.cli.app import CliApp
    from hsmmkit.online.session import OnlineSession

POLL_SECONDS = 0.5
EMIT_FIELDS = ("filtered", "smoothed", "events")


def _parse_emit(value: str) -> set[str]:
    fields = {f.strip() for f in value.split(",") if f.strip()}
    unknown = fields - set(EMIT_FIELDS)
    if unknown:
        raise ConfigError(
            f"unknown --emit field(s) {sorted(unknown)}; allowed: {', '.join(EMIT_FIELDS)}", path="--emit"
        )
    return fields


def _lines(stream: IO[str], follow: bool, stop: list[bool]) -> Any:  # noqa: ANN401
    """Yield complete lines; with `follow` wait for more at the end of the file until `stop[0]`."""
    buffer = ""
    while not stop[0]:
        chunk = stream.readline()
        if chunk:
            buffer += chunk
            if buffer.endswith("\n"):
                yield buffer.rstrip("\r\n")
                buffer = ""
            continue
        if not follow:
            break
        time.sleep(POLL_SECONDS)
    if buffer and not follow:
        yield buffer.rstrip("\r\n")


def _emit_line(result: dict[str, Any], emit: set[str]) -> str:
    return json.dumps(
        {
            "t": result["t"],
            "time": result["time"],
            "filtered": result["filtered"] if "filtered" in emit else None,
            "smoothed_lag": result["smoothed_lag"] if "smoothed" in emit else None,
            "events": result["events"] if "events" in emit else [],
        }
    )


def run_stream(
    session: OnlineSession,
    stream: IO[str],
    out: IO[str],
    *,
    emit: set[str],
    every: int,
    strict: bool,
    follow: bool,
    state_dir: Path | None,
    checkpoint_every: int,
) -> int:
    """Process the CSV `stream` (header first) and write JSONL to `out`; returns the number of rows pushed."""
    stop = [False]

    def request_stop(signum: int, frame: Any) -> None:  # noqa: ANN401
        stop[0] = True

    previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        lines = _lines(stream, follow, stop)
        header_line = next(lines, None)
        if header_line is None:
            raise DataError("the input is empty (no header line)", path="--input")
        header = next(csv.reader([header_line]))
        missing = [c for c in session.required_columns if c not in header]
        if missing:
            raise DataError(f"missing column {missing[0]!r} in the input header {header}", path=missing[0])
        pushed = emitted = 0
        for number, line in enumerate(lines, start=2):
            if not line.strip():
                continue
            fields = next(csv.reader([line]))
            try:
                if len(fields) != len(header):
                    raise DataError(
                        f"line {number}: {len(fields)} fields, expected {len(header)}", path="--input"
                    )
                result = session.push(dict(zip(header, fields, strict=True)))
            except DataError as exc:
                if strict:
                    raise
                print(f"skipped line {number}: {exc}", file=sys.stderr)
                continue
            pushed += 1
            if result["filtered"] is None:
                continue
            emitted += 1
            if emitted % every == 0 or result["events"]:
                out.write(_emit_line(result, emit) + "\n")
                out.flush()
            if state_dir is not None and pushed % checkpoint_every == 0:
                session.save(state_dir)
        return pushed
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if state_dir is not None:
            session.save(state_dir)


def add_stream_command(app: CliApp, exit_on_error: Callable[[], AbstractContextManager[None]]) -> None:
    """Register `stream` on `app` (called by `build_app`, so every application inherits it).

    `exit_on_error` is the error-to-exit-code context manager of `hsmmkit.cli.app` (passed in so that this
    module does not import the app module, which `python -m hsmmkit.cli.app` runs as `__main__`).
    """
    ctx = app.context

    @app.command("stream")
    def stream_cmd(
        config_path: Path,
        model: Path = typer.Option(..., "--model", help="Model bundle directory."),
        input_: str = typer.Option("-", "--input", help="CSV file, or - for stdin."),
        follow: bool = typer.Option(False, "--follow", help="Tail a growing file (poll 0.5 s)."),
        state: Path | None = typer.Option(
            None, "--state", help="State directory: resume if it exists, save on exit."
        ),
        emit: str = typer.Option("filtered,smoothed,events", "--emit", help="Fields to emit."),
        every: int = typer.Option(1, "--every", min=1, help="Emit every k-th row (events always)."),
        strict: bool = typer.Option(False, "--strict", help="A malformed row is an error (exit 3)."),
    ) -> None:
        """Stream rows through the online filter: JSONL of filtered / smoothed probabilities and events."""
        with exit_on_error():
            fields = _parse_emit(emit)
            if not model.exists():
                raise ConfigError(f"model bundle not found: {model}", path="--model")
            project = ctx.project(config_path, logging=False)
            fit = ctx.project_cls.load(model)
            nominal_dt = None
            if project.config.data.clock != "steps":
                nominal_dt = ctx.load_prepared(project).nominal_dt
            session = project.online(fit, nominal_dt=nominal_dt)
            if state is not None and (state / "state.json").exists():
                session.load(state)
            online_cfg = project.config.online
            checkpoint = online_cfg.checkpoint_every if online_cfg else 1000
            if input_ == "-":
                run_stream(
                    session, sys.stdin, sys.stdout, emit=fields, every=every, strict=strict,
                    follow=follow, state_dir=state, checkpoint_every=checkpoint,
                )  # fmt: skip
                return
            path = Path(input_)
            if not path.is_file():
                raise DataError(f"input file not found: {path}", path="--input")
            with path.open(encoding="utf-8", newline="") as fh:
                run_stream(
                    session, fh, sys.stdout, emit=fields, every=every, strict=strict,
                    follow=follow, state_dir=state, checkpoint_every=checkpoint,
                )  # fmt: skip
