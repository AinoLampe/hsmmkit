"""Typed exception hierarchy (overview §4.4); the CLI maps it to exit codes (`hsmmkit.cli.app`)."""

from __future__ import annotations


class HsmmkitError(Exception):
    """Base class for every hsmmkit exception.

    Args:
        message: Human-readable description of the failure.
        path: Config path (`model.duration.r[2]`) or array/field name the error concerns, if any.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        """Store `path` and prefix it onto `message`."""
        self.path = path
        full_message = f"{path}: {message}" if path is not None else message
        super().__init__(full_message)


class ConfigError(HsmmkitError):
    """Invalid or inconsistent configuration."""


class DataError(HsmmkitError):
    """Invalid data contract, shape, or content."""


class NumericalError(HsmmkitError):
    """A numerical guard (NaN/Inf, disabled x64, missing device, ...) was tripped."""


class PersistenceError(HsmmkitError):
    """A model bundle failed to save, load, or validate."""


class ConstraintError(HsmmkitError):
    """The label / boundary constraints (E12-T06) admit no state path: the likelihood is exactly zero.

    Attributes:
        step: the first step at which every filtered state has probability zero (`None` if not located).
        sequence: the id of the offending sequence of a batch (`None` for a single sequence).
    """

    def __init__(
        self, message: str, *, path: str | None = None, step: int | None = None, sequence: str | None = None
    ) -> None:
        """Store `step` / `sequence` and extend `message` with them."""
        self.step = step
        self.sequence = sequence
        where = []
        if sequence is not None:
            where.append(f"sequence {sequence!r}")
        if step is not None:
            where.append(f"step {step}")
        super().__init__(f"{message} ({', '.join(where)})" if where else message, path=path)
