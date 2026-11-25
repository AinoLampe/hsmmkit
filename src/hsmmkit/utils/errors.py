"""Typed exception hierarchy (overview §4.4); the CLI maps it to exit codes (`regimekit.cli.app`)."""

from __future__ import annotations


class RegimekitError(Exception):
    """Base class for every regimekit exception.

    Args:
        message: Human-readable description of the failure.
        path: Config path (`model.duration.r[2]`) or array/field name the error concerns, if any.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        """Store `path` and prefix it onto `message`."""
        self.path = path
        full_message = f"{path}: {message}" if path is not None else message
        super().__init__(full_message)


class ConfigError(RegimekitError):
    """Invalid or inconsistent configuration."""


class DataError(RegimekitError):
    """Invalid data contract, shape, or content."""


class NumericalError(RegimekitError):
    """A numerical guard (NaN/Inf, disabled x64, missing device, ...) was tripped."""


class PersistenceError(RegimekitError):
    """A model bundle failed to save, load, or validate."""
