"""Structured logging (structlog): JSON in CI, human-readable console otherwise."""

from __future__ import annotations

import logging
import os

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog: JSON renderer when env `CI` is set, console renderer otherwise."""
    renderer = (
        structlog.processors.JSONRenderer() if os.environ.get("CI") else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level.upper()]),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.types.FilteringBoundLogger:
    """Return a structlog logger bound to `name`."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
