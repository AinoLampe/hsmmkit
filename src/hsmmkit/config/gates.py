"""Feature gates (E10-T10, Part II overview §5.11).

The schema accepts every key of the final `hsmmkit` schema 1 from E10-T10 on, so presets and the JSON
Schema validate. A key whose implementation has not landed is rejected when the model is **built**
(`check`), not when the config is validated. Each ticket that implements a feature removes its own row
from `GATES` and adds a test that the key now builds.
"""

from __future__ import annotations

from collections.abc import Callable

from hsmmkit.config.schema import ProjectConfig
from hsmmkit.utils.errors import ConfigError

__all__ = ["GATES", "check", "used_gates"]

Gate = tuple[Callable[[ProjectConfig], bool], str]


GATES: dict[str, Gate] = {
    "estimation.method=em": (lambda c: c.estimation.method == "em", "E13-T02a"),
    "estimation.method=gem": (lambda c: c.estimation.method == "gem", "E13-T03a"),
    "estimation.method=vb": (lambda c: c.estimation.method == "vb", "E13-T05a"),
    "estimation.method=mcmc": (lambda c: c.estimation.method == "mcmc", "E13-T06a"),
    "evaluation.cv": (lambda c: c.evaluation.cv is not None, "E14-T01"),
    "evaluation.grid": (lambda c: c.evaluation.grid is not None, "E14-T02"),
    "online": (lambda c: c.online is not None, "E15-T01"),
    "online.lag": (lambda c: c.online is not None and c.online.lag > 0, "E15-T02"),
    "online.alerts": (lambda c: c.online is not None and bool(c.online.alerts), "E15-T03"),
}
"""Key path (or `key=value`) -> (predicate "the feature is used", ticket that removes the row). Ordered."""


def used_gates(cfg: ProjectConfig) -> list[tuple[str, str]]:
    """`(path, ticket)` of every gated feature that `cfg` uses, in the order of `GATES`."""
    return [(path, ticket) for path, (used, ticket) in GATES.items() if used(cfg)]


def check(cfg: ProjectConfig) -> None:
    """Reject the first gated feature that `cfg` uses.

    Raises:
        ConfigError: `path` = the gated key, "not available before <ticket>".
    """
    used = used_gates(cfg)
    if used:
        path, ticket = used[0]
        raise ConfigError(f"not available before {ticket}", path=path.partition("=")[0])
