"""Feature gates (E10-T10, Part II overview §5.11).

The schema accepts every key of the final `hsmmkit` schema 1 from E10-T10 on, so presets and the JSON
Schema validate. A key whose implementation has not landed is rejected when the model is **built**
(`check`), not when the config is validated. Each ticket that implements a feature removes its own row
from `GATES` and adds a test that the key now builds.
"""

from __future__ import annotations

from collections.abc import Callable

from hsmmkit.config.schema import BandedCfg, MatrixCfg, ProjectConfig
from hsmmkit.utils.errors import ConfigError

__all__ = ["GATES", "PLANNED_METRICS", "check", "used_gates"]

PLANNED_METRICS: dict[str, str] = {
    "boundary_f1": "E14-T03",
    "hausdorff": "E14-T03",
    "covering": "E14-T03",
    "hamming_matched": "E14-T03",
    "ari": "E14-T03",
    "nmi": "E14-T03",
}
"""Metric names that the schema accepts before the metric exists (E14-T03); `check` rejects them at build."""

Gate = tuple[Callable[[ProjectConfig], bool], str]


def _topology_not_full(c: ProjectConfig) -> bool:
    t = c.model.topology
    return isinstance(t, (BandedCfg, MatrixCfg)) or t != "full"


GATES: dict[str, Gate] = {
    "data.sequences": (lambda c: c.data.sequences is not None, "E12-T01a"),
    "data.source.sequence_column": (
        lambda c: c.data.source.option("sequence_column") is not None,
        "E12-T01a",
    ),
    "model.initial_groups": (lambda c: c.model.initial_groups == "per_group", "E12-T02"),
    "model.topology": (_topology_not_full, "E12-T03"),
    "model.left_censored": (lambda c: c.model.left_censored and c.model.engine == "explicit", "E12-T04"),
    "model.duration.per_state": (lambda c: c.model.duration.per_state is not None, "E12-T05"),
    "model.duration.duration_dependent_jumps": (
        lambda c: c.model.duration.duration_dependent_jumps,
        "E12-T05",
    ),
    "data.labels": (lambda c: c.data.labels is not None, "E12-T06"),
    "estimation.init": (lambda c: "labels" in c.estimation.init, "E12-T06"),
    "model.emission.covariance": (lambda c: c.model.emission.covariance == "full", "E11-T02"),
    "model.emission.mixture": (lambda c: c.model.emission.mixture is not None, "E11-T06"),
    "estimation.method=em": (lambda c: c.estimation.method == "em", "E13-T02a"),
    "estimation.method=gem": (lambda c: c.estimation.method == "gem", "E13-T03a"),
    "estimation.method=vb": (lambda c: c.estimation.method == "vb", "E13-T05a"),
    "estimation.method=mcmc": (lambda c: c.estimation.method == "mcmc", "E13-T06a"),
    "evaluation.cv": (lambda c: c.evaluation.cv is not None, "E14-T01"),
    "evaluation.grid": (lambda c: c.evaluation.grid is not None, "E14-T02"),
    "evaluation.metrics": (lambda c: any(m in PLANNED_METRICS for m in c.evaluation.metrics), "E14-T03"),
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
