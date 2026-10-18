"""Placeholders for the strategy metrics that E7-T03 owns (`sharpe`, `max_drawdown`).

Appendix A's `evaluation.metrics` list names them, and E7-T01 AC4 requires every name in that list to
resolve, so they are registered here with `needs={"strategy"}`. A runner without a strategy trace skips
them; calling one directly raises. E7-T03 replaces both with the real implementations
(`register(..., replace=True)`) and removes this module.
"""

from __future__ import annotations

from regimekit.evaluation.metrics.registry import metric
from regimekit.utils.errors import ConfigError


def _pending(name: str) -> float:
    raise ConfigError(
        f"strategy metric {name!r} is implemented by E7-T03 and not available yet", path="evaluation.metrics"
    )


@metric("sharpe", {"strategy"})
def sharpe(**ctx: object) -> float:
    """Annualized Sharpe ratio of the strategy (placeholder until E7-T03)."""
    del ctx
    return _pending("sharpe")


@metric("max_drawdown", {"strategy"})
def max_drawdown(**ctx: object) -> float:
    """Maximum drawdown of the strategy equity curve (placeholder until E7-T03)."""
    del ctx
    return _pending("max_drawdown")
