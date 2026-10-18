"""Metric protocol and name-resolved registry (E7-T01).

Concept refs: §3.1 FR-E01, §5.5 (`Metric`). A metric is a named callable over a common context
(`**ctx`) that declares which context entries it `needs`; a runner (E7-T02b) skips metrics whose needs
it cannot supply. Context keys: `forecast` (`ForecastResult`), `realized (H, M)` realized future
log-returns for steps `1..H`, `horizon` (int, `<= H`), `posterior` (`PosteriorResult`), `fit`
(`FitResult`), `labels (T,)` ints or `None`, `strategy` (`pd.DataFrame` or `None`), plus the optional
`dt_next` / `z_emission_next` used by `log_score_mixture`.

`aggregate` tells a runner how to combine per-origin values: `"mean"` (default) or `"rms"` (root of the
mean square, for `rmse` -- a per-origin value is `|error|`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from regimekit.utils.errors import ConfigError

Need = Literal["forecast", "posterior", "fit", "labels", "strategy"]
MetricValue = float | dict[str, float]


class Metric(Protocol):
    """Structural type of a registered metric."""

    @property
    def name(self) -> str:
        """Registry key."""
        ...

    @property
    def needs(self) -> frozenset[str]:
        """Context entries the metric reads (a runner skips it when one is missing)."""
        ...

    @property
    def aggregate(self) -> str:
        """How a runner combines per-origin values: `"mean"` or `"rms"`."""
        ...

    def __call__(self, **ctx: object) -> MetricValue:
        """Evaluate on the context; a dict result holds named components."""
        ...


@dataclass(frozen=True)
class FunctionMetric:
    """A metric backed by a plain function of the context."""

    name: str
    needs: frozenset[str]
    fn: Callable[..., MetricValue]
    aggregate: str = "mean"

    def __call__(self, **ctx: object) -> MetricValue:
        """Call the wrapped function with the context."""
        return self.fn(**ctx)


_REGISTRY: dict[str, Metric] = {}
_BUILTIN_MODULES = (
    "regimekit.evaluation.metrics.predictive",
    "regimekit.evaluation.metrics.model",
    "regimekit.evaluation.metrics.labels",
    "regimekit.evaluation.metrics.strategy",
)


def register(metric: Metric, *, replace: bool = False) -> Metric:
    """Add `metric` under `metric.name`.

    Raises:
        ConfigError: the name is already taken and `replace` is false.
    """
    _load_builtin()  # so a custom metric can never be shadowed by (or clash with) a lazily loaded builtin
    if metric.name in _REGISTRY and not replace:
        raise ConfigError(f"metric {metric.name!r} is already registered", path="evaluation.metrics")
    _REGISTRY[metric.name] = metric
    return metric


def metric(
    name: str, needs: set[Need] | frozenset[Need], aggregate: Literal["mean", "rms"] = "mean"
) -> Callable[[Callable[..., MetricValue]], Callable[..., MetricValue]]:
    """Decorator registering a function of `**ctx` as the metric `name`."""

    def wrap(fn: Callable[..., MetricValue]) -> Callable[..., MetricValue]:
        register(FunctionMetric(name, frozenset(needs), fn, aggregate))
        return fn

    return wrap


_builtin_loaded = False


def _load_builtin() -> None:
    """Import the builtin metric modules once (they register themselves on import)."""
    global _builtin_loaded
    if _builtin_loaded:
        return
    _builtin_loaded = True  # set first: the modules call `register`, which calls this function
    try:
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)
    except Exception:
        _builtin_loaded = False
        raise


def available() -> list[str]:
    """Sorted names of every registered metric."""
    _load_builtin()
    return sorted(_REGISTRY)


def resolve(names: list[str]) -> list[Metric]:
    """Metrics for `names`, in order.

    Raises:
        ConfigError: an unknown name (`path="evaluation.metrics"`, the message lists the known ones).
    """
    _load_builtin()
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise ConfigError(
            f"unknown metric(s) {unknown}; known: {sorted(_REGISTRY)}", path="evaluation.metrics"
        )
    return [_REGISTRY[n] for n in names]
