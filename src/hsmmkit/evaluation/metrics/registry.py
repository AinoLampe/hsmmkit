"""Metric protocol and name-resolved registry (E7-T01).

Concept refs: §3.1 FR-E01, §5.5 (`Metric`). A metric is a named callable over a common context
(`**ctx`) that declares which context entries it `needs`; a runner (E7-T02b) skips metrics whose needs
it cannot supply. Context keys: `forecast` (`ForecastResult`), `realized (H, M)` realized future
observations for steps `1..H`, `horizon` (int, `<= H`), `posterior` (`PosteriorResult`), `fit`
(`FitResult`), `labels (T,)` ints or `None`, `extras` (`dict[str, pd.DataFrame]`: the frames finalized by the
backtest's origin hooks, or one fold's slice of them), plus the optional
`dt_next` / `z_emission_next` used by `log_score_mixture`.

`aggregate` tells a runner how to combine per-origin values: `"mean"` (default) or `"rms"` (root of the
mean square, for `rmse` -- a per-origin value is `|error|`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from hsmmkit.plugins import CORE_PROVENANCE, REGISTRY
from hsmmkit.utils.errors import ConfigError

Need = str  # "forecast", "posterior", "fit", "labels", or the name of an origin hook's extras frame
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
    better: Literal["lower", "higher"] | None = None

    def __call__(self, **ctx: object) -> MetricValue:
        """Call the wrapped function with the context."""
        return self.fn(**ctx)


_BUILTIN_MODULES = (
    "hsmmkit.evaluation.metrics.predictive",
    "hsmmkit.evaluation.metrics.model",
    "hsmmkit.evaluation.metrics.labels",
    "hsmmkit.evaluation.metrics.segmentation",
    "hsmmkit.evaluation.metrics.clustering",
)


def register(metric: Metric, *, replace: bool = False, provenance: str = CORE_PROVENANCE) -> Metric:
    """Add `metric` under `metric.name` (a facade over `REGISTRY` kind `metric`).

    Raises:
        ConfigError: the name is already taken and `replace` is false.
    """
    _load_builtin()  # so a custom metric can never be shadowed by (or clash with) a lazily loaded builtin
    if REGISTRY.has("metric", metric.name) and not replace:
        raise ConfigError(f"metric {metric.name!r} is already registered", path="evaluation.metrics")
    REGISTRY.register("metric", metric.name, metric, provenance=provenance, replace=True)
    return metric


def metric(
    name: str,
    needs: set[Need] | frozenset[Need],
    aggregate: Literal["mean", "rms"] = "mean",
    *,
    better: Literal["lower", "higher"] | None = None,
    provenance: str = CORE_PROVENANCE,
) -> Callable[[Callable[..., MetricValue]], Callable[..., MetricValue]]:
    """Decorator registering a function of `**ctx` as the metric `name`.

    `better` tells `compare` which direction ranks best; `provenance` is the origin label (an application
    passes its own).
    """

    def wrap(fn: Callable[..., MetricValue]) -> Callable[..., MetricValue]:
        register(FunctionMetric(name, frozenset(needs), fn, aggregate, better), provenance=provenance)
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


def better_of(name: str) -> Literal["lower", "higher"] | None:
    """The ranking direction a metric declares (`None`: not ranked or unknown)."""
    _load_builtin()
    if not REGISTRY.has("metric", name):
        return None
    direction: Literal["lower", "higher"] | None = getattr(REGISTRY.get("metric", name), "better", None)
    return direction


def available() -> list[str]:
    """Sorted names of every registered metric."""
    _load_builtin()
    return list(REGISTRY.names("metric"))


def resolve(names: list[str]) -> list[Metric]:
    """Metrics for `names`, in order.

    Raises:
        ConfigError: an unknown name (`path="evaluation.metrics"`, the message lists the known ones).
    """
    _load_builtin()
    unknown = [n for n in names if not REGISTRY.has("metric", n)]
    if unknown:
        raise ConfigError(
            f"unknown metric(s) {unknown}; known: {list(REGISTRY.names('metric'))}", path="evaluation.metrics"
        )
    return [REGISTRY.get("metric", n) for n in names]
