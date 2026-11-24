"""Component registries (E10-T02a, FR-G05, D15, D24): one typed place for every named component.

Every named component of the framework (emission and duration families, metrics, initialization
strategies, covariate forecasters, and from later tickets sources, clocks, transforms and presets) is
registered here under a *kind* and a *name*. Extension packages contribute components by registering them;
they never patch core objects (D24).

Registration happens at import time of the module that defines the component (usually through the
`register` decorator). `Registry.get`, `names` and `entries` only read, so no locks are needed. Entry-point
discovery is added by E16-T04 (`Registry._discover` is the hook).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from regimekit.utils.errors import ConfigError

Kind = Literal[
    "emission",
    "duration",
    "metric",
    "source",
    "clock",
    "transform",
    "init",
    "covariate_forecaster",
    "preset",
]
KINDS: tuple[str, ...] = (
    "emission",
    "duration",
    "metric",
    "source",
    "clock",
    "transform",
    "init",
    "covariate_forecaster",
    "preset",
)

T = TypeVar("T")

CORE_PROVENANCE = "regimekit.core"
FINANCE_PROVENANCE = "regimekit.finance"


@dataclass(frozen=True)
class Entry:
    """One registered component.

    Attributes:
        kind: The registry kind.
        name: The name (the config's own spelling).
        obj: The registered object (a class, a function, a factory, ...).
        provenance: Where it came from, e.g. `"regimekit.core"`, `"regimekit.finance"`.
    """

    kind: Kind
    name: str
    obj: Any
    provenance: str


def provenance_of(module: str) -> str:
    """Provenance label of a module name: core, finance, or the top-level package of an extension."""
    if module == "regimekit.finance" or module.startswith("regimekit.finance."):
        return FINANCE_PROVENANCE
    if module == "regimekit" or module.startswith("regimekit."):
        return CORE_PROVENANCE
    return module.partition(".")[0] or "unknown"


def _caller_provenance(depth: int) -> str:
    module = str(sys._getframe(depth).f_globals.get("__name__", ""))
    return provenance_of(module)


class Registry:
    """Typed registries, one per `Kind` (see the module docstring)."""

    def __init__(self) -> None:
        """Create empty registries for every kind."""
        self._entries: dict[str, dict[str, Entry]] = {kind: {} for kind in KINDS}

    def _kind(self, kind: str) -> dict[str, Entry]:
        try:
            return self._entries[kind]
        except KeyError:
            raise ConfigError(f"unknown registry kind {kind!r}; kinds: {', '.join(KINDS)}") from None

    def register(self, kind: Kind, name: str, obj: object, *, provenance: str, replace: bool = False) -> None:
        """Register `obj` as `name` of `kind`.

        Args:
            kind: Registry kind.
            name: Name under which the component is looked up.
            obj: The component.
            provenance: Origin label (`"regimekit.core"`, `"regimekit.finance"`, an extension package).
            replace: Overwrite an existing entry (used by the metric facade's `replace=True`).

        Raises:
            ConfigError: the name is taken and `replace` is false; the message names both provenances.
        """
        entries = self._kind(kind)
        old = entries.get(name)
        if old is not None and not replace:
            raise ConfigError(
                f"{kind} '{name}' is already registered by {old.provenance}; "
                f"cannot register it again from {provenance}"
            )
        entries[name] = Entry(kind, name, obj, provenance)

    def get(self, kind: Kind, name: str) -> Any:  # noqa: ANN401  (kinds hold different object types)
        """The component registered as `name`.

        Raises:
            ConfigError: unknown name; the message lists every registered name with its provenance.
        """
        entries = self._kind(kind)
        entry = entries.get(name)
        if entry is None:
            listing = ", ".join(f"{e.name} ({e.provenance})" for e in self.entries(kind)) or "<none>"
            raise ConfigError(f"unknown {kind} '{name}'; registered: {listing}")
        return entry.obj

    def names(self, kind: Kind) -> tuple[str, ...]:
        """Sorted registered names of `kind`."""
        return tuple(sorted(self._kind(kind)))

    def entries(self, kind: Kind) -> tuple[Entry, ...]:
        """Entries of `kind`, sorted by name."""
        entries = self._kind(kind)
        return tuple(entries[n] for n in sorted(entries))

    def has(self, kind: Kind, name: str) -> bool:
        """Whether `name` is registered for `kind`."""
        return name in self._kind(kind)

    def _discover(self) -> None:
        """Hook for entry-point discovery (E16-T04); a no-op until then."""


REGISTRY = Registry()


def register(kind: Kind, name: str, *, provenance: str | None = None) -> Callable[[T], T]:
    """Decorator registering the decorated class or function in `REGISTRY`.

    `provenance` defaults to the calling module's origin: `regimekit.finance` for modules under
    `regimekit.finance`, `regimekit.core` for other `regimekit` modules, else the top-level package.
    """
    label = provenance if provenance is not None else _caller_provenance(2)

    def wrap(obj: T) -> T:
        REGISTRY.register(kind, name, obj, provenance=label)
        return obj

    return wrap
