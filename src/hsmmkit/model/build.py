"""`ProjectConfig` + `PreparedDataset` -> `ModelSpec` (CLAUDE.md §3.3 unowned deliverable).

No ticket's "Creates" named this module; CLAUDE.md §3.3 assigns it to "the first ticket that needs a
`ModelSpec` from a config, at the latest E6-T02" -- that is this ticket (`decode` needs to rebuild
a `ModelSpec` from a stored configuration; `persistence.store` now does so through `build_model_spec_from`).

`EMISSION_FAMILIES`/`DURATION_FAMILIES` are keyed by the config's own family names (`config/schema.py`'s
`EmissionFamily`/`DurationFamily` Literal types); every family already has a `from_config` classmethod
(the E3-T01 pattern). `model.lattice` (multi-start initialization grids, E3-T06) is not consulted here --
`build_model_spec` only needs the model's *structure*, not `lattice_jitter`'s init values -- so lattice
validation (`model.lattice.validate_lattice`, flagged as unwired in E3-T06's STATUS findings) stays a
follow-up for whoever first builds a lattice-configured model end to end.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import regimekit.model.duration
import regimekit.model.emission  # noqa: F401  (registers the emission families)
from regimekit.model.emission.product import ProductEmission
from regimekit.model.spec import ModelSpec
from regimekit.model.transition import JumpTransition
from regimekit.plugins import REGISTRY, Kind
from regimekit.utils.clock import days_to_steps
from regimekit.utils.errors import ConfigError

if TYPE_CHECKING:
    from regimekit.config.schema import DurationCfg, ProjectConfig
    from regimekit.data.contracts import PreparedDataset
    from regimekit.model.transition import TransitionModel
    from regimekit.utils.clock import Clock


class _FamilyView(Mapping[str, Any]):
    """Read-only view of the registered `kind` families a config can name (those with `from_config`)."""

    def __init__(self, kind: Kind) -> None:
        self._kind = kind

    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        if REGISTRY.has(self._kind, name):
            obj = REGISTRY.get(self._kind, name)
            if hasattr(obj, "from_config"):
                return obj
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return (n for n in REGISTRY.names(self._kind) if n in self)

    def __len__(self) -> int:
        return sum(1 for _ in self)


EMISSION_FAMILIES: Mapping[str, Any] = _FamilyView("emission")
DURATION_FAMILIES: Mapping[str, Any] = _FamilyView("duration")


def _with_mean_sojourn_steps(
    cfg: DurationCfg, clock: Clock, nominal_dt: float, bars_per_day: int
) -> DurationCfg:
    """`init_mean_sojourn_days` -> `init_mean_sojourn_steps` via `days_to_steps` (CLAUDE.md §3.3).

    Every `DurationFamily.from_config` rejects `init_mean_sojourn_days` directly (it has no clock or
    `nominal_dt`); this is the one place that conversion happens, before the family ever sees the config.
    """
    new_params = {}
    for key, param in cfg.params.items():
        days = getattr(param, "init_mean_sojourn_days", None)
        if days is not None:
            steps = [days_to_steps(d, clock, nominal_dt, bars_per_day) for d in days]
            param = param.model_copy(
                update={"init_mean_sojourn_days": None, "init_mean_sojourn_steps": steps}
            )
        new_params[key] = param
    return cfg.model_copy(update={"params": new_params})


@dataclass(frozen=True)
class BuildInputs:
    """What the config alone cannot say about a model: the facts it takes from the dataset.

    `n_series`, the per-entry-point covariate name tuples (`cov_names`), and `clock`/`nominal_dt` for the
    duration families' days->steps conversion. A model bundle stores these so that its `ModelSpec` can be
    rebuilt without data.
    """

    n_series: int
    cov_names: dict[str, tuple[str, ...]]
    clock: Clock
    nominal_dt: float

    @classmethod
    def from_data(cls, data: PreparedDataset) -> BuildInputs:
        """The inputs of a prepared dataset."""
        return cls(int(data.observations.shape[1]), dict(data.cov_names), data.clock, float(data.nominal_dt))


def build_model_spec(cfg: ProjectConfig, data: PreparedDataset) -> ModelSpec:
    """Assemble a `ModelSpec` (structure + `param_specs`) from `cfg.model` and `data`'s covariate names.

    `data` supplies what the config alone cannot (see `BuildInputs`). `cfg.model.d_max` is always a
    concrete int by this point (`config/schema.py`'s `_resolve_d_max` cross-field validator fills in a
    default during `load_config`).

    Raises:
        ConfigError: an unregistered `emission.family`/`duration.family` name.
    """
    return build_model_spec_from(cfg, BuildInputs.from_data(data))


def build_model_spec_from(cfg: ProjectConfig, inputs: BuildInputs) -> ModelSpec:
    """`build_model_spec` from explicit `BuildInputs` (no dataset needed).

    Raises:
        ConfigError: an unregistered `emission.family`/`duration.family` name.
    """
    m = cfg.model
    n = m.n_states
    n_series = inputs.n_series

    emis_cls = EMISSION_FAMILIES.get(m.emission.family)
    if emis_cls is None:
        raise ConfigError(f"unknown emission family {m.emission.family!r}", path="model.emission.family")
    emission_cov_names = inputs.cov_names.get("emission", ())
    emission, emission_specs = emis_cls.from_config(m.emission, n, n_series, emission_cov_names)
    if m.emission.scale_by_dt is not None and m.emission.scale_by_dt != emission.scale_by_dt:
        raise ConfigError(
            f"scale_by_dt={m.emission.scale_by_dt} contradicts the family's own ({emission.scale_by_dt})",
            path="model.emission.scale_by_dt",
        )
    if n_series > 1:  # per-series masks and shared parameters (E9-T01); M = 1 keeps the bare family
        emission = ProductEmission.of(emission, n_series)

    dur_cls = DURATION_FAMILIES.get(m.duration.family)
    if dur_cls is None:
        raise ConfigError(f"unknown duration family {m.duration.family!r}", path="model.duration.family")
    duration_cfg = _with_mean_sojourn_steps(
        m.duration, inputs.clock, inputs.nominal_dt, int(cfg.data.clock_option("bars_per_day", 1))
    )
    if m.duration.family == "nonparametric":
        duration, duration_specs = dur_cls.from_config(
            duration_cfg, n, inputs.cov_names.get("duration", ()), inputs.nominal_dt, d_max=m.d_max
        )
    else:
        duration, duration_specs = dur_cls.from_config(
            duration_cfg, n, inputs.cov_names.get("duration", ()), inputs.nominal_dt
        )

    transition, transition_specs = JumpTransition.from_config(
        m.transition, n, m.transition.order, inputs.cov_names.get("transition", ())
    )
    assert m.d_max is not None  # `_resolve_d_max` (config/schema.py) always fills this in

    return ModelSpec(
        n_states=n,
        n_series=n_series,
        emission=emission,
        duration=duration,
        transition=cast("TransitionModel", transition),
        order=m.transition.order,
        duration_conditioning=m.transition.duration_conditioning,
        engine=m.engine,
        d_max=m.d_max,
        initial=m.initial_state,
        param_specs=tuple(emission_specs) + tuple(duration_specs) + tuple(transition_specs),
    )
