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

import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

import hsmmkit.model.duration
import hsmmkit.model.emission  # noqa: F401  (registers the emission families)
from hsmmkit.config.gates import check as gates_check
from hsmmkit.model.emission.product import ProductEmission
from hsmmkit.model.spec import ModelSpec
from hsmmkit.model.topology import from_config as topology_from_config
from hsmmkit.model.transition import JumpTransition
from hsmmkit.plugins import REGISTRY, Kind
from hsmmkit.utils.clock import days_to_steps
from hsmmkit.utils.errors import ConfigError

if TYPE_CHECKING:
    from hsmmkit.config.schema import DurationCfg, ProjectConfig
    from hsmmkit.data.contracts import PreparedDataset, Support
    from hsmmkit.model.params.spec import ParamSpec
    from hsmmkit.model.transition import TransitionModel


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


def _fix_absorbing(specs: Sequence[ParamSpec], absorbing: tuple[int, ...], n_states: int) -> list[ParamSpec]:
    """Absorbing regimes never end a sojourn: their duration entries are `fixed` (dropped from `u`)."""
    out: list[ParamSpec] = []
    for sp in specs:
        if sp.shape[:1] == (n_states,) and sp.init is not None and sp.constraint != "increasing":
            modes = [sp.mode] * n_states if isinstance(sp.mode, str) else list(sp.mode)
            for i in absorbing:
                modes[i] = "fixed"
            value = np.array(sp.init, copy=True) if sp.value is None else sp.value
            sp = dataclasses.replace(sp, mode=tuple(modes), value=value)
        out.append(sp)
    return out


def _with_mean_sojourn_steps(
    cfg: DurationCfg, clock: str, nominal_dt: float, clock_options: Mapping[str, Any]
) -> DurationCfg:
    """`init_mean_sojourn_days` -> `init_mean_sojourn_steps` via `days_to_steps` (CLAUDE.md §3.3).

    Every `DurationFamily.from_config` rejects `init_mean_sojourn_days` directly (it has no clock or
    `nominal_dt`); this is the one place that conversion happens, before the family ever sees the config.
    """
    new_params = {}
    for key, param in cfg.params.items():
        days = getattr(param, "init_mean_sojourn_days", None)
        if days is not None:
            steps = [days_to_steps(d, clock, nominal_dt, clock_options) for d in days]
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
    clock: str
    nominal_dt: float

    @classmethod
    def from_data(cls, data: PreparedDataset) -> BuildInputs:
        """The inputs of a prepared dataset."""
        return cls(int(data.observations.shape[1]), dict(data.cov_names), data.clock, float(data.nominal_dt))


def required_own_lags(cfg: ProjectConfig) -> int:
    """Own lags the preparer must generate for `cfg`'s emission (`requires_own_lags`; `0` for most families).

    Pass the result as `Preparer(..., own_lags=...)`.
    """
    emis_cls = EMISSION_FAMILIES.get(cfg.model.emission.family)
    if emis_cls is None or cfg.model.emission.ar is None:
        return 0
    return int(getattr(emis_cls, "lags_required", lambda _cfg: 0)(cfg.model.emission))


def _class_support(name: str, mixture_base: str | None) -> str:
    """The support a registered emission family declares (the dataclass default of its `support` field)."""
    cls = EMISSION_FAMILIES.get(mixture_base if name == "mixture" and mixture_base else name)
    field = getattr(cls, "__dataclass_fields__", {}).get("support")
    return str(field.default) if field is not None and isinstance(field.default, str) else "real"


def required_series_support(cfg: ProjectConfig) -> tuple[Support, ...] | None:
    """Per-column supports of a `product` emission (`Preparer(..., series_support=...)`), else `None`."""
    blocks = cfg.model.emission.blocks
    if cfg.model.emission.family != "product" or blocks is None:
        return None
    pairs: list[tuple[int, str]] = []
    for b in blocks:
        support = _class_support(b.family, b.mixture.base if b.mixture is not None else None)
        pairs.extend((int(c), support) for c in (b.columns or []) if not isinstance(c, str))
    return tuple(cast("Support", s) for _, s in sorted(pairs))


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
    gates_check(cfg)
    m = cfg.model
    n = m.n_states
    n_series = inputs.n_series

    if (m.emission.family == "mixture") != (m.emission.mixture is not None):
        raise ConfigError(
            "model.emission.mixture goes with family 'mixture' (and only with it)",
            path="model.emission.mixture",
        )
    if (m.emission.family == "product") != (m.emission.blocks is not None):
        raise ConfigError(
            "model.emission.blocks goes with family 'product' (and only with it)",
            path="model.emission.blocks",
        )
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
    ito = getattr(emission, "ito_correction", None)
    if m.emission.ito_correction is not None and ito is not None and m.emission.ito_correction != ito:
        raise ConfigError(
            f"ito_correction={m.emission.ito_correction} contradicts the family's own ({ito})",
            path="model.emission.ito_correction",
        )
    memory_mode: Literal["segments", "steps"] = "segments"
    if getattr(emission, "memory_steps", 0):
        memory_mode = "steps"
        if m.engine != "expanded" or m.transition.mode != "jump":
            raise ConfigError(
                "a mean-adjusted MS-AR runs on the expanded engine with jump transitions",
                path="model.engine",
            )
        if m.duration.family != "geometric":
            raise ConfigError(
                "memory_mode 'steps' (mean-adjusted MS-AR) requires geometric durations",
                path="model.duration.family",
            )
    if n_series > 1 and not getattr(
        emission, "joint", False
    ):  # per-series masks and shared parameters (E9-T01); M = 1 keeps the bare family
        emission = ProductEmission.of(emission, n_series)

    if m.duration.family is None:  # `per_state` (gated until E12-T05)
        raise ConfigError("a duration family per state is not supported yet", path="model.duration.per_state")
    dur_cls = DURATION_FAMILIES.get(m.duration.family)
    if dur_cls is None:
        raise ConfigError(f"unknown duration family {m.duration.family!r}", path="model.duration.family")
    duration_cfg = _with_mean_sojourn_steps(
        m.duration, inputs.clock, inputs.nominal_dt, cfg.data.clock_options
    )
    if m.duration.family == "nonparametric":
        duration, duration_specs = dur_cls.from_config(
            duration_cfg, n, inputs.cov_names.get("duration", ()), inputs.nominal_dt, d_max=m.d_max
        )
    else:
        duration, duration_specs = dur_cls.from_config(
            duration_cfg, n, inputs.cov_names.get("duration", ()), inputs.nominal_dt
        )

    topology = topology_from_config(m.topology, n)
    transition, transition_specs = JumpTransition.from_config(
        m.transition,
        n,
        m.transition.order,
        inputs.cov_names.get("transition", ()),
        None if topology.is_full else topology,
    )
    if not topology.is_full and topology.absorbing:
        duration_specs = _fix_absorbing(duration_specs, topology.absorbing, n)
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
        memory_mode=memory_mode,
    )
