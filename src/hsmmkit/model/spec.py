"""`ModelSpec`: structure only, no data (E1-T05)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx

if TYPE_CHECKING:
    from regimekit.model.duration.base import DurationFamily
    from regimekit.model.emission.base import EmissionFamily
    from regimekit.model.params.spec import ParamSpec
    from regimekit.model.transition import TransitionModel


class ModelSpec(eqx.Module):
    """The full model structure: family instances, state-space shape, and the union of param specs.

    Every field is static (`ModelSpec` carries no array data of its own; all array-valued
    parameters live in `Params`/`u`, managed separately by `ParamManager`).
    """

    n_states: int = eqx.field(static=True)
    n_series: int = eqx.field(static=True)
    emission: EmissionFamily = eqx.field(static=True)
    duration: DurationFamily = eqx.field(static=True)
    transition: TransitionModel = eqx.field(static=True)
    order: int = eqx.field(static=True)
    duration_conditioning: Literal["current", "pair"] = eqx.field(static=True)
    engine: Literal["expanded", "explicit"] = eqx.field(static=True)
    d_max: int = eqx.field(static=True)
    initial: Literal["uniform", "stationary", "free"] = eqx.field(static=True)
    param_specs: tuple[ParamSpec, ...] = eqx.field(static=True)
