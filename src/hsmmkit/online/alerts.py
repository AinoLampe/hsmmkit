"""Alert rules with hysteresis and structured events (E15-T03).

Concept refs: Part II §14.11 (alerts), §15.4 (`online.alerts`). Rules are evaluated in plain Python on the
`(N,)` vectors a streaming step returns (negligible cost); the active flags live in `OnlineState.alerts`.

* probability rule `{regime, p_on, p_off}`: `on` when `P(regime) >= p_on` while inactive, `off` when
  `P(regime) < p_off` while active (`p_off < p_on`: a path hovering around `p_on` does not flap);
* `use: smoothed` (probability rules) reads the fixed-lag marginal of step `t - lag`; duration rules always use
  the filtered probability of the current step;
* remaining-duration rule `{regime, remaining_duration_below: d, p_min}`: `on` when `P(regime) >= p_min` and
  the expected remaining duration of the current segment (clock units) is `< d`; `off` when it rises above
  `HYSTERESIS * d` or the regime's probability falls below `p_min`.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from hsmmkit.utils.errors import ConfigError

HYSTERESIS = 1.2
"""A remaining-duration alert switches off when the remaining duration exceeds `HYSTERESIS * d`."""


@dataclass(frozen=True)
class AlertRule:
    """One resolved alert rule (`online.alerts[i]`)."""

    name: str
    regime: int
    p_on: float | None = None
    p_off: float | None = None
    remaining_below: float | None = None
    p_min: float = 0.5
    use: Literal["filtered", "smoothed"] = "filtered"

    @property
    def kind(self) -> Literal["probability", "duration"]:
        """The rule family."""
        return "duration" if self.remaining_below is not None else "probability"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | Any, index: int) -> AlertRule:  # noqa: ANN401
        """From an `AlertCfg` (or its dict); the default name is `alert<index>`."""
        get = cfg.get if isinstance(cfg, Mapping) else lambda k, d=None: getattr(cfg, k, d)
        return cls(
            name=get("name") or f"alert{index}",
            regime=int(get("regime") or 0),
            p_on=get("p_on"),
            p_off=get("p_off"),
            remaining_below=get("remaining_duration_below"),
            p_min=float(get("p_min", 0.5) or 0.5),
            use=get("use", "filtered") or "filtered",
        )


@dataclass(frozen=True)
class AlertEvent:
    """A regime alert turning `on` or `off` at step `t`.

    `value` is the quantity that crossed `threshold` (a probability, or a remaining duration in clock units).
    """

    time: Any
    t: int
    rule: str
    kind: Literal["on", "off"]
    regime: int
    value: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form (`time` as ISO text or a number)."""
        time = self.time
        if isinstance(time, np.datetime64):
            time = str(np.datetime_as_string(time, unit="ns"))
        elif isinstance(time, np.integer):
            time = int(time)
        elif isinstance(time, np.floating):
            time = float(time)
        return {
            "time": time,
            "t": int(self.t),
            "rule": self.rule,
            "kind": self.kind,
            "regime": int(self.regime),
            "value": float(self.value),
            "threshold": float(self.threshold),
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> AlertEvent:
        """Inverse of `to_dict` (a string `time` that parses as a date comes back as `datetime64[ns]`)."""
        time = blob["time"]
        if isinstance(time, str):
            with contextlib.suppress(ValueError):
                time = np.datetime64(time, "ns")
        return cls(
            time,
            int(blob["t"]),
            blob["rule"],
            blob["kind"],
            int(blob["regime"]),
            float(blob["value"]),
            float(blob["threshold"]),
        )


def rules_from_config(alerts: Sequence[Mapping[str, Any] | Any]) -> tuple[AlertRule, ...]:
    """Resolve and validate the rules (`regime` is checked against `n_states` by the filter)."""
    rules = tuple(AlertRule.from_config(a, i) for i, a in enumerate(alerts))
    names = [r.name for r in rules]
    if len(set(names)) != len(names):
        raise ConfigError("alert names must be unique", path="online.alerts.name")
    return rules


def evaluate(
    rules: Sequence[AlertRule],
    active: np.ndarray,
    t: int,
    time: Any,  # noqa: ANN401
    filtered: np.ndarray,
    smoothed: np.ndarray | None,
    remaining: np.ndarray | None,
) -> tuple[np.ndarray, list[AlertEvent]]:
    """Advance the active flags by one step.

    Args:
        rules: the resolved rules.
        active: `(n_rules,)` bool flags before the step.
        t: the step index (number of observations processed, `1`-based after this step).
        time: the step's timestamp (or `None`).
        filtered: `(N,)` filtered regime probabilities.
        smoothed: `(N,)` fixed-lag smoothed marginals (`NaN` while the buffer fills) or `None`.
        remaining: `(N,)` expected remaining duration per regime in clock units, or `None`.

    Returns:
        `(active', events)`; events are ordered by rule.
    """
    out = np.array(active, dtype=bool, copy=True)
    events: list[AlertEvent] = []
    for i, rule in enumerate(rules):
        # `use: smoothed` applies to probability rules; a duration rule is about the current step
        use_smoothed = rule.use == "smoothed" and smoothed is not None and rule.kind == "probability"
        source = smoothed if use_smoothed else filtered
        p = float(source[rule.regime])
        if math.isnan(p):
            continue  # the smoothed source is still filling
        if rule.kind == "probability":
            assert rule.p_on is not None
            assert rule.p_off is not None
            if not out[i] and p >= rule.p_on:
                out[i] = True
                events.append(AlertEvent(time, t, rule.name, "on", rule.regime, p, rule.p_on))
            elif out[i] and p < rule.p_off:
                out[i] = False
                events.append(AlertEvent(time, t, rule.name, "off", rule.regime, p, rule.p_off))
            continue
        assert rule.remaining_below is not None
        rem = float("nan") if remaining is None else float(remaining[rule.regime])
        if not out[i] and p >= rule.p_min and rem < rule.remaining_below:
            out[i] = True
            events.append(AlertEvent(time, t, rule.name, "on", rule.regime, rem, rule.remaining_below))
        elif out[i]:
            if p < rule.p_min:
                out[i] = False
                events.append(AlertEvent(time, t, rule.name, "off", rule.regime, p, rule.p_min))
            elif rem > HYSTERESIS * rule.remaining_below:
                out[i] = False
                events.append(
                    AlertEvent(time, t, rule.name, "off", rule.regime, rem, HYSTERESIS * rule.remaining_below)
                )
    return out, events
