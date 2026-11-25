"""Preset registry (E10-T10): validated YAML configurations with a synthetic fixture generator.

A preset is a registered object (kind `preset`) with a name, a description, the YAML text and an optional
`fixture(seed)` that returns the synthetic data files the YAML refers to (relative path -> DataFrame).
Applications register their own presets through `register("preset", name)`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pandas as pd

from hsmmkit.plugins import REGISTRY, register
from hsmmkit.presets import fixtures
from hsmmkit.utils.errors import ConfigError

__all__ = ["Preset", "get_preset", "list_presets", "register_preset", "write_preset"]


@dataclass(frozen=True)
class Preset:
    """A named configuration with its synthetic data generator."""

    name: str
    description: str
    yaml: str
    fixture: Callable[[int], dict[str, pd.DataFrame]] | None = None


def register_preset(preset: Preset) -> Preset:
    """Register `preset` under `preset.name` (kind `preset`) and return it."""
    register("preset", preset.name)(preset)
    return preset


def _core_yaml(name: str) -> str:
    return resources.files("hsmmkit.presets").joinpath("core", f"{name}.yaml").read_text(encoding="utf-8")


def _register_core() -> None:
    core = (
        (
            "segmentation_gaussian_ltr",
            "Left-to-right 4-regime segmentation of 50 labelled 3-dim sessions (Part II Appendix H).",
            fixtures.segmentation_gaussian_ltr,
        ),
        (
            "two_regime_gaussian",
            "Two-regime Gaussian sanity example on a daily calendar grid (Part I Appendix E, Gaussian form).",
            fixtures.two_regime_gaussian,
        ),
    )
    for short, description, fixture in core:
        name = f"core/{short}"
        if name not in REGISTRY.names("preset"):
            register_preset(Preset(name, description, _core_yaml(short), fixture))


_register_core()


def list_presets() -> list[Preset]:
    """All registered presets, sorted by name."""
    return sorted(
        (e.obj for e in REGISTRY.entries("preset") if isinstance(e.obj, Preset)), key=lambda p: p.name
    )


def get_preset(name: str) -> Preset:
    """The preset `name`.

    Raises:
        ConfigError: unknown name; the message lists the registered presets.
    """
    for preset in list_presets():
        if preset.name == name:
            return preset
    names = ", ".join(p.name for p in list_presets()) or "<none>"
    raise ConfigError(f"unknown preset '{name}'; registered: {names}", path="preset")


def write_preset(name: str, directory: str | Path, *, seed: int = 0) -> list[Path]:
    """Write the YAML of preset `name` and its synthetic data files into `directory`; return the paths.

    The YAML is `<directory>/<short name>.yaml`; the data files go where the YAML's relative paths point,
    below `directory`.
    """
    preset = get_preset(name)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written = [out / f"{name.rpartition('/')[2]}.yaml"]
    written[0].write_text(preset.yaml, encoding="utf-8")
    if preset.fixture is not None:
        for rel, frame in preset.fixture(seed).items():
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix == ".parquet":
                frame.to_parquet(target, index=False)
            else:
                frame.to_csv(target, index=False)
            written.append(target)
    return written
