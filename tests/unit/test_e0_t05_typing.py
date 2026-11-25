"""E0-T05 AC5: `typechecked` raises on a wrong-shape call only when `HSMMKIT_TYPECHECK=1`."""

import os

import jax.numpy as jnp
import pytest
from jaxtyping import Float, TypeCheckError

from hsmmkit.utils.typing import Array, typechecked


@typechecked
def sum_vector(x: Float[Array, " n"]) -> Float[Array, ""]:  # noqa: F722
    """Sum a 1-D vector (used only to exercise the shape check)."""
    return jnp.sum(x)


def test_e0_t05_typechecked_raises_on_wrong_shape_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2-D array passed where a 1-D vector is annotated raises when `HSMMKIT_TYPECHECK=1`."""
    monkeypatch.setenv("HSMMKIT_TYPECHECK", "1")
    with pytest.raises(TypeCheckError):
        sum_vector(jnp.zeros((2, 2)))


def test_e0_t05_typechecked_permissive_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same wrong-shape call does not raise when `HSMMKIT_TYPECHECK` is unset."""
    monkeypatch.delenv("HSMMKIT_TYPECHECK", raising=False)
    assert os.environ.get("HSMMKIT_TYPECHECK") != "1"
    result = sum_vector(jnp.zeros((2, 2)))
    assert result.shape == ()


def test_e0_t05_typechecked_accepts_correct_shape_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A correctly-shaped call succeeds regardless of `HSMMKIT_TYPECHECK`."""
    monkeypatch.setenv("HSMMKIT_TYPECHECK", "1")
    result = sum_vector(jnp.ones(3))
    assert float(result) == 3.0
