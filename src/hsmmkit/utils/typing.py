"""Array-shape typing helpers (overview §4.6)."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable

from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped

__all__ = ["Array", "Bool", "Float", "Int", "typechecked"]


def typechecked[F: Callable[..., object]](func: F) -> F:
    """Enforce jaxtyping shape/dtype annotations via beartype, gated by `REGIMEKIT_TYPECHECK`.

    Checking happens per call (not once at decoration time) so tests and callers can toggle
    `REGIMEKIT_TYPECHECK` at runtime; when unset or not `"1"`, `func` runs unchanged.
    """
    checked = jaxtyped(typechecker=beartype)(func)

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        if os.environ.get("REGIMEKIT_TYPECHECK") == "1":
            return checked(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
