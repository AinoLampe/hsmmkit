"""Array-shape typing helpers (overview §4.6)."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable

from jaxtyping import Array, Bool, Float, Int, jaxtyped

__all__ = ["Array", "Bool", "Float", "Int", "typechecked"]


def typechecked[F: Callable[..., object]](func: F) -> F:
    """Enforce jaxtyping shape/dtype annotations via beartype, gated by `HSMMKIT_TYPECHECK`.

    Checking happens per call (not once at decoration time) so tests and callers can toggle
    `HSMMKIT_TYPECHECK` at runtime; when unset or not `"1"`, `func` runs unchanged.
    """
    checked: list[Callable[..., object]] = []

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        if os.environ.get("HSMMKIT_TYPECHECK") == "1":
            if not checked:  # beartype is a dev dependency: import it only when checking is switched on
                from beartype import beartype

                checked.append(jaxtyped(typechecker=beartype)(func))
            return checked[0](*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
