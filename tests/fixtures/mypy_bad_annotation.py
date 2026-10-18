"""Fixture with a deliberately wrong annotation; used to prove `mypy --strict` catches it."""


def add_one(x: int) -> int:
    """Return x + 1, but claim x is a str to trigger a type error at the call site below."""
    return x + 1


bad: str = add_one(3)
