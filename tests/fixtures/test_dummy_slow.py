"""Fixture test used only by `test_e0_t02_markers.py` to prove `-m slow` filtering works.

Never meant to run as part of the real suite; `addopts` deselects it by default.
"""

import pytest


@pytest.mark.slow
def test_dummy_slow() -> None:
    """Trivial slow-marked test; must be deselected unless `-m slow` is passed."""
    assert True
