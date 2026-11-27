"""Model selection: splitters, cross-validation (E14-T01) and grid search (E14-T02)."""

from hsmmkit.selection import splitters as _splitters  # noqa: F401  (registers the `splitter` kinds)
from hsmmkit.selection.cv import CVResult, cross_validate
from hsmmkit.selection.grid import make_cells, recommend, run_grid
from hsmmkit.selection.splitters import SequenceKFold, Split, Splitter, WalkForward

__all__ = [
    "CVResult",
    "SequenceKFold",
    "Split",
    "Splitter",
    "WalkForward",
    "cross_validate",
    "make_cells",
    "recommend",
    "run_grid",
]
