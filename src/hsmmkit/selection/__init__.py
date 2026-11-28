"""Model selection: splitters, cross-validation, grid search, parametric bootstrap (E14-T01, T02, T04)."""

from hsmmkit.selection import splitters as _splitters  # noqa: F401  (registers the `splitter` kinds)
from hsmmkit.selection.bootstrap import BootstrapResult, LRTResult, bootstrap_lrt, bootstrap_se
from hsmmkit.selection.cv import CVResult, cross_validate
from hsmmkit.selection.grid import make_cells, recommend, run_grid
from hsmmkit.selection.splitters import SequenceKFold, Split, Splitter, WalkForward

__all__ = [
    "BootstrapResult",
    "CVResult",
    "LRTResult",
    "SequenceKFold",
    "Split",
    "Splitter",
    "WalkForward",
    "bootstrap_lrt",
    "bootstrap_se",
    "cross_validate",
    "make_cells",
    "recommend",
    "run_grid",
]
