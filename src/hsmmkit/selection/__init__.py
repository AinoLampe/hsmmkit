"""Model selection: splitters and cross-validation (E14-T01)."""

from hsmmkit.selection import splitters as _splitters  # noqa: F401  (registers the `splitter` kinds)
from hsmmkit.selection.cv import CVResult, cross_validate
from hsmmkit.selection.splitters import SequenceKFold, Split, Splitter, WalkForward

__all__ = ["CVResult", "SequenceKFold", "Split", "Splitter", "WalkForward", "cross_validate"]
