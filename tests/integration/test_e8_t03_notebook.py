"""E8-T03 AC3: `examples/tutorial.ipynb` executes top to bottom, offline, in under five minutes.

Run by the nox `examples` session (`-m slow`).
"""

from __future__ import annotations

import time
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]


def test_e8_t03_tutorial_notebook_executes_offline_in_under_five_minutes() -> None:
    """Every cell runs without error and produces output; wall time < 300 s."""
    notebook = nbformat.read(ROOT / "examples" / "tutorial.ipynb", as_version=4)
    client = NotebookClient(
        notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    started = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - started
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.get("outputs") is not None for c in code_cells)
    assert any("loglik" in "".join(o.get("text", "") for o in c.outputs) for c in code_cells)
    assert elapsed < 300.0, f"notebook took {elapsed:.0f} s"
