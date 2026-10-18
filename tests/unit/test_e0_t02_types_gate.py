"""E0-T02 AC2: `nox -s types` (mypy --strict) catches a deliberately wrong annotation."""

import subprocess
import sys


def test_e0_t02_mypy_catches_bad_annotation() -> None:
    """Mypy reports an error on the `tests/fixtures/mypy_bad_annotation.py` fixture."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "tests/fixtures/mypy_bad_annotation.py"],
        capture_output=True,
        text=True,
        cwd=".",
        check=False,
    )
    assert result.returncode != 0
    assert "error" in result.stdout.lower()
