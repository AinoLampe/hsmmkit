"""E0-T05 AC4: JSON logs under `CI=1`, human-readable console logs otherwise."""

import json
import os
import subprocess
import sys

import pytest
import structlog

from regimekit.utils.logging import configure_logging, get_logger

_CODE = (
    "from regimekit.utils.logging import configure_logging, get_logger\n"
    "configure_logging('INFO')\n"
    "get_logger('t').info('hello', x=1)\n"
)


def test_e0_t05_logging_json_when_ci_set() -> None:
    """A log line is valid JSON when `CI=1`."""
    env = dict(os.environ, CI="1")
    result = subprocess.run(
        [sys.executable, "-c", _CODE], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["event"] == "hello"
    assert parsed["x"] == 1


def test_e0_t05_logging_human_readable_without_ci() -> None:
    """Without `CI`, the log line is not valid JSON (it's the console renderer's format)."""
    env = {k: v for k, v in os.environ.items() if k != "CI"}
    result = subprocess.run(
        [sys.executable, "-c", _CODE], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip().splitlines()[-1]
    assert "hello" in line
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


def test_e0_t05_configure_and_get_logger_in_process(capsys: pytest.CaptureFixture[str]) -> None:
    """`configure_logging`/`get_logger` work in-process (exercised in isolation, then reset)."""
    try:
        configure_logging("INFO")
        get_logger("t").info("hello", x=1)
        captured = capsys.readouterr()
        assert "hello" in captured.out or "hello" in captured.err
    finally:
        structlog.reset_defaults()
