"""E0-T04: the twelve baseline ADRs exist, each with all four required sections."""

import re
from pathlib import Path

import pytest

ADR_DIR = Path(__file__).resolve().parents[2] / "docs" / "adr"

REQUIRED_ADR_NUMBERS = {
    "0001",
    "0002",
    "0003",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
    "0012",
    "0013",
}
REQUIRED_SECTIONS = ("## Context", "## Decision", "## Consequences", "## Status")


def _adr_path_for(number: str) -> Path:
    matches = list(ADR_DIR.glob(f"{number}-*.md"))
    assert len(matches) == 1, f"expected exactly one ADR file for {number}, found {matches}"
    return matches[0]


@pytest.mark.parametrize("number", sorted(REQUIRED_ADR_NUMBERS))
def test_e0_t04_adr_exists_with_required_sections(number: str) -> None:
    """Each required ADR file exists with Context/Decision/Consequences/Status sections."""
    path = _adr_path_for(number)
    text = path.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"{path.name} missing section {section!r}"


@pytest.mark.parametrize("number", sorted(REQUIRED_ADR_NUMBERS))
def test_e0_t04_adr_status_is_accepted(number: str) -> None:
    """Each required ADR's `## Status` section says "accepted"."""
    path = _adr_path_for(number)
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## Status\s*\n+(\w+)", text)
    assert match is not None
    expected = "superseded" if number == "0012" else "accepted"  # ADR-0012 is superseded by ADR-0014
    assert match.group(1).strip().lower() == expected


@pytest.mark.parametrize("number", sorted(REQUIRED_ADR_NUMBERS))
def test_e0_t04_adr_consequences_non_empty(number: str) -> None:
    """Each required ADR's `## Consequences` section has real (non-empty) content."""
    path = _adr_path_for(number)
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## Consequences\s*\n+(.*?)\n##", text, re.DOTALL)
    assert match is not None
    assert len(match.group(1).strip()) > 20


def test_e0_t04_adr_0011_is_the_safetensors_adr_written_by_e8_t01() -> None:
    """E0-T04 leaves ADR-0011 to E8-T01 (it was absent before); now exactly one, about safetensors."""
    found = list(ADR_DIR.glob("0011-*.md"))
    assert len(found) == 1
    assert "safetensors" in found[0].read_text(encoding="utf-8").splitlines()[0].lower()
