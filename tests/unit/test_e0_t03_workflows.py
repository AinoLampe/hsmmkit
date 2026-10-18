"""E0-T03: validate the GitHub Actions workflow YAML (structure only; cannot execute CI locally)."""

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_YAML = YAML(typ="safe")


def _load(name: str) -> dict[str, Any]:
    return _YAML.load(WORKFLOWS_DIR / name)  # type: ignore[no-any-return]


def test_e0_t03_pr_workflow_has_lint_test_docs_jobs() -> None:
    """`pr.yml` runs lint, test and docs jobs on PR and push to main."""
    doc = _load("pr.yml")
    assert {"lint", "test", "docs"} <= doc["jobs"].keys()
    assert "pull_request" in doc["on"]
    assert "main" in doc["on"]["push"]["branches"]


def test_e0_t03_pr_workflow_uploads_coverage_artifact() -> None:
    """The `test` job uploads `coverage.json` as an artifact."""
    doc = _load("pr.yml")
    steps = doc["jobs"]["test"]["steps"]
    upload_steps = [s for s in steps if s.get("uses", "").startswith("actions/upload-artifact")]
    assert upload_steps
    assert "coverage.json" in upload_steps[0]["with"]["path"]
    assert "htmlcov/" in upload_steps[0]["with"]["path"]


def test_e0_t03_nightly_workflow_scheduled_and_dispatchable() -> None:
    """`nightly.yml` has a cron schedule and can be triggered manually, with recovery + benchmark jobs."""
    doc = _load("nightly.yml")
    assert "schedule" in doc["on"]
    assert doc["on"]["schedule"][0]["cron"]
    assert "workflow_dispatch" in doc["on"]
    assert {"recovery", "benchmark"} <= doc["jobs"].keys()


def test_e0_t03_nightly_benchmark_job_uploads_bench_json() -> None:
    """The `benchmark` job uploads `bench.json` as an artifact."""
    doc = _load("nightly.yml")
    steps = doc["jobs"]["benchmark"]["steps"]
    upload_steps = [s for s in steps if s.get("uses", "").startswith("actions/upload-artifact")]
    assert upload_steps
    assert "bench.json" in upload_steps[0]["with"]["path"]


def test_e0_t03_gpu_workflow_is_manual_dispatch_only() -> None:
    """`gpu.yml` only triggers on `workflow_dispatch`, targeting a self-hosted GPU runner."""
    doc = _load("gpu.yml")
    assert set(doc["on"].keys()) == {"workflow_dispatch"}
    runs_on = doc["jobs"]["gpu-tests"]["runs-on"]
    assert "self-hosted" in runs_on
    assert "gpu" in runs_on


def test_e0_t03_nightly_recovery_job_runs_only_the_smoke_session() -> None:
    """The nightly `recovery` job runs the quick T=5000 smoke session within a 60-minute timeout."""
    job = _load("nightly.yml")["jobs"]["recovery"]
    assert job["timeout-minutes"] <= 60
    assert any(step.get("run") == "uv run nox -s recovery_smoke" for step in job["steps"])


def test_e0_t03_weekly_workflow_runs_the_full_recovery_suite() -> None:
    """`weekly.yml` is scheduled, dispatchable and runs `nox -s recovery` (six scenarios at T=20000)."""
    doc = _load("weekly.yml")
    assert doc["on"]["schedule"][0]["cron"]
    assert "workflow_dispatch" in doc["on"]
    steps = doc["jobs"]["recovery-full"]["steps"]
    assert any(step.get("run") == "uv run nox -s recovery" for step in steps)
    assert doc["jobs"]["recovery-full"]["timeout-minutes"] >= 360
    uploads = [s for s in steps if s.get("uses", "").startswith("actions/upload-artifact")]
    assert uploads
    assert "runs/recovery" in uploads[0]["with"]["path"]


def test_e0_t03_noxfile_has_full_and_smoke_recovery_sessions() -> None:
    """Both recovery sessions exist; the smoke one is restricted to the smoke test file."""
    text = (WORKFLOWS_DIR.parents[1] / "noxfile.py").read_text(encoding="utf-8")
    assert "def recovery(" in text
    assert "def recovery_smoke(" in text
    assert "test_e5_t05a_recovery_smoke.py" in text
