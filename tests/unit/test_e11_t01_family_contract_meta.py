"""E11-T01: meta-tests of the family contract suite on good and broken fixtures."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hsmmkit.testing import family_contract
from hsmmkit.testing.family_contract import CHECKS, FamilyCase, check_family
from tests.helpers.toy_family import toy_case

GOOD = toy_case()


@pytest.mark.parametrize("check", sorted(CHECKS))
def test_e11_t01_good_family_passes_every_applicable_check(check: str) -> None:
    """The toy count family (with `moments`, `sufficient_stats` and `m_step`) passes all checks or skips."""
    try:
        check_family(GOOD, check)
    except pytest.skip.Exception:
        assert check in {"dt_scaling", "covariate_links"} or check == "optional_methods"


def test_e11_t01_support_validation_and_m_step_are_exercised() -> None:
    """The toy family runs check 4 (count support rejects negatives) and check 7 (m_step stationary)."""
    CHECKS["support_validation"](GOOD)
    CHECKS["optional_methods"](GOOD)


def test_e11_t01_wrong_normalizing_constant_fails_check_1_readably() -> None:
    """A density off by a constant fails `log_prob` against the reference, naming the constant."""
    broken = toy_case("poisson-broken", offset=0.1)
    with pytest.raises(AssertionError, match=r"poisson-broken: log_prob differs .*normalizing constant"):
        CHECKS["log_prob"](broken)


def test_e11_t01_wrong_moments_fail_check_2_without_a_reference() -> None:
    """With no scipy reference, wrong `moments` are caught by the sampling check."""
    base = toy_case("poisson-moments", bad_moments=True)
    case = FamilyCase(
        name=base.name,
        factory=base.factory,
        theta_factory=base.theta_factory,
        data_factory=base.data_factory,
    )
    with pytest.raises(AssertionError, match=r"poisson-moments: .*variance"):
        CHECKS["sample"](case)


def test_e11_t01_register_case_rejects_duplicates() -> None:
    """Case names are unique."""
    case = toy_case("poisson-dup")
    family_contract.register_case(case)
    try:
        with pytest.raises(ValueError, match="already registered"):
            family_contract.register_case(case)
    finally:
        family_contract.CASES.pop("poisson-dup")


def test_e11_t01_suite_is_usable_from_an_external_package(tmp_path: Path) -> None:
    """A temp package registers a case and runs the suite through `pytest --pyargs`."""
    pkg = tmp_path / "ext_family_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "test_contract.py").write_text(
        textwrap.dedent(
            """
            import jax
            import jax.numpy as jnp

            from hsmmkit.model.emission.gaussian import GaussianFamily
            from hsmmkit.testing import family_contract
            from hsmmkit.testing.family_contract import (  # noqa: F401
                FamilyCase, pytest_generate_tests, test_family_contract,
            )


            def theta(key, n, m, k):
                return {"emission.mean": jnp.zeros((n, m)), "emission.scale": jnp.full((n, m), 0.2)}


            def data(key, t, m):
                return 0.1 * jax.random.normal(key, (t, m)), jnp.full((t,), 0.1), None


            family_contract.register_case(FamilyCase("ext-gaussian", GaussianFamily, theta, data))
            """
        )
    )
    env = {
        "PYTHONPATH": f"{tmp_path}:{Path(__file__).resolve().parents[2]}",
        "JAX_PLATFORMS": "cpu",
        "PATH": "",
    }
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--pyargs",
            "ext_family_pkg",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-rA",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-1000:]
    assert "test_family_contract[ext-gaussian-log_prob]" in res.stdout


def test_e11_t01_only_appendix_tolerances_are_used() -> None:
    """The suite's numeric tolerances are the documented constants."""
    assert family_contract.LOGPDF_ATOL == 1e-10
    assert family_contract.ROUNDTRIP_ATOL == 1e-12
    assert family_contract.MOMENT_SE == 3.0
