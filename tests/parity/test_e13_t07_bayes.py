"""E13-T07: the sampler rows of the parity table (`mcmc_truth`, `gibbs_nuts`); nightly (`-m slow`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("numpyro")
pytest.importorskip("arviz")

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation.mcmc import MCMCEstimator
from tests.helpers import em_models
from tests.helpers.parity import ParityRow
from tests.integration.test_e13_t06b_gibbs import _BetaGeometric, _NGGaussian

pytestmark = [pytest.mark.bayes, pytest.mark.slow]

_TRUTH = {
    "emission.mean": np.array([[0.0], [1.5]]),
    "emission.scale": np.array([[0.5], [1.2]]),
    "duration.rate": np.full(2, -np.log(1 - 1 / 25)),
}


@pytest.fixture(scope="module")
def fits(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """NUTS and Gibbs on the two-regime fixture (T = 5000, 4 chains x 1000 draws), the same priors."""
    spec, _ = em_models.build(tmp_path_factory.mktemp("bayes"), "gaussian", "geometric")
    data = em_models.dataset("gaussian", 5000, 4)
    y = np.asarray(data.observations)
    base = {f.name: getattr(spec.emission, f.name) for f in dataclasses.fields(spec.emission) if f.init}
    dur = {f.name: getattr(spec.duration, f.name) for f in dataclasses.fields(spec.duration) if f.init}
    ng = dataclasses.replace(
        spec,
        emission=_NGGaussian(**base, m0=float(y.mean()), var0=float(y.var())),  # type: ignore[arg-type]
        duration=_BetaGeometric(**dur),  # type: ignore[arg-type]
    )
    out: dict[str, object] = {}
    for sampler, model in (("gibbs", spec), ("nuts", ng)):
        cfg = EstimationSection.model_validate(
            {
                "method": "mcmc",
                "restarts": 1,
                "init": ["rolling_kmeans"],
                "mcmc": {"sampler": sampler, "default_priors": sampler == "gibbs"},
            }
        )
        out[sampler] = MCMCEstimator().fit(model, data, cfg, 0)
    return out


@pytest.mark.parametrize("sampler", ["nuts", "gibbs"])
def test_e13_t07_samplers_recover_the_truth(
    fits: dict[str, object], sampler: str, parity_log: Callable[[ParityRow], ParityRow]
) -> None:
    """Posterior means within 2 posterior SD of the truth; R-hat <= 1.05; bulk ESS >= 200."""
    res = fits[sampler]
    info = res.diagnostics["mcmc"]  # type: ignore[attr-defined]
    zmax = max(
        float(np.max(np.abs(np.asarray(res.params[k]) - v) / np.asarray(res.std_errors[k])))  # type: ignore[attr-defined]
        for k, v in _TRUTH.items()
    )
    ok = zmax <= 2.0 and info["max_r_hat"] <= 1.05 and info["min_ess_bulk"] >= 200
    row = parity_log(
        ParityRow(
            "gaussian / geometric (Appendix E)",
            f"{sampler} vs truth",
            "T = 5000, 4 chains x 1000 draws",
            "2 posterior SD; R-hat 1.05; ESS 200",
            f"z {zmax:.2f}; R-hat {info['max_r_hat']:.3f}; ESS {info['min_ess_bulk']:.0f}",
            "1/1" if ok else "0/1",
            "ok" if ok else "FAILED",
        )
    )
    assert row.status == "ok", row


def test_e13_t07_gibbs_and_nuts_agree(
    fits: dict[str, object], parity_log: Callable[[ParityRow], ParityRow]
) -> None:
    """Posterior means within 2 Monte-Carlo SE of each other (identical priors)."""
    worst = 0.0
    for name in _TRUTH:
        se = []
        for sampler in ("gibbs", "nuts"):
            res = fits[sampler]
            ess = np.array([v for k, v in res.diagnostics["mcmc"]["ess_bulk"].items() if k.startswith(name)])  # type: ignore[attr-defined]
            se.append(np.asarray(res.std_errors[name]) / np.sqrt(ess.reshape(np.shape(res.params[name]))))  # type: ignore[attr-defined]
        gap = np.abs(np.asarray(fits["gibbs"].params[name]) - np.asarray(fits["nuts"].params[name]))  # type: ignore[attr-defined]
        worst = max(worst, float(np.max(gap / np.sqrt(se[0] ** 2 + se[1] ** 2))))
    ok = worst <= 2.0
    row = parity_log(
        ParityRow(
            "gaussian / geometric (Appendix E)",
            "gibbs vs nuts",
            "same priors, posterior means",
            "2 MC SE",
            f"{worst:.2f} MC SE",
            "1/1" if ok else "0/1",
            "ok" if ok else "FAILED",
        )
    )
    assert row.status == "ok", row
