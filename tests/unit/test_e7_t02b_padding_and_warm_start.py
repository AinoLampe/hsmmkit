"""E7-T02b building blocks: masked padding, mask-aware initialization and the `fit` warm start."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.config.schema import EstimationSection
from regimekit.estimation.init import observed_rows, rollvol_kmeans
from regimekit.estimation.map import fit
from regimekit.estimation.objective import make_objective
from regimekit.evaluation.padding import PAD_MULTIPLE, pad_dataset, padded_length
from regimekit.model.params.manager import ParamManager
from regimekit.statespace.builders import build_expanded
from regimekit.utils.errors import ConfigError
from regimekit.utils.rng import key_for
from tests.helpers.objective_fixtures import make_case


def test_e7_t02b_padded_length_is_the_next_multiple_of_256() -> None:
    """`padded_length` rounds up to a multiple of 256 (at least one) and validates the multiple."""
    assert [padded_length(n) for n in (1, 255, 256, 257, 512, 513)] == [256, 256, 256, 512, 512, 768]
    assert padded_length(10, multiple=64) == 64
    with pytest.raises(ConfigError, match="multiple"):
        padded_length(10, multiple=0)


def test_e7_t02b_pad_dataset_appends_masked_rows_and_keeps_the_original() -> None:
    """Padded rows: `mask=False`, positive `dt`, zero covariates, increasing times; originals untouched."""
    case = make_case("gbm", "negbin", 1, t_len=200, seed=3, cov_tr=True, cov_du=True, cov_em=True)
    data = case.data
    padded = pad_dataset(data)
    t = data.times.shape[0]
    assert padded.times.shape[0] == padded.observations.shape[0] == PAD_MULTIPLE * -(-t // PAD_MULTIPLE)
    pad_dataset(make_case("gbm", "negbin", 1, t_len=200, seed=3).data).validate()  # contract-valid
    for name in ("observations", "dt", "mask", "cov_transition", "cov_duration", "cov_emission"):
        before, after = np.asarray(getattr(data, name)), np.asarray(getattr(padded, name))
        np.testing.assert_array_equal(after[:t], before, err_msg=name)
    assert not np.asarray(padded.mask)[t:].any()
    assert np.all(np.asarray(padded.dt)[t:] == np.asarray(data.dt)[-1])
    for name in ("cov_transition", "cov_duration", "cov_emission"):
        assert not np.asarray(getattr(padded, name))[t:].any()
    assert padded.prep_stats is data.prep_stats
    assert pad_dataset(padded) is padded  # already a multiple of 256


def test_e7_t02b_padded_loglik_and_filter_equal_the_unpadded_ones() -> None:
    """Appending masked rows leaves `loglik` (<= 1e-8 rel) and the filter rows `0..T-1` unchanged."""
    case = make_case("gbm", "negbin", 2, t_len=200, seed=3, cov_tr=True)
    theta = case.manager.unflatten(case.u0)
    padded = pad_dataset(case.data)
    ll = float(case.engine.loglik(theta, case.spec, case.space, case.data))
    ll_pad = float(case.engine.loglik(theta, case.spec, case.space, padded))
    assert ll_pad == pytest.approx(ll, rel=1e-8)
    alpha, _ = case.engine.forward(theta, case.spec, case.space, case.data)
    alpha_pad, _ = case.engine.forward(theta, case.spec, case.space, padded)
    np.testing.assert_allclose(np.asarray(alpha_pad)[:200], np.asarray(alpha), rtol=0, atol=1e-10)


def test_e7_t02b_observed_rows_is_a_noop_without_masked_rows_and_drops_padding() -> None:
    """No masked row: the very same object; with padding: exactly the original rows."""
    case = make_case("gbm", "negbin", 1, t_len=200, seed=3, cov_em=True)
    assert observed_rows(case.data) is case.data
    trimmed = observed_rows(pad_dataset(case.data))
    for name in ("times", "observations", "dt", "mask", "cov_emission"):
        np.testing.assert_array_equal(
            np.asarray(getattr(trimmed, name)), np.asarray(getattr(case.data, name))
        )


def test_e7_t02b_padding_rows_no_longer_bias_the_data_driven_init() -> None:
    """`rollvol_kmeans` on the observed rows of a padded window equals the init on the raw window."""
    case = make_case("gbm", "negbin", 1, t_len=200, seed=3)
    padded = pad_dataset(case.data)
    key = key_for(0, 1)
    plain = rollvol_kmeans(case.data, case.spec, key)
    via_observed = rollvol_kmeans(observed_rows(padded), case.spec, key)
    biased = rollvol_kmeans(padded, case.spec, key)
    for name in plain:
        np.testing.assert_array_equal(np.asarray(via_observed[name]), np.asarray(plain[name]), err_msg=name)
    assert not np.allclose(np.asarray(biased["emission.scale"]), np.asarray(plain["emission.scale"]))


def _cfg(restarts: int = 1) -> EstimationSection:
    return EstimationSection(restarts=restarts, init=["rollvol_kmeans"], max_iter=60, device="cpu")


def test_e7_t02b_warm_start_adds_one_extra_restart_last() -> None:
    """`warm_start` appends a restart (`init_strategy == "warm_start"`); regular restarts are unchanged."""
    case = make_case("gbm", "negbin", 1, t_len=200, seed=3)
    plain = fit(case.spec, case.data, _cfg(), seed=4)
    warm = fit(case.spec, case.data, _cfg(), seed=4, warm_start=plain.uparams)
    assert list(warm.restarts["init_strategy"]) == ["rollvol_kmeans", "warm_start"]
    assert warm.restarts.loc[0, "objective"] == plain.restarts.loc[0, "objective"]  # same key, same restart
    # started at the previous optimum: at least as good as the cold restart it was added to
    assert warm.restarts.loc[1, "objective"] <= warm.restarts.loc[0, "objective"] + 1e-6
    assert warm.objective <= plain.objective + 1e-9
    again = fit(case.spec, case.data, _cfg(), seed=4, warm_start=plain.uparams)
    assert np.array_equal(np.asarray(again.uparams), np.asarray(warm.uparams))


def test_e7_t02b_warm_start_with_the_wrong_shape_is_a_config_error() -> None:
    """A `warm_start` of the wrong length is a `ConfigError` with the estimation path."""
    case = make_case("gbm", "negbin", 1, t_len=100, seed=3)
    bad = jnp.zeros((case.manager.n_free + 1,))
    with pytest.raises(ConfigError, match="warm_start") as err:
        fit(case.spec, case.data, _cfg(), seed=4, warm_start=bad)
    assert err.value.path == "estimation.warm_start"


def test_e7_t02b_objective_and_gradient_on_a_padded_window_equal_the_unpadded_ones() -> None:
    """At a fixed `u` the MAP objective and its gradient do not see the masked rows (1e-10).

    Whole fits of a flat likelihood can still end at different points: L-BFGS amplifies the 1e-13
    round-off that the longer scan introduces, so end-to-end equality needs a well-conditioned fit
    (`test_e7_t02b_padding_gives_the_same_results_as_no_padding`).
    """
    case = make_case("gbm", "negbin", 1, t_len=200, seed=3, priors=True)
    u = case.u0 + 0.05
    out = []
    for data in (case.data, pad_dataset(case.data)):
        objective = make_objective(case.spec, case.space, case.engine, case.manager, data, use_priors=True)
        out.append(objective.value_and_grad(u))
    (v0, g0), (v1, g1) = out
    assert float(v1) == pytest.approx(float(v0), rel=1e-12)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g0), rtol=1e-10, atol=1e-10)


def test_e7_t02b_param_manager_and_state_space_are_reused_across_builds() -> None:
    """Equal specs give equal, equally hashed managers and the very same `StateSpace` (jit cache keys)."""
    case = make_case("gbm", "negbin", 2, t_len=40, seed=3)
    again = ParamManager(case.spec.param_specs)
    assert again == case.manager
    assert hash(again) == hash(case.manager)
    assert again != ParamManager(case.spec.param_specs[:-1])
    assert build_expanded(case.spec) is build_expanded(case.spec)


def test_e7_t02b_explicit_engine_ignores_padded_rows() -> None:
    """The explicit-duration engine also gives the same `loglik` and filter rows on a padded window (1e-8)."""
    case = make_case("gbm", "gamma", 1, t_len=120, seed=3)
    theta = case.manager.unflatten(case.u0)
    padded = pad_dataset(case.data)
    ll = float(case.engine.loglik(theta, case.spec, case.space, case.data))
    assert float(case.engine.loglik(theta, case.spec, case.space, padded)) == pytest.approx(ll, rel=1e-8)
    alpha, _ = case.engine.forward(theta, case.spec, case.space, case.data)
    alpha_pad, _ = case.engine.forward(theta, case.spec, case.space, padded)
    np.testing.assert_allclose(np.asarray(alpha_pad)[:120], np.asarray(alpha), rtol=0, atol=1e-8)
