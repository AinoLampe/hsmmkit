"""`cross_validate`: out-of-sample scores over the folds of a `Splitter` (E14-T01; Part II §14.12).

**Sequence CV** (`SequenceKFold`): per fold the model is fitted on the training sequences (the preparer fitted
on them only) and every test sequence is scored out of sample with the training estimate `theta_hat`:

* `log_score`: `loglik_b(theta_hat)` per observed step (the per-step log predictive density of the sequence);
  the labels and boundaries of the test data never enter the scoring (they only serve the label metrics);
* `crps`: the continuous ranked probability score of the one-step-ahead predictive distribution
  `sum_i P(regime_t = i | y_{1:t-1}) f_i`, estimated from `estimation.n_sim` draws (`crps_from_samples`),
  averaged over the observed steps and series (expanded engine; `NaN` for the explicit engine);
* every configured metric that needs only a decoded posterior (`aic`, `bic`, `icl`, `regime_*`, segmentation
  metrics when the test sequence has single-regime labels) evaluated on the test sequence with `theta_hat`.

**Walk-forward** (`WalkForward`): delegates to `run_backtest` (E7-T02b), so the folds, metrics, summary and
hook frames are bitwise those of the backtest; only the long-format `scores` frame is added.

Imports from `hsmmkit.data` are limited to `contracts` and `batch`; the preparer is `preparer_factory`.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from hsmmkit.estimation import fit as fit_model
from hsmmkit.evaluation.backtest import OriginHook, run_backtest
from hsmmkit.evaluation.metrics.predictive import predictive_regime_probs
from hsmmkit.evaluation.metrics.registry import resolve
from hsmmkit.inference.decode import decode
from hsmmkit.model.build import build_model_spec
from hsmmkit.selection.splitters import PreparerFactory, RawData, SequenceKFold, Splitter, WalkForward
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.logging import get_logger
from hsmmkit.utils.rng import PURPOSE_CV_FIT, PURPOSE_CV_SCORE, derive_seed, key_for

if TYPE_CHECKING:
    from hsmmkit.config.schema import ProjectConfig
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.inference.results import PosteriorResult

__all__ = ["CVResult", "cross_validate"]

_LOG = get_logger("hsmmkit.cv")
_COLUMNS = ["fold", "sequence", "horizon", "metric", "value"]


@dataclass(frozen=True)
class CVResult:
    """Folds, long-format scores and their summary.

    Attributes:
        folds: one row per fold (sizes, `fit_time`, fit summary; the backtest's table for walk-forward).
        scores: long format `fold, sequence, horizon, metric, value` (`horizon` is 1 for the one-step
            scores of a sequence CV, 0 for per-sequence posterior metrics; the backtest's horizons for
            walk-forward).
        summary: `horizon, metric, mean, sd, n` across the folds.
        extras: the origin hooks' frames (walk-forward).
    """

    folds: pd.DataFrame
    scores: pd.DataFrame
    summary: pd.DataFrame
    extras: dict[str, pd.DataFrame] = field(default_factory=dict)


def _sample_crps(
    fit: FitResult,
    seq: PreparedDataset,
    post: PosteriorResult,
    n_sim: int,
    key: jax.Array,
) -> float:
    """Mean sample CRPS of the one-step predictive over the observed steps (`NaN` for the explicit engine)."""
    spec = fit.spec
    if spec.engine != "expanded":
        return float("nan")
    from hsmmkit.statespace.builders import build_expanded

    space = build_expanded(spec)
    probs = predictive_regime_probs(post.augmented, fit.params, spec, space, seq)  # (T, N)
    t_len = int(seq.observations.shape[0])
    z = seq.cov_emission if seq.cov_emission is not None else jnp.zeros((t_len, 0))
    logits = jnp.log(jnp.maximum(probs, 1e-300))
    keys = jax.random.split(key, n_sim)

    def draw(k: jax.Array) -> jax.Array:
        k1, k2 = jax.random.split(k)
        regime = jax.random.categorical(k1, logits, axis=-1)  # (T,)
        return jnp.asarray(spec.emission.sample(k2, regime, seq.dt, fit.params, z))  # (T, M)

    samples = np.sort(np.asarray(jax.vmap(draw)(keys)), axis=0)  # (n, T, M)
    y = np.asarray(seq.observations)  # (T, M)
    mask = np.asarray(seq.mask)
    n = samples.shape[0]
    weights = (2.0 * np.arange(1, n + 1) - n - 1.0)[:, None, None]
    term_y = np.mean(np.abs(samples - y[None]), axis=0)
    term_xx = 2.0 * np.sum(weights * samples, axis=0) / n**2
    crps = term_y - 0.5 * term_xx  # (T, M)
    observed = mask  # per series
    return float(np.mean(crps[observed])) if observed.any() else float("nan")


def _posterior_metrics(
    cfg: ProjectConfig,
    fit: FitResult,
    seq: PreparedDataset,
    post: PosteriorResult,
) -> dict[str, float]:
    """The configured metrics that need only the decoded posterior (and labels)."""
    out: dict[str, float] = {}
    labels = None
    if seq.labels is not None:
        lab = np.asarray(seq.labels)
        if bool(np.all(lab.sum(axis=-1) == 1)):  # one admissible regime per step: usable as reference labels
            labels = lab.argmax(axis=-1)
    boundary = cfg.evaluation.boundary_tolerance if hasattr(cfg.evaluation, "boundary_tolerance") else None
    for m in resolve(list(cfg.evaluation.metrics)):
        needs = set(m.needs)
        if not needs or not needs <= {"posterior", "fit", "labels"}:
            continue
        if "labels" in needs and labels is None:
            continue
        ctx: dict[str, object] = {"posterior": post, "fit": fit}
        if labels is not None:
            ctx["labels"] = labels
        if boundary is not None:
            ctx["boundary_tolerance"] = boundary
        value = m(**ctx)
        if isinstance(value, dict):
            out.update({f"{m.name}_{k}": float(v) for k, v in value.items()})
        else:
            out[m.name] = float(value)
    return out


def _summary(scores: pd.DataFrame) -> pd.DataFrame:
    """`horizon, metric, mean, sd, n`: per-fold means of the sequence scores, then mean / sd across folds."""
    per_fold = scores.groupby(["fold", "horizon", "metric"], sort=True)["value"].mean().reset_index()
    rows = []
    for (horizon, name), g in per_fold.groupby(["horizon", "metric"], sort=True):
        v = g["value"].to_numpy()
        rows.append(
            {
                "horizon": int(horizon),  # type: ignore[call-overload]
                "metric": str(name),
                "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)) if v.shape[0] > 1 else float("nan"),
                "n": int(v.shape[0]),
            }
        )
    return pd.DataFrame(rows, columns=["horizon", "metric", "mean", "sd", "n"])


def cross_validate(
    cfg: ProjectConfig,
    raw: RawData,
    splitter: Splitter,
    seed: int,
    preparer_factory: PreparerFactory,
    hooks: Sequence[OriginHook] = (),
    *,
    on_fold: Callable[[int, FitResult], None] | None = None,
) -> CVResult:
    """Cross-validate the configured model with `splitter`.

    Args:
        cfg: the project configuration (model, estimation, `forecast.n_sim`, metrics).
        raw: `(ObservationSet, covariates)` before preparation.
        splitter: `SequenceKFold` or `WalkForward` (or any `Splitter`).
        seed: the base seed (`derive_seed(seed, PURPOSE_CV_FIT, fold)` seeds each fit).
        preparer_factory: builds an unfitted preparer for the configuration.
        hooks: origin hooks (walk-forward only).
        on_fold: called as `on_fold(fold, fit)` after each fit (sequence CV).

    Raises:
        ConfigError: unusable data for the splitter (no sequence ids, too few sequences) or a missing
            `evaluation.backtest` for the walk-forward splitter.
    """
    if isinstance(splitter, WalkForward):
        result = run_backtest(cfg, lambda: raw, seed, hooks)
        scores = result.metrics.assign(sequence="0")[_COLUMNS]
        return CVResult(result.folds, scores, result.summary, result.extras)
    if hooks:
        raise ConfigError(
            "origin hooks only apply to the walk_forward splitter", path="evaluation.cv.splitter"
        )
    n_sim = cfg.forecast.n_sim
    fold_rows: list[dict[str, object]] = []
    score_rows: list[tuple[int, str, int, str, float]] = []
    for split in splitter.split(raw, cfg.evaluation, preparer_factory):
        spec = build_model_spec(cfg, split.train)
        started = time.perf_counter()
        fit = fit_model(spec, split.train, cfg.estimation, derive_seed(seed, PURPOSE_CV_FIT, split.fold))
        fit_time = time.perf_counter() - started
        if on_fold is not None:
            on_fold(split.fold, fit)
        if fit.spec.n_groups > 1:
            raise ConfigError(
                "cross-validation does not support per-group initial distributions yet",
                path="model.initial_groups",
            )
        for b, (sid, seq_full) in enumerate(zip(split.test.ids, split.test.sequences, strict=True)):
            # the labels / boundaries of the test data are for the metrics only: never constrain the scoring
            seq = dataclasses.replace(seq_full, labels=None, boundaries=None)
            post = decode(fit, seq)
            n_obs = max(int(np.asarray(seq.mask).any(axis=-1).sum()), 1)
            score_rows.append((split.fold, sid, 1, "log_score", float(post.loglik) / n_obs))
            key = key_for(derive_seed(seed, PURPOSE_CV_SCORE, split.fold), b)
            score_rows.append((split.fold, sid, 1, "crps", _sample_crps(fit, seq, post, n_sim, key)))
            for name, value in _posterior_metrics(cfg, fit, seq_full, post).items():
                score_rows.append((split.fold, sid, 0, name, value))
        fold_rows.append(
            {
                "fold": split.fold,
                "n_train": len(split.train.sequences),
                "n_test": len(split.test.sequences),
                "fit_time": fit_time,
                "objective": fit.objective,
                "loglik": fit.loglik,
                "converged": fit.converged,
                "n_iter": fit.n_iter,
            }
        )
        _LOG.info("cv_fold_done", fold=split.fold, n_test=len(split.test.sequences))
    scores = pd.DataFrame(score_rows, columns=_COLUMNS).astype(
        {"fold": "int64", "horizon": "int64", "value": "float64"}
    )
    scores = scores.sort_values(["fold", "sequence", "horizon", "metric"], kind="stable").reset_index(
        drop=True
    )
    return CVResult(pd.DataFrame(fold_rows), scores, _summary(scores))


def make_splitter(cfg: ProjectConfig) -> Splitter:
    """The splitter `evaluation.cv.splitter` names (registry kind `splitter`).

    Raises:
        ConfigError: `evaluation.cv` is absent or the splitter is not registered.
    """
    from hsmmkit.plugins import REGISTRY

    cv = cfg.evaluation.cv
    if cv is None:
        raise ConfigError("evaluation.cv is required for cross-validation", path="evaluation.cv")
    cls = REGISTRY.get("splitter", cv.splitter)
    group_by = cfg.data.sequences.group_by if cfg.data.sequences is not None else None
    out: Splitter = cls.from_config(cv, group_by)
    return out


__all__ += ["SequenceKFold", "make_splitter", "render_cv_section", "replace_cv_section"]


_CV_HEADING = "## Cross-validation"


def render_cv_section(result: CVResult, splitter: str) -> str:
    """The `## Cross-validation` section of `report.md` (summary table across folds)."""
    lines = [
        _CV_HEADING,
        "",
        f"Splitter `{splitter}`, {len(result.folds)} folds. Mean over folds ± sd across folds "
        "(`log_score`: per-step out-of-sample log-likelihood, higher is better; `crps`: one-step "
        "predictive CRPS, lower is better).",
        "",
        "| horizon | metric | mean | sd | n |",
        "|---|---|---|---|---|",
    ]
    for _, row in result.summary.iterrows():
        cells = (int(row["horizon"]), row["metric"], f"{row['mean']:.6g}", f"{row['sd']:.3g}", int(row["n"]))
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def replace_cv_section(report: str, section: str) -> str:
    """`report` with its cross-validation section replaced by `section` (appended when absent)."""
    start = report.find(_CV_HEADING)
    if start < 0:
        return (report.rstrip("\n") + "\n\n" if report.strip() else "") + section
    end = report.find("\n## ", start + len(_CV_HEADING))
    tail = "" if end < 0 else report[end + 1 :]
    return report[:start] + section + ("\n" + tail if tail else "")
