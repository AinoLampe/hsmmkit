# E16-T01 — scikit-learn `HSMM` estimator (`sklearn` extra)

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-IO01, D20 | M12 | Part II Appendix I, §18 (sklearn impedance), §15.5 |

## Goal
Offer a familiar `fit/predict` API for users coming from scikit-learn and `hmmlearn`, mapped onto the full configuration underneath.

## Context (self-contained)
`hsmmkit.interop.sklearn.HSMM(BaseEstimator)` (imports scikit-learn via `optional_import(..., extra="sklearn")`); mapping per Appendix I:
| call | behaviour |
|---|---|
| `HSMM(n_states=3, emission="gaussian", duration="negbin", r=3, topology="full", estimator="map", engine="expanded", covariance="diag", covariate_columns=None, random_state=None, **config_overrides)` | builds a `ProjectConfig` with `clock: steps`; constructor stores args only (sklearn rule); `config_overrides` keys are dotted config paths (`"estimation.restarts": 4`) |
| `get_params/set_params` | flat dict of constructor args incl. `config_overrides` |
| `fit(X, lengths=None, y=None, sample_weight=None)` | `X (ΣT_b, M)`; `lengths` as `hmmlearn` (None = one sequence); `covariate_columns` indices split off `X` into covariates (used for transitions/durations/emissions as declared in overrides, default transitions); `y` int labels with `-1` unknown → `data.labels` allowed-sets; `sample_weight` → `ValueError` (unsupported) |
| `predict(X, lengths)` | Viterbi regimes concatenated `(ΣT_b,)` |
| `predict_proba(X, lengths, filtered=False)` | smoothed (or filtered) regime probabilities `(ΣT_b, N)` |
| `score(X, lengths)` | total log-likelihood; `score_samples` per-step one-step predictive log-density |
| `decode(X, lengths)` | `(loglik, regimes)` (hmmlearn signature) |
| `sample(n_samples=1, random_state=None)` | `(X, regimes)` via simulate |
| `forecast(X, horizon, lengths=None)` | `ForecastResult` for the last sequence (extension) |
| fitted attributes | `result_` (`FitResult`), `n_features_in_`, `n_states_`, `means_`/`startprob_`-like convenience properties where meaningful |
- `random_state` → `project.seed` (int; `None` → 0 with a documented note; `RandomState` objects → their first draw).
- Input validation via `sklearn.utils.validation.check_array` (float64 for real support, int for discrete supports declared by the emission).

## Creates
- `src/hsmmkit/interop/sklearn.py`, `tests/integration/test_e16_t01_sklearn.py` (marker `sklearn`)
- `docs/guide/core/sklearn.md`

## Acceptance criteria
1. The applicable `sklearn.utils.estimator_checks.check_estimator` checks pass (documented exclusion list with reasons: sample weights, 1-D inputs, …, maintained in the test).
2. `lengths` semantics match `hmmlearn` (same splitting on a shared fixture; `GaussianHMM`-equivalent config gives the same loglik as E16-T02's import within 1e-8).
3. Labels via `y` reach the constraint layer (fully labelled `y` → `predict` returns `y`).
4. `clone(est)` and `GridSearchCV` with a custom scorer (`score`) work on a small example.

## Tests
Integration with scikit-learn's checks, behaviour tests above.

## Out of scope
Pipelines with transformers changing `lengths` (documented limitation).

## Connections
- Depends on: E12-T02, E12-T06, E10-T11
- Blocks: E17-T03
- Relates to: E16-T02 (`hmmlearn` import)
