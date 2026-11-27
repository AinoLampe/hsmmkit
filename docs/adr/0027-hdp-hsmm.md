# ADR-0027: HDP-HSMM (weak-limit) — design spike and go / no-go

Status: accepted (E13-T09). Decision: **no-go for 0.x up to 0.5.0; go as a post-0.5.0 epic** with the interface
and effort given below.

## Context

A hierarchical-Dirichlet-process HSMM would let the data decide the number of regimes instead of fixing
`model.n_states`. The weak-limit approximation (Johnson & Willsky 2013) truncates the HDP at `L` states:

* global weights `beta ~ Dirichlet(gamma / L, ..., gamma / L)`;
* jump rows `pi_i ~ Dirichlet(alpha beta)` with the self-transition removed (the HSMM's segments are separated by
  real jumps), i.e. `Dirichlet(alpha beta_{-i})` over the other `L - 1` states;
* explicit durations `p_i(d)` per state (nonparametric `Dirichlet(lambda)` or a parametric family) and emissions
  with conjugate priors;
* a blocked Gibbs sampler: segment-level forward-filtering backward-sampling of the state / duration path given
  the parameters, then conjugate draws of emissions and durations, and for `(beta, pi)` the Chinese-restaurant
  auxiliary variables `m_ij` (the number of "tables" of the jump counts `n_ij`) followed by
  `beta | m ~ Dirichlet(gamma/L + m_.j)` and `pi_i | n ~ Dirichlet(alpha beta + n_i.)`.

Questions of the spike and what the repository already provides (0.3.0 APIs):

| # | Question | Answer |
|---|---|---|
| 1 | Can the existing FFBS and conjugate blocks serve the sampler unchanged? | **FFBS: yes.** `engine.ffbs.ffbs_explicit` draws exact regime paths (right-censored last segment, topology, labels) for the explicit engine; `estimation/gibbs.py` already implements the sweep for the expanded engine (path -> counts -> conjugate draws). Missing: the sweep for the **explicit** engine (counts from sampled *segments*: `eta`, `xi`; nonparametric durations get a Dirichlet block) and the self-transition-free jump rows. |
| 2 | New conjugate blocks | One: the **hierarchical Dirichlet block** for `(beta, pi)` (auxiliary table counts by the usual Antoniak sampler, `jax.random` only), plus the standard Dirichlet block for nonparametric durations (an `eta` count update; `model.conjugate.DirichletConjugate` already has `update` / `sample_params`). The Gamma hyperpriors on `alpha`, `gamma` (Escobar-West auxiliary updates) are optional first. |
| 3 | Cost at `L = 20`, `D_max = 200`, `T = 10 000` | Measured with `spikes/hdp_hsmm_cost.py` on the 4-core cloud VM used for the tickets (CPU, x64): **one explicit-engine FFBS sweep: 0.43 s warm** (1.2 s first call), peak RSS **576 MB**; the explicit E-step (not needed by Gibbs) takes 9.4 s warm. Smaller cases: `L = 3, D = 60, T = 1000`: 0.01 s; `L = 10, D = 100, T = 5000`: 0.11 s. A 2000-sweep chain at the large size is ~15 minutes per chain on this machine, chains `vmap`ped, so the sweep cost is acceptable; memory is dominated by the `(T, N, D_max)` guard (`5e7` cells, here `4e7`), which sets the practical ceiling of `L D_max T`. |
| 4 | Interaction with relabelling and pruning | Unused states play the role of pruned states: the posterior over the **number of occupied states** is read from the sampled paths (`path_samples`, `keep_paths_every`), and `estimation.vb`'s occupancy / `prune_below` logic carries over unchanged to the per-sweep occupancy. Relabelling (`estimation.relabel`) works on the used states; unused states have prior-only parameters and must be excluded from the pivot matching (a mask on the occupancy, small change). The HDP is exchangeable in the labels only up to the `beta` ordering: relabel by decreasing `beta` instead of the MAP pivot (a variant of `relabel_draws`). |
| 5 | API surface | `estimation.mcmc.sampler: hdp_gibbs` (new value of the existing literal); `model.n_states` is the truncation `L`; a new section `estimation.mcmc.hdp: {alpha, gamma, duration_lambda, sample_hyperparameters}`; `engine: explicit` required; the result keeps the `MCMCEstimator` contract and adds `diagnostics["mcmc"]["n_occupied"]` (draws of the number of used states) and the `beta` draws in `posterior`. No change to `FitResult`, the registries or the bundle. |
| 6 | Testing strategy | (a) exact: for `L = 2` and a tiny `T`, the sweep's conditional draws against enumeration of the segment-path posterior; (b) Geweke / joint-distribution test of the full sweep; (c) recovery: 3 true regimes, `L = 10`: the posterior mode of `n_occupied` is 3 in at least 4 of 5 seeds; (d) agreement with the finite-`N` Gibbs sampler (`sampler: gibbs`, `engine: explicit`) when `beta` is fixed uniform; (e) the usual reproducibility-by-seed test. |

A prototype of the occupancy concentration (the optional part of the ticket) was **not** written: it would only
re-demonstrate Johnson & Willsky's result on a sampler that does not exist here, while the questions above (reuse,
cost, interface) are answered by the existing code and the measurement.

## Decision

* **No-go before 0.5.0.** The 0.5.0 milestone is the general release of the existing estimators; an HDP sampler
  adds a new inference algorithm, hyperparameters and diagnostics, and it is not required by any 0.5.0 criterion.
* **Go afterwards**, as an epic of about **4 M + 2 S** tickets (10-12 working days):
  1. M — explicit-engine blocked Gibbs sweep (segment counts, nonparametric Dirichlet durations; also useful for
     the finite `sampler: gibbs` on `engine: explicit`);
  2. M — hierarchical Dirichlet block (`beta`, `pi`, auxiliary table counts, self-transition-free rows) with its own
     conjugacy / Geweke tests;
  3. M — `sampler: hdp_gibbs` assembly, `n_occupied` diagnostics, hyperparameter updates (Escobar-West);
  4. M — recovery suite (3 true regimes, `L = 10 / 20`) and the Geweke harness;
  5. S — relabelling by `beta` and the occupancy mask;
  6. S — documentation (`estimators.md` row, tutorial).
* The weak-limit form is the only variant considered; the direct-assignment sampler and slice samplers are out of
  scope (they do not reuse the segment FFBS).

## Interface diff against 0.3.0

| Area | Change |
|---|---|
| config | `estimation.mcmc.sampler` gains `hdp_gibbs`; new `estimation.mcmc.hdp` section (`alpha`, `gamma`, `duration_lambda`, `sample_hyperparameters`); a feature-gate row until the epic lands |
| `estimation/` | `gibbs.py`: explicit-engine sweep; new `hdp.py` (Antoniak table counts, `beta` / `pi` draws, hyperparameter updates); `relabel.py`: order-by-`beta` pivot and an occupancy mask |
| `model/conjugate.py` | `HierarchicalDirichlet` block (`update`, `sample_params`, `kl` is not needed by the sampler) |
| results | `diagnostics["mcmc"]["n_occupied"]`, posterior variable `beta`; no new `FitResult` field |
| engine | none (`ffbs_explicit` is used as is); the explicit E-step's `ExplicitEStats` already carries `eta` / `xi` |

## Consequences

* Until the epic exists, the number of regimes is chosen by `model.n_states` and, as a poor-man's substitute, by
  the pruning of `estimation.method: vb` (a model with too many regimes reports `pruned_states`).
* No code is merged to `src/` by this ticket; `spikes/hdp_hsmm_cost.py` is the measurement script.
