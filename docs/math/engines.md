# Engines

!!! note
    Expanded engine (E4-T03a/b) documented below; explicit-duration engine (E4-T06a/b) and the final pass follow in E8-T04.

The forward recursion in log space, over the expanded state space built by `statespace` (ADR-0004):

$$
\log \alpha_t(j) = \log\!\left(\sum_i \alpha_{t-1}(i)\, A_t(i,j)\right) + \log b_j(y_t), \qquad
\log p(y_{1:T}) = \operatorname{logsumexp}_j \log \alpha_T(j)
$$

## Backward recursion and smoothing

With $\ell_t(s) = \log b_{r(s)}(y_t)$ (regime of state $s$; a fully masked step has $\ell_t \equiv 0$),
the backward pass runs a reverse `lax.scan`, rebuilding $A_{t+1}$ at each step (the $(T, S, S)$ array is never stored):

$$
\log \beta_T(s) = 0, \qquad
\log \beta_t(s) = \operatorname{logsumexp}_{s'}\!\big[\log A_{t+1}(s, s') + \ell_{t+1}(s') + \log \beta_{t+1}(s')\big]
$$

The smoothed posterior and its regime marginal (with the collapse matrix $C$, $C_{s,i} = 1$ iff state $s$ belongs to regime $i$):

$$
\gamma_t(s) = \operatorname{softmax}_s\big(\log \alpha_t(s) + \log \beta_t(s)\big), \qquad
\gamma^{\text{regime}}_t = \gamma_t\, C
$$

Invariants used by the tests: $\operatorname{logsumexp}_s(\log\alpha_t + \log\beta_t) = \log p(y_{1:T})$ for every $t$, and
the log-likelihood does not depend on the ordering of the expanded states. Structural zeros ($-\infty$) are masked
before `exp`, and a `logsumexp` over an all-$-\infty$ slice returns $-\infty$ with a zero gradient.

## Augmented posterior summaries

`engine/augmented.py` buckets the smoothed posterior by `(regime, phase)` (`phase_marginals`, `(T, N, r_max)`) and by
memory row (`memory_marginals`, `(T, R)`). For the negative-binomial phase chain, in phase $c$ of $r_i$ the expected
remaining sojourn is $(r_i - c + 1)/a_i(t)$ steps, and the phase-based lower bound on the elapsed time is
$(c - 1)/a_i(t)$; both are averaged over the phase marginals conditional on regime $i$ (0 where the regime has no mass).
Exact elapsed-duration posteriors come from the explicit-duration engine.

## Viterbi and information criteria (E4-T05)

`engine/viterbi.py` runs max-product on the expanded space in log space:
$\delta_t(s) = \ell_t(s) + \max_{s'}[\delta_{t-1}(s') + \log A_t(s', s)]$, storing only the
argmax table $\psi_t$ of shape `(T, S)` (int32); a reverse scan backtracks the path. Ties break to
the lowest state index. The regime path is `regime_of[path]`, and `segments` run-length encodes it as
`(start, end, regime)` with `end` inclusive. Within a regime segment the phase-advance moments of the negative-binomial chain can shift at
identical joint probability when the advance probabilities agree, so distinct expanded paths with the
same regime path can tie exactly.

`engine/criteria.py`: with $p$ = `ParamManager.n_free`, $T$ the number of unmasked steps and
$\ell$ the log-likelihood, $\mathrm{AIC} = 2p - 2\ell$, $\mathrm{BIC} = p\ln T - 2\ell$ and
$\mathrm{ICL} = \mathrm{BIC} + 2H(\gamma)$ with
$H(\gamma) = -\sum_t\sum_s \gamma_t(s)\ln\gamma_t(s)$ over the smoothed expanded posterior.

## Parallel-scan smoothing and benchmarks (E4-T07)

With `estimation.parallel_scan: true` (engine `ExpandedHMMEngine(parallel_scan=True)`, off by default) the
smoothing pass of `forward_backward` uses `jax.lax.associative_scan` over the log-space matrix products
$M_t = \log A_t + \ell_t^{\top}$: prefix products give
$\alpha_t = \alpha_0 \otimes M_1 \otimes \dots \otimes M_t$, suffix products
$\beta_t = M_{t+1} \otimes \dots \otimes M_{T-1} \otimes \mathbf 1$. Each matrix is stored as
$\exp(M - c)$ with a scalar scale $c$, so a product is a plain matrix multiplication; entries more than about
700 nats below the maximum of their matrix (a prefix or suffix product) underflow to $0$. Loglik, smoothed and
filtered probabilities are unaffected (agreement $10^{-10}$, tested at $T = 2000$ with sharp emissions), but
`log_alpha` / `log_beta` can be $-\infty$ for such entries where the sequential scan is finite. The parallel
path is not differentiated (estimation uses the sequential `loglik`). Depth is $O(\log T)$ but the $(T, S, S)$ tables need $T S^2 \le 5 \cdot 10^7$ cells (otherwise a
warning and the sequential scan). Results equal the sequential scan to $10^{-10}$; the option pays off on GPUs.

Performance guardrails (NFR-02, `tests/benchmark`, `pytest -m benchmark`): expanded `forward_backward` for
$T = 5000$, $N = 4$, NB $r = 5$, two covariates: at most 100 ms; explicit-duration `forward_backward` with
$D_{\max} = 250$: at most 2 s. `benchmarks/baseline.json` holds the reference means; the nightly job fails on a
30 % regression (`scripts/compare_benchmarks.py`).

## Symbols and code names

| Math | Code |
|---|---|
| `\log \alpha_t` | `log_alpha` inside `ExpandedHMMEngine` |
| `\ell` | `InferenceEngine.loglik` (0-d array) |
| filtered / smoothed posteriors | `InferenceEngine.forward_backward` -> `Posterior` |
| Viterbi path | `InferenceEngine.viterbi` |
