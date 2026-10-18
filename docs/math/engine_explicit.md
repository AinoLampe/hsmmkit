# Explicit-duration engine

`engine/explicit_duration.py` (`ExplicitDurationEngine`, `spec.engine == "explicit"`) runs the
Guédon/Yu forward-backward over *segments* for an arbitrary sojourn pmf $p_j(d)$, $d = 1..D_{\max}$.
The grid is assumed regular (ADR-0007): `nominal_dt` replaces $\Delta_t$ everywhere and an irregular
grid warns once per call.

## Recursion

Boundary $\tau \in \{0, \dots, T\}$ means $\tau$ steps consumed; a segment starting at boundary $s$
covers steps $s, \dots, s+d-1$. With $C[\tau, j] = \sum_{u<\tau} \ell_u(j)$ (masked steps add 0), the
emission of that segment is $C[s+d, j] - C[s, j]$ and

$$
E[t, j] = C[t, j] + \operatorname{logsumexp}_d \big( G[t-d, j] + \log p_{t-d}(j, d) \big), \qquad
\text{inner}[s, j] = \begin{cases} \log\rho_j & s = 0 \
\operatorname{logsumexp}_i \big( E[s, i] + \log\pi_s(i, j) \big) & s > 0 \end{cases}
$$

with $G = \text{inner} - C$. $\log p_s$ and $\log\pi_s$ are those of a segment *starting* at $s$
(entry-time covariates use covariate row $s$). The forward pass is one `lax.scan` over $s$ that
scatters each start into a rolling `(D_max, N)` buffer of pending ends: $O(N^2 + N D_{\max})$ per step.

## Right-censoring

The last segment may still be running at $T$. With $S_s(j, d) = P(\text{duration} \ge d)$ from the
same pmf,

$$
\log p(y_{1:T}) = \operatorname{logsumexp}_{s, j} \big( \text{inner}[s, j] + \log S_s(j, T-s)
+ C[T, j] - C[s, j] \big), \qquad T - s \le D_{\max}.
$$

## Backward and smoothing

$B[\tau, i]$ is the log-probability of the future given that a regime-$i$ segment ends at $\tau$. With
$H = C + B$ it needs only the last $D_{\max}$ values of $H$ (a reverse scan), the "difference of
cumulative" trick, and the final segment enters through the survival term above. The elapsed posterior
is, for step $t$ at elapsed $e$ (segment start $s = t - e + 1$),

$$
P(z_t = j, e \mid y) = \exp\!\Big( \operatorname{logsumexp}_{d \ge e} w(s, j, d) \oplus w_T(s, j) - \ell \Big)
$$

where $w$ are the closed-segment weights, $w_T$ the final-segment weight and $\ell$ the log-likelihood.
Suffix sums over $d$ are a reverse `cumlogsumexp`: $O(T N D_{\max})$ in total.

## Augmented state

State $s = j D_{\max} + (e - 1)$ (`statespace.builders.build_explicit`). The `Posterior` fields are

* `log_alpha[t, (j, e)] = log P(y_{1:t}, regime j, e steps elapsed) = G[t-e+1, j] + log S + C[t+1, j]`
  (so `softmax(log_alpha)` is the filter and `logsumexp(log_alpha[-1])` the log-likelihood),
* `smoothed[t, (j, e)]` the elapsed posterior (`Posterior.elapsed_posterior()` is the `(T, N, D_max)` view),
* `log_beta = log(smoothed) + loglik - log_alpha`, exact because `(regime, elapsed)` is Markov.

## Segmental Viterbi

`ExplicitDurationEngine.viterbi` is the max-product twin of the forward scan. A rolling buffer keeps, for
every pending segment end, the best score and the *start* that achieved it; at each boundary the best
previous regime is the argmax of the jump step. The last segment is scored with the survival term. The
backtrack is a reverse scan over steps that jumps from a segment start to the segment ending there in the
best previous regime. The result is a `(T,)` array of augmented states `j * D_max + (e - 1)`, so
`regime_of[path]` is the regime path (fed to `segments`, as for the expanded engine) and `phase_of[path]`
the elapsed count. Ties break to the shortest duration and then the lowest previous regime.

## Chunked scan

Forward and backward passes only store `(T, N)` arrays; the `(T, N, D_max)` tables come from the
post-processing, which `chunk_size` (`estimation.chunk_size`) evaluates for `chunk_size` steps at a
time: the steps `a..a+c-1` only involve the `c + D_max - 1` segment starts `s = a - D_max + 1 .. a + c - 1`,
so a chunk needs `O((c + D_max) N D_max)` memory, independent of `T`. A chunked `Posterior` is collapsed
over the elapsed count: `(T, N)` arrays with `phase_of == 1`. `chunk_size=None` chunks automatically
above `T N D_max = 5 \cdot 10^7` (chunks of `max(D_max, 1024)` steps); `loglik` and `viterbi` never build
`(T, N, D_max)` tables. With entry-time duration covariates the pmf table is itself `(T, N, D_max)`, so
those runs still raise above the limit.

## Equivalence with the expanded engine

On shifted-negative-binomial sojourns the phase chain and the explicit engine define the same model. The
golden test (`r = (2, 3)`, mean sojourn about 30 steps, `D_max = 400`, 20 simulated data sets) sees
log-likelihood differences below `1e-10` (relative) and smoothed-marginal differences below `1e-9`. The data
must come from the model: the truncation at `D_max` removes tail mass below `1e-10` from every sojourn but
that mass is *relatively* large in the survival of a final segment near `D_max` steps long, which
noise-like data can favour.

## Limits

Full `(T, N, D_max)` tables (`forward`, unchunked `forward_backward`) raise `NumericalError`
("use chunked scan") when `T N D_max > 5e7`. Only `order == 1`, `duration_conditioning == "current"` and
`initial in {uniform, free}` (over the first segment's regime) are supported.

## Symbols and code names

| Math | Code |
|---|---|
| `p_i(d)`, `h_i(d)` | `DurationFamily` explicit pmf / hazard |
| `D_{max}` | `model.d_max` |
| engine | `ExplicitDurationEngine` (`chunk_size` from `estimation.chunk_size`) |
| nominal step `\Delta` | `PreparedDataset.nominal_dt` |
