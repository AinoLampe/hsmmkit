"""E13-T08 spike: online EM (Cappe & Moulines 2009) for a Gaussian / geometric-duration HSMM (not packaged).

Pure NumPy prototype of the recursion in `docs/adr/0026-online-em.md`. A geometric duration on the expanded
engine makes the regime chain an ordinary `N`-state Markov chain, `A_ii = 1 - a_i`, `A_ij = a_i pi_ij`, so the
prototype keeps the full `(N, N)` matrix `A` and maps the transition statistics back to `(a_i, pi_ij)` in the
M-step, exactly as the closed-form EM M-step of `hsmmkit.estimation.em` does.

Statistics vector of one step `s(i, j, y)` (`K = N^2 + 3 N`): the transition one-hot `e_{ij}` (`N^2` entries)
and, for the current state `j`, `[1, y, y^2]` in the block of `j`. The recursion keeps `rho_t(j) in R^K`:

    phi_t(i | j) = alpha_{t-1}(i) A_ij / sum_i' alpha_{t-1}(i') A_i'j          (backward kernel)
    rho_t(j)     = sum_i phi_t(i | j) [(1 - g_t) rho_{t-1}(i) + g_t s(i, j, y_t)]
    S_t          = sum_j alpha_t(j) rho_t(j)                                  (alpha_t: filter at t)

With the parameters held fixed and `g_t = 1 / t`, `t S_t` equals the smoothed expected statistics
`sum_t E[s | y_1:t]` of the offline E-step (the recursion is exact), which the spike test checks against
`hsmmkit.engine.estep.expanded_estep`. With `g_t = t^-kappa` and `theta_t = m_step(S_t)` after a burn-in it
tracks slowly drifting parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_SIGMA_FLOOR = 1e-3


@dataclass
class Params:
    """Initial distribution, transition matrix and Gaussian emissions of an `N`-regime model."""

    pi0: np.ndarray
    a: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray

    def copy(self) -> Params:
        """Deep copy."""
        return Params(self.pi0.copy(), self.a.copy(), self.mu.copy(), self.sigma.copy())


def log_emission(theta: Params, y: float) -> np.ndarray:
    """`(N,)` Gaussian log-densities of `y`."""
    z = (y - theta.mu) / theta.sigma
    return -0.5 * z * z - np.log(theta.sigma) - 0.5 * np.log(2.0 * np.pi)


def m_step(stats: np.ndarray, n: int, previous: Params) -> Params:
    """Closed-form M-step from the statistics vector (geometric durations: stay / jump split of `A`)."""
    counts = stats[: n * n].reshape(n, n)
    s0, s1, s2 = (stats[n * n + k * n : n * n + (k + 1) * n] for k in range(3))
    a = previous.a.copy()
    row = counts.sum(axis=1)
    for i in range(n):
        if row[i] <= 1e-12:
            continue
        stay = counts[i, i] / row[i]
        others = np.arange(n) != i
        off = counts[i, others].sum()
        jump = np.zeros(n)
        jump[others] = counts[i, others] / off if off > 1e-12 else 1.0 / (n - 1)
        a[i] = (1.0 - stay) * jump
        a[i, i] = stay
    ok = s0 > 1e-8
    mu = np.where(ok, s1 / np.where(ok, s0, 1.0), previous.mu)
    var = np.where(ok, s2 / np.where(ok, s0, 1.0) - mu**2, previous.sigma**2)
    sigma = np.sqrt(np.maximum(var, _SIGMA_FLOOR**2))
    return Params(previous.pi0, a, mu, sigma)


@dataclass
class OnlineEM:
    """Cappe-Moulines online EM state: filter `alpha`, `rho (N, K)`, parameters and the step counter."""

    theta: Params
    kappa: float = 0.6
    burn_in: int = 500
    update: bool = True
    t: int = 0
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rho: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    stats: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loglik: float = 0.0

    def __post_init__(self) -> None:
        """Sizes."""
        n = self.theta.mu.shape[0]
        self.n = n
        self.k = n * n + 3 * n
        self.alpha = self.theta.pi0.copy()
        self.rho = np.zeros((n, self.k))
        self.stats = np.zeros(self.k)

    def _gamma(self) -> float:
        return 1.0 if self.t == 1 else float(self.t) ** (-self.kappa)

    def step(self, y: float) -> np.ndarray:
        """Process one observation; returns the current statistics `S_t`."""
        n, theta = self.n, self.theta
        self.t += 1
        gamma = self._gamma()
        emit = np.zeros((n, self.k))
        for j in range(n):  # emission part of s(., j, y)
            base = n * n
            emit[j, base + j] = 1.0
            emit[j, base + n + j] = y
            emit[j, base + 2 * n + j] = y * y
        if self.t == 1:
            pred = self.alpha
            self.rho = emit.copy()  # gamma_1 = 1, no transition at the first step
        else:
            joint = self.alpha[:, None] * theta.a  # (i, j)
            pred = joint.sum(axis=0)
            phi = joint / np.maximum(pred, 1e-300)[None, :]  # phi[i, j] = phi(i | j)
            new = (1.0 - gamma) * (phi.T @ self.rho) + gamma * emit
            for i in range(n):
                for j in range(n):
                    new[j, i * n + j] += gamma * phi[i, j]
            self.rho = new
        w = pred * np.exp(log_emission(theta, y))
        norm = float(w.sum())
        self.loglik += float(np.log(max(norm, 1e-300)))
        self.alpha = w / norm
        self.stats = self.alpha @ self.rho
        if self.update and self.t >= self.burn_in:
            self.theta = m_step(self.stats, n, self.theta)
        return self.stats


def offline_estats(theta: Params, y: np.ndarray) -> np.ndarray:
    """`sum_t E[s | y_1:T]` by a NumPy forward-backward pass (transitions of `t = 2..T`, every emission)."""
    t_len, n = y.shape[0], theta.mu.shape[0]
    b = np.exp(np.stack([log_emission(theta, v) for v in y]))  # (T, N)
    alpha = np.zeros((t_len, n))
    scale = np.zeros(t_len)
    a0 = theta.pi0 * b[0]
    scale[0] = a0.sum()
    alpha[0] = a0 / scale[0]
    for t in range(1, t_len):
        v = (alpha[t - 1] @ theta.a) * b[t]
        scale[t] = v.sum()
        alpha[t] = v / scale[t]
    beta = np.ones((t_len, n))
    for t in range(t_len - 2, -1, -1):
        beta[t] = theta.a @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
    gamma = alpha * beta
    out = np.zeros(n * n + 3 * n)
    for t in range(1, t_len):
        xi = alpha[t - 1][:, None] * theta.a * (b[t] * beta[t])[None, :] / scale[t]
        out[: n * n] += xi.reshape(-1)
    out[n * n : n * n + n] = gamma.sum(axis=0)
    out[n * n + n : n * n + 2 * n] = gamma.T @ y
    out[n * n + 2 * n :] = gamma.T @ (y * y)
    return out


def simulate_drift(
    t_len: int = 20_000, seed: int = 0, drift: float = 0.5, sojourn: float = 25.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A 2-regime series, regime-1 sigma growing linearly by `drift` (relative): `(y, regimes, sigma_1)`."""
    rng = np.random.default_rng(seed)
    sigma1 = 1.2 * (1.0 + drift * np.arange(t_len) / (t_len - 1))
    y, z = np.empty(t_len), np.empty(t_len, dtype=int)
    state = 0
    for t in range(t_len):
        z[t] = state
        y[t] = rng.normal((0.0, 1.5)[state], (0.5, sigma1[t])[state])
        if rng.random() < 1.0 / sojourn:
            state = 1 - state
    return y, z, sigma1


def track(y: np.ndarray, theta0: Params, kappa: float, burn_in: int = 500) -> tuple[np.ndarray, OnlineEM]:
    """Run the filter over `y`; returns the `(T, N)` path of `sigma` estimates and the final state."""
    online = OnlineEM(theta0.copy(), kappa=kappa, burn_in=burn_in)
    sig = np.empty((y.shape[0], theta0.mu.shape[0]))
    for t, v in enumerate(y):
        online.step(float(v))
        sig[t] = online.theta.sigma
    return sig, online


def initial_params() -> Params:
    """A deliberately rough starting point (means / scales off by ~30 %, stay probability 0.9)."""
    return Params(
        pi0=np.array([0.5, 0.5]),
        a=np.array([[0.9, 0.1], [0.1, 0.9]]),
        mu=np.array([0.2, 1.2]),
        sigma=np.array([0.7, 1.0]),
    )


if __name__ == "__main__":
    import time

    for kappa_ in (0.55, 0.6, 0.7, 0.8, 1.0):
        errors = []
        for seed_ in range(5):
            y_, _, s1 = simulate_drift(seed=seed_)
            path, _ = track(y_, initial_params(), kappa_)
            errors.append(float(np.mean(np.abs(path[2000:, 1] - s1[2000:]) / s1[2000:])))
        print(
            f"kappa={kappa_}: mean |sigma_1 error| after t=2000 over 5 seeds: {np.round(errors, 4).tolist()}"
        )
    y_, _, _ = simulate_drift()
    started = time.perf_counter()
    track(y_, initial_params(), 0.7)
    print(f"cost: {(time.perf_counter() - started) / y_.shape[0] * 1e6:.0f} microseconds per step")
