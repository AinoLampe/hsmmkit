"""Pure NumPy reference for the Gaussian GARCH(1,1) log-likelihood (E9-T03; no `hsmmkit` import).

`eps_t = y_t - m * dt_t`, `h_1 = omega / (1 - alpha - beta)`, `h_{t+1} = omega + alpha eps_t^2 + beta h_t`,
`y_t ~ N(m dt_t, h_t)`; the log-likelihood is the sum of the normal log-densities. This is the model the
`arch` package fits (constant mean, GARCH(1,1), normal errors) with the recursion started at the
unconditional variance; `arch` is not a dependency, so the formula is implemented directly.
"""

from __future__ import annotations

import numpy as np


def garch11_loglik(y: np.ndarray, dt: np.ndarray, m: float, omega: float, alpha: float, beta: float) -> float:
    """Sum of `log N(y_t; m dt_t, h_t)` under the GARCH(1,1) variance recursion."""
    h = omega / (1.0 - alpha - beta)
    total = 0.0
    for t in range(y.shape[0]):
        eps = y[t] - m * dt[t]
        total += -0.5 * (np.log(2.0 * np.pi * h) + eps**2 / h)
        h = omega + alpha * eps**2 + beta * h
    return float(total)


def simulate_garch11(
    n: int, dt: float, m: float, omega: float, alpha: float, beta: float, seed: int
) -> np.ndarray:
    """A synthetic GARCH(1,1) return series `(n,)`."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    h = omega / (1.0 - alpha - beta)
    for t in range(n):
        eps = np.sqrt(h) * rng.standard_normal()
        y[t] = m * dt + eps
        h = omega + alpha * eps**2 + beta * h
    return y
