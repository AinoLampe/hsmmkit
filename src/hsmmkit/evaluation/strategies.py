"""Regime-conditioned trading rules, proportional costs and the strategy trace (E7-T03).

Concept refs: §3.1 FR-E03, §5.5 `Strategy`, Appendix A `evaluation.strategy`.

Timing. A position is *decided* at a row `t` from information up to `t` (the filtered regime probabilities
of the origin, the forecast made there) and *held* over the next step. With the realized log-returns `y`,
the strategy return of step `t` is (log-return approximation, costs subtracted in log space)

    r_t = pos_{t-1} . y_t - cost * sum_m |pos_{t-1,m} - pos_{t-2,m}|,    pos_{-1} = pos_{-2} = 0,

with `cost = cost_bps * 1e-4`. The equity curve is `exp(cumsum(r))`. Only filtered probabilities are
admissible in a backtest: asking for `smoothed` warns (`check_backtest_admissible`) and the backtest uses the
filtered probabilities of the origin, which is all that a decision at that origin can know.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
import pandas as pd

from regimekit.utils.errors import ConfigError, NumericalError

if TYPE_CHECKING:
    from regimekit.config.schema import StrategyCfg
    from regimekit.inference.results import ForecastResult, PosteriorResult
    from regimekit.utils.typing import Array

BPS = 1e-4  # one basis point as a fraction


@runtime_checkable
class Strategy(Protocol):
    """A map from the state of knowledge at an origin to positions in `[-1, 1]`."""

    @property
    def uses_smoothed(self) -> bool:
        """`True` when the rule asked for smoothed probabilities (illustration only, never admissible)."""
        ...

    def positions(self, posterior: PosteriorResult, forecast: ForecastResult | None) -> Array:
        """`(T, M)` positions: row `t` is decided with the information of row `t` and held over step `t+1`."""
        ...

    def decide(self, filtered: Array, forecast: ForecastResult | None) -> Array:
        """`(M,)` position decided at one origin from its `(N,)` regime probabilities and forecast."""
        ...


@dataclass(frozen=True)
class RegimeLongFlat:
    """Long (`1`) unless `P(regime) > prob_above`, then flat (`0`); the same position in every series."""

    regime: int
    prob_above: float
    n_series: int = 1
    use: str = "filtered"

    @property
    def uses_smoothed(self) -> bool:
        """Whether the config asked for the (look-ahead) smoothed probabilities."""
        return self.use == "smoothed"

    def _check(self, n_regimes: int) -> None:
        if not 0 <= self.regime < n_regimes:
            raise ConfigError(
                f"flat_if.regime={self.regime} is outside 0..{n_regimes - 1}",
                path="evaluation.strategy.flat_if.regime",
            )

    def positions(self, posterior: PosteriorResult, forecast: ForecastResult | None) -> Array:
        """`(T, M)` rule applied to every row of the posterior's `use` probabilities."""
        del forecast
        probs = np.asarray(posterior.smoothed if self.uses_smoothed else posterior.filtered)
        self._check(probs.shape[1])
        flat = probs[:, self.regime] > self.prob_above
        return jnp.asarray(np.tile(np.where(flat, 0.0, 1.0)[:, None], (1, self.n_series)))

    def decide(self, filtered: Array, forecast: ForecastResult | None) -> Array:
        """Position for the step after an origin with regime probabilities `filtered (N,)`."""
        del forecast
        probs = np.asarray(filtered)
        self._check(probs.shape[0])
        return jnp.full((self.n_series,), 0.0 if probs[self.regime] > self.prob_above else 1.0)


@dataclass(frozen=True)
class RegimeTargetVol:
    """Volatility targeting from the one-step predictive volatility of the forecast.

    `sigma_pred = sqrt(var[0] / nominal_dt)` (annualized under the trading/calendar clocks, per step under
    `steps`), magnitude `min(cap, target_vol / sigma_pred)`, sign from `mean[0]`: long for `mean >= 0`,
    otherwise flat, or short (same magnitude) with `allow_short`.
    """

    target_vol: float
    nominal_dt: float
    cap: float = 1.0
    allow_short: bool = False

    @property
    def uses_smoothed(self) -> bool:
        """Volatility targeting reads the forecast only."""
        return False

    def decide(self, filtered: Array, forecast: ForecastResult | None) -> Array:
        """`(M,)` position from `forecast.var[0]` / `forecast.mean[0]` (regime probabilities unused)."""
        del filtered
        if forecast is None:
            raise ConfigError("regime_target_vol needs a forecast", path="evaluation.strategy.type")
        var = np.asarray(forecast.var, dtype=np.float64)[0]
        mean = np.asarray(forecast.mean, dtype=np.float64)[0]
        if not (np.all(np.isfinite(var)) and np.all(np.isfinite(mean))):
            raise NumericalError("non-finite forecast mean/variance in regime_target_vol", path="forecast")
        sigma = np.sqrt(var / self.nominal_dt)
        size = np.minimum(self.cap, self.target_vol / np.maximum(sigma, np.finfo(np.float64).tiny))
        sign = np.where(mean >= 0.0, 1.0, -1.0 if self.allow_short else 0.0)
        return jnp.asarray(size * sign)

    def positions(self, posterior: PosteriorResult, forecast: ForecastResult | None) -> Array:
        """`(T, M)` positions where only the last row (the forecast's origin) is defined; the others are 0."""
        pos = np.zeros((posterior.times.shape[0], np.asarray(forecast.var).shape[1] if forecast else 1))
        last = np.asarray(self.decide(posterior.filtered[-1], forecast))
        pos[-1] = last
        return jnp.asarray(pos)


def make_strategy(cfg: StrategyCfg, nominal_dt: float, n_series: int = 1) -> Strategy:
    """The configured rule.

    Raises:
        ConfigError: a missing or inconsistent `flat_if` / `target_vol` / `cap`, with the config path.
    """
    path = "evaluation.strategy"
    if not 0.0 < cfg.cap <= 1.0:
        raise ConfigError(f"cap must be in (0, 1], got {cfg.cap}", path=f"{path}.cap")
    if cfg.type == "regime_long_flat":
        flat = cfg.flat_if
        if flat is None or "regime" not in flat or "prob_above" not in flat:
            raise ConfigError(
                "regime_long_flat needs flat_if.regime and flat_if.prob_above", path=f"{path}.flat_if"
            )
        use = str(flat.get("use", "filtered"))
        if use not in ("filtered", "smoothed"):
            raise ConfigError(
                f"flat_if.use must be filtered or smoothed, got {use!r}", path=f"{path}.flat_if.use"
            )
        prob_above = float(flat["prob_above"])
        if not 0.0 <= prob_above <= 1.0:
            raise ConfigError("flat_if.prob_above must lie in [0, 1]", path=f"{path}.flat_if.prob_above")
        return RegimeLongFlat(int(flat["regime"]), prob_above, n_series, use)
    if cfg.target_vol is None or cfg.target_vol <= 0.0:
        raise ConfigError("regime_target_vol needs target_vol > 0", path=f"{path}.target_vol")
    return RegimeTargetVol(cfg.target_vol, nominal_dt, cfg.cap, cfg.allow_short)


def check_backtest_admissible(strategy: Strategy) -> None:
    """Warn when the rule asks for smoothed probabilities, which a backtest decision cannot have."""
    if strategy.uses_smoothed:
        warnings.warn(
            "evaluation.strategy.flat_if.use='smoothed' looks ahead; a backtest decides with the filtered "
            "probabilities of each origin instead",
            UserWarning,
            stacklevel=3,
        )


def strategy_returns(held: np.ndarray, y: np.ndarray, cost_bps: float) -> tuple[np.ndarray, np.ndarray]:
    """`(returns (n,), turnover (n,))` of positions `held (n, M)` against log-returns `y (n, M)`.

    `held[i]` is the position held over step `i` (decided one step earlier); the position before the first
    row is 0, so entering costs. `turnover[i] = sum_m |held[i, m] - held[i-1, m]|` and the return is
    `held[i] . y[i] - cost_bps * 1e-4 * turnover[i]`.
    """
    held, y = np.asarray(held, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if held.shape != y.shape or held.ndim != 2:
        raise ConfigError(
            f"held {held.shape} and y {y.shape} must be equal (n, M) arrays", path="evaluation.strategy"
        )
    previous = np.vstack([np.zeros((1, held.shape[1])), held[:-1]])
    turnover = np.abs(held - previous).sum(axis=1)
    return (held * y).sum(axis=1) - cost_bps * BPS * turnover, turnover


def strategy_trace(
    times: np.ndarray,
    held: np.ndarray,
    y: np.ndarray,
    dt: np.ndarray,
    cost_bps: float,
    fold: np.ndarray | None = None,
) -> pd.DataFrame:
    """The strategy table: `fold, time, position, return, gross, equity, turnover, dt`.

    `position` is the position held over the step ending at `time` (`position_<m>` columns when `M > 1`);
    `gross = position . y` (before costs), `equity = exp(cumsum(return))`, continuous across folds.
    `dt` is the step's clock length (needed to annualize the metrics).
    """
    ret, turnover = strategy_returns(held, y, cost_bps)
    held = np.asarray(held, dtype=np.float64)
    frame: dict[str, object] = {
        "fold": np.zeros(ret.shape[0], dtype=np.int64) if fold is None else np.asarray(fold, dtype=np.int64),
        "time": np.asarray(times).astype("datetime64[ns]"),
    }
    if held.shape[1] == 1:
        frame["position"] = held[:, 0]
    else:
        frame.update({f"position_{m}": held[:, m] for m in range(held.shape[1])})
    frame.update(
        {
            "return": ret,
            "gross": (held * np.asarray(y, dtype=np.float64)).sum(axis=1),
            "equity": np.exp(np.cumsum(ret)),
            "turnover": turnover,
            "dt": np.asarray(dt, dtype=np.float64),
        }
    )
    return pd.DataFrame(frame)
