# Evaluation

!!! note
    Stub page (E0-T04). Filled in by E7-T01…E7-T04 (metrics, walk-forward, strategies); completed in E8-T04.

Walk-forward evaluation compares realized outcomes against forecast distributions fold by fold, e.g. the
pinball (quantile) loss at quantile `\tau`:

$$
\mathcal{L}_\tau(y, \hat{q}_\tau) = \max\big(\tau (y - \hat{q}_\tau),\, (\tau - 1)(y - \hat{q}_\tau)\big)
$$
