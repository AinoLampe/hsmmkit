# Guide

`hsmmkit` is a **domain-neutral** library for hidden semi-Markov models. Observations are vectors `y_t` on a grid
with step sizes `Δ_t` (a *clock*), optionally with covariates; nothing in the core knows prices, tickers,
providers or trading calendars. Data enters through *sources*, time through *clocks*, and applications extend
the library by registering components (emissions, durations, metrics, sources, clocks, transforms, inits) and by
subclassing config sections.

`regimekit` (market-regime detection and forecasting) is an example extension built on top of `hsmmkit`.

Start with [Install](install.md) and the [Quickstart](quickstart.md).
