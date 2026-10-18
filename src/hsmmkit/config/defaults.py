"""Numeric defaults referenced by `config/schema.py` cross-field validation."""

D_MAX_CAP = 500
"""Hard cap on the default `model.d_max` (overview §5.7)."""

D_MAX_SOJOURN_MULTIPLIER = 10
"""`d_max` defaults to `min(D_MAX_CAP, D_MAX_SOJOURN_MULTIPLIER * expected_sojourn_steps)`."""

D_MAX_WARN_SOJOURN_MULTIPLIER = 2
"""Warn when `d_max < D_MAX_WARN_SOJOURN_MULTIPLIER * max(expected_sojourn_steps)`."""

MAX_ORDER_WARN = 2
"""Warn when `transition.order > MAX_ORDER_WARN`."""
