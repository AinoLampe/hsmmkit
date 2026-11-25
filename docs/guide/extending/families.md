# Emission families and the family contract

An emission family implements the `EmissionFamily` protocol (`hsmmkit.model.emission.base`). The required
methods are `param_specs`, `log_prob`, `sample`, `init_carry` and `step_carry`. The optional estimation hooks
are implemented only when the family has them; the `BaseEmission` mixin supplies the `None`-returning
defaults, and estimators discover capabilities by calling a hook and checking for `None`, never by family
name:

| Hook | Meaning |
|---|---|
| `sufficient_stats(y, dt, gamma, z)` | weighted sufficient statistics for regime weights `gamma (T, N)` |
| `m_step(stats)` | closed-form maximizer of the expected complete-data log-likelihood |
| `conjugate_prior()` | the conjugate prior family of the parameters |
| `expected_log_prob(y, dt, q_theta, z)` | `E_q[log f_i(y_t)]` for variational Bayes |
| `moments(regime, dt, theta, z)` | analytic `(mean, variance)`, each `(T, M)` |

## The contract suite

`hsmmkit.testing.family_contract` is one importable pytest suite that every family must pass. A package
registers a `FamilyCase` and imports the test function and the parametrization hook into a test module:

```python
from hsmmkit.testing import family_contract
from hsmmkit.testing.family_contract import (  # noqa: F401
    FamilyCase,
    pytest_generate_tests,
    test_family_contract,
)

family_contract.register_case(
    FamilyCase(
        name="my-family",
        factory=MyFamily,
        theta_factory=lambda key, n, m, k: {...},  # parameters of n regimes, m series, k covariates
        data_factory=lambda key, t, m: (y, dt, z),  # valid for the family's support
        scipy_logpdf=my_reference_density,  # optional (T, N) reference
    )
)
```

`pytest` then yields `test_family_contract[<case>-<check>]` for these checks:

1. `log_prob` is `(T, N)`, finite, and equals `scipy_logpdf` within 1e-10.
2. `sample` is support-valid; its mean and variance are within 3 SE of `moments()` (`n = 20 000` per regime).
3. `dt` scaling: with `scale_by_dt` the moments are proportional to `dt`; otherwise `log_prob` ignores `dt`.
4. The preparer rejects data outside the family's support.
5. Covariate links: zero coefficients reproduce the covariate-free density bitwise; non-zero ones change it
   (needs `covariate_factory`).
6. `ParamManager` round trip `theta = theta(u(theta))` within 1e-12, unique names, jit and gradient finite.
7. Optional hooks: `m_step` is a stationary point of `sum_t sum_i gamma_ti log f_i(y_t)`; a conjugate prior
   provides `update`.
8. The density evaluates under every clock in `case.clocks` (default `steps` and `calendar`; an extension adds
   its own, for example `trading`) without branching on clock names.

Checks that do not apply to a family skip. The checks are also plain functions (`family_contract.CHECKS`)
that raise `AssertionError` with a readable message, so a failing family can be debugged without pytest.
The tolerances are those of the specification's appendices (`LOGPDF_ATOL`, `ROUNDTRIP_ATOL`, `MOMENT_SE`);
a case may override one only with a justification in its definition.

The built-in cases (`gaussian` diagonal with and without the GBM options, Student-t, the product emission)
live in `tests/contract/test_e11_t01_builtin_families.py`.
