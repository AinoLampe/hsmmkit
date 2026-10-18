# Invalid config fixtures

`profile_core.yaml` is a standalone example of an invalid config (kept as a real file since
`profile: core` is the most load-bearing single-field rejection, AC5). The rest of the validation
rules (per-state length mismatches, covariate cross-references, mode-specific required fields,
quantile ordering, unknown keys at various nesting levels, ...) are exercised in
`tests/unit/test_e1_t02a_schema.py` by mutating a deep copy of `../appendix_a.yaml` in Python
rather than maintaining a full near-duplicate YAML file per rule — the mutations are one-line,
directly readable at the assertion site, and avoid ~15 largely-identical multi-page YAML fixtures
drifting out of sync with Appendix A over time.
