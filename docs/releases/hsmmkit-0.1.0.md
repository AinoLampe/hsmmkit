# Release hsmmkit 0.1.0

Status: **release ready** (the maintainer creates the tag `v0.1.0`, the GitHub release and the TestPyPI upload).

`hsmmkit` 0.1.0 is the domain-neutral core of `regimekit` 0.2.0, extracted with its git history (E10-T08,
`tools/extract_hsmmkit.sh`) after the decoupling phase E10-T01…T07b.

- Package, distribution and console script `hsmmkit`; `HsmmkitError`, `HSMMKIT_TYPECHECK`, `HSMMKIT_RECOVERY_T`,
  bundle suffix `.hsmm` (option of `ModelStore`).
- Sources `numpy`, `pandas`, `csv`, `parquet`, `xarray`; clock `steps`; `gaussian` emission; registries and
  config extension; CLI app factory (`hsmmkit.cli.app.build_app`).
- The v0.2.0 finance golden values stay in `RegimeDetectionAndForecast`; the core keeps the reference and golden
  tests of Part I.
- The nightly `downstream` job runs the finance application against this repository; it is expected to be red
  until E10-T09 merges.
- Part I history: the source repository was squashed per ticket before the extraction, so `git log --follow`
  shows one commit per ticket (not the original E4-T03a/b split commits).
