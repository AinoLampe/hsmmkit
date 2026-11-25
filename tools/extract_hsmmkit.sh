#!/usr/bin/env bash
# E10-T08: extract `hsmmkit` from RegimeDetectionAndForecast with history (run once, on a fresh clone).
#
#   git clone --no-hardlinks <RegimeDetectionAndForecast> extract && cd extract
#   git checkout -B extract <E10-T07b merge commit>     # 42528543b9f0a4b7856e72c809016aee5b12a3e9
#   (delete every other branch, tag and remote of the clone)
#   uv run --with git-filter-repo bash tools/extract_hsmmkit.sh   # i.e. `git filter-repo` must be on the PATH
#
# Pass 1 keeps the paths of the new repository, pass 2 drops the finance paths (one run cannot mix include and
# exclude), pass 3 renames the source directory. The rename of the package contents is a separate commit.
set -euo pipefail

git filter-repo --force \
  --path src/regimekit --path tests --path docs --path benchmarks --path scripts --path noxfile.py \
  --path pyproject.toml --path .github --path mkdocs.yml --path CONTRIBUTING.md --path LICENSE \
  --path README.md --path .pre-commit-config.yaml --path .python-version --path CLAUDE.md --path .claude \
  --path spec --path OVERVIEW.md --path CHANGELOG.md --path changelog.d --path towncrier.toml \
  --path uv.lock --path .gitignore --path .gitattributes

git filter-repo --force --invert-paths \
  --path src/regimekit/finance --path tests/finance --path tests/fixtures/cassettes --path tests/live \
  --path tests/golden/v0_2_0 --path-glob 'docs/guide/finance*'

git filter-repo --force --path-rename src/regimekit/:src/hsmmkit/
