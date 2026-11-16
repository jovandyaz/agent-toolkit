#!/bin/sh
# Type-check every workspace package, then the root tests. Run: uv run scripts/typecheck.sh
#
# One mypy run per package, because a single run cannot give every file a unique module name:
# package directories have hyphens and test files repeat basenames across packages
# (tests/test_core.py). MYPYPATH holds every package's src, so the agent_toolkit namespace
# resolves across packages, plus the tests of the package being checked. Each run keeps its
# own cache so same-named test modules of different packages never share an entry.
set -eu
cd "$(dirname "$0")/.."

src_path=""
for src in packages/*/src; do
    if [ -d "$src" ]; then src_path="$src_path$src:"; fi
done

for package in packages/*/; do
    [ -d "$package" ] || continue
    name=$(basename "$package")
    echo "mypy: $name"
    MYPYPATH="$src_path${package}tests" \
        mypy --cache-dir ".mypy_cache/packages/$name" "${package}src" "${package}tests"
done

echo "mypy: root tests"
mypy tests
