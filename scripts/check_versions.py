"""Fail unless every package shares one version, and (with --tag) the tag matches it.

All packages release in lockstep: one PEP 440 version, one tag `v{version}`.
Usage: `python scripts/check_versions.py [--tag vX.Y.Z]`.
"""

import argparse
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def package_versions(root: Path) -> dict[str, str]:
    """`[project] version` of every `packages/*/pyproject.toml`, keyed by package name."""
    versions: dict[str, str] = {}
    for pyproject in sorted(root.glob("packages/*/pyproject.toml")):
        project = tomllib.loads(pyproject.read_text())["project"]
        versions[project["name"]] = project["version"]
    return versions


def check(versions: dict[str, str], tag: str | None) -> list[str]:
    """Problems found: an empty list means the versions are consistent."""
    if not versions:
        return ["no packages found under packages/"]
    problems: list[str] = []
    distinct = set(versions.values())
    if len(distinct) > 1:
        listing = ", ".join(f"{name}={version}" for name, version in sorted(versions.items()))
        problems.append(f"packages do not share one version: {listing}")
    if tag is not None:
        if not tag.startswith("v"):
            problems.append(f"tag {tag!r} must look like v<version>")
        elif {tag.removeprefix("v")} != distinct:
            problems.append(
                f"tag {tag!r} does not match package version(s) {', '.join(sorted(distinct))}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag to match, for example v0.1.0")
    args = parser.parse_args(argv)
    problems = check(package_versions(ROOT), args.tag)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
