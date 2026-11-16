"""Every non-adapter module must import with frameworks, drivers and providers blocked.

Complements the import-linter contract, which cannot see core -> adapter -> framework chains
once adapter imports are ignored.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("langgraph", "langchain", "langchain_core", "psycopg")

_PROBE = """
import importlib, importlib.abc, sys
forbidden = set(sys.argv[2].split(","))
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in forbidden:
            raise ImportError(f"core module imported forbidden package {name!r}")
        return None
sys.meta_path.insert(0, Block())
importlib.import_module(sys.argv[1])
"""


def _core_modules() -> list[str]:
    modules = []
    for path in sorted(ROOT.glob("packages/*/src/agent_toolkit/**/*.py")):
        src = path.relative_to(ROOT).parts.index("src")
        parts = list(path.relative_to(ROOT).with_suffix("").parts[src + 1 :])
        if "adapters" in parts or parts[-1] in {"testing", "pytest_plugin"}:
            continue
        if parts[-1] == "__init__":
            parts.pop()
        modules.append(".".join(parts))
    return modules


_MODULES: list[object] = [*_core_modules()] or [
    pytest.param("", marks=pytest.mark.skip(reason="no packages yet"))
]


@pytest.mark.parametrize("module", _MODULES)
def test_core_module_imports_without_frameworks(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, module, ",".join(FORBIDDEN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
