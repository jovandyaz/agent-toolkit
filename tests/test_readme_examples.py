"""Every runnable ```python block in the READMEs executes; ```python notest blocks are skipped.

A block marked `notest` needs a database, a real model or a framework graph, and says so in the
text around it. Blocks of one README share a namespace and run in order.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
READMES = sorted([ROOT / "README.md", *ROOT.glob("packages/*/README.md")])
_FENCE = re.compile(r"^```python([^\n]*)\n(.*?)^```$", re.DOTALL | re.MULTILINE)


def runnable_blocks(text: str) -> list[str]:
    return [body for info, body in _FENCE.findall(text) if info.strip() != "notest"]


def test_every_package_has_a_readme_with_a_runnable_example() -> None:
    packages = sorted(ROOT.glob("packages/*"))
    assert len(packages) == 8
    for package in packages:
        readme = package / "README.md"
        assert readme.is_file(), f"{package.name} has no README.md"
        assert runnable_blocks(readme.read_text()), f"{package.name} README has no runnable example"


@pytest.mark.parametrize("readme", READMES, ids=lambda p: str(p.relative_to(ROOT)))
def test_readme_examples_run(readme: Path) -> None:
    namespace: dict[str, object] = {"__name__": "__readme__"}
    for index, block in enumerate(runnable_blocks(readme.read_text())):
        code = compile(block, f"{readme.relative_to(ROOT)}[block {index}]", "exec")
        exec(code, namespace)
