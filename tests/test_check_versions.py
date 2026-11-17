import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_versions.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_versions", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cv = _load()


def _write(root: Path, name: str, version: str) -> None:
    folder = root / "packages" / name
    folder.mkdir(parents=True)
    (folder / "pyproject.toml").write_text(
        f'[project]\nname = "agent-toolkit-{name}"\nversion = "{version}"\n'
    )


def test_identical_versions_pass(tmp_path: Path) -> None:
    _write(tmp_path, "a", "0.1.0")
    _write(tmp_path, "b", "0.1.0")
    assert cv.check(cv.package_versions(tmp_path), None) == []


def test_divergent_versions_fail(tmp_path: Path) -> None:
    _write(tmp_path, "a", "0.1.0")
    _write(tmp_path, "b", "0.1.1")
    (problem,) = cv.check(cv.package_versions(tmp_path), None)
    assert "agent-toolkit-a=0.1.0" in problem and "agent-toolkit-b=0.1.1" in problem


def test_matching_tag_passes_and_other_tag_fails(tmp_path: Path) -> None:
    _write(tmp_path, "a", "0.1.0a1")
    versions = cv.package_versions(tmp_path)
    assert cv.check(versions, "v0.1.0a1") == []
    assert cv.check(versions, "v0.1.0")
    assert cv.check(versions, "0.1.0a1")


def test_no_packages_fail(tmp_path: Path) -> None:
    assert cv.check(cv.package_versions(tmp_path), None)


def test_real_repo_versions_are_lockstep() -> None:
    assert cv.check(cv.package_versions(cv.ROOT), None) == []


def test_main_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    version = next(iter(cv.package_versions(cv.ROOT).values()))
    assert cv.main(["--tag", f"v{version}"]) == 0
    assert cv.main(["--tag", "v99.0.0"]) == 1
    assert "does not match" in capsys.readouterr().err
