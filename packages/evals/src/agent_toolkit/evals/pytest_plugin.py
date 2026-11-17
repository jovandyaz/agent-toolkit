"""pytest plugin (extra `pytest`): `--eval-n` option and `eval_n` fixture to run suites as tests.

Registered through the `pytest11` entry point. It only adds an option and a fixture, so it does
not change how other tests run. Never imported by the core modules.
"""

import pytest


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise ValueError("must be >= 1")
    return n


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("evals").addoption(
        "--eval-n",
        type=_positive_int,
        default=5,
        help="attempts per eval task (default: 5)",
    )


@pytest.fixture
def eval_n(request: pytest.FixtureRequest) -> int:
    """Number of attempts per task for `run_suite`, from `--eval-n`."""
    n: int = request.config.getoption("--eval-n")
    return n
