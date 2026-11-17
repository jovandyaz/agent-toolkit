import pytest


def test_plugin_adds_eval_n_option_default_5(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_default(eval_n):
            assert eval_n == 5
        """
    )
    pytester.runpytest().assert_outcomes(passed=1)


def test_eval_n_override(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_n(eval_n):
            assert eval_n == 3
        """
    )
    result = pytester.runpytest("--eval-n=3")
    result.assert_outcomes(passed=1)


def test_eval_n_rejects_nonpositive(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--eval-n=0")
    assert result.ret != 0
