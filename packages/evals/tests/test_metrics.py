import pytest

from agent_toolkit.evals import pass_at_k, pass_hat_k


def test_pass_at_k_all_correct() -> None:
    assert pass_at_k(5, 5, 3) == 1.0


def test_pass_at_k_none_correct() -> None:
    assert pass_at_k(5, 0, 1) == 0.0


def test_pass_at_k_partial() -> None:
    assert pass_at_k(5, 2, 2) == pytest.approx(1 - 3 / 10)


def test_pass_hat_k_needs_every_attempt() -> None:
    assert pass_hat_k(5, 4, 5) == 0.0
    assert pass_hat_k(5, 5, 5) == 1.0


def test_pass_hat_k_one_is_success_rate() -> None:
    assert pass_hat_k(5, 4, 1) == 0.8


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [(5, 3, 6), (5, 3, 0), (5, -1, 1), (5, 6, 1), (0, 0, 1), (-1, 0, 1)],
)
def test_invalid_arguments_raise(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError, match="require"):
        pass_at_k(n, c, k)
    with pytest.raises(ValueError, match="require"):
        pass_hat_k(n, c, k)
