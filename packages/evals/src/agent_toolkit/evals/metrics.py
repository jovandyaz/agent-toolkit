"""pass@k and pass^k from n attempts of which c passed.

pass@k is the chance that at least one of k attempts passes; pass^k is the chance that all k
pass ("Demystifying evals for AI agents", Anthropic). Both are exact, computed from the counts
with `math.comb`, not estimated by resampling.
"""

from math import comb


def _validate(n: int, c: int, k: int) -> None:
    if not (n >= 1 and 0 <= c <= n and 1 <= k <= n):
        raise ValueError(f"require n >= 1, 0 <= c <= n and 1 <= k <= n; got n={n}, c={c}, k={k}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k attempts, drawn from n with c passing, passes."""
    _validate(n, c, k)
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Probability that all k attempts, drawn from n with c passing, pass (pass^k)."""
    _validate(n, c, k)
    return comb(c, k) / comb(n, k)
