"""Sequential eval runner: k attempts per task, deterministic grading of the final state.

Isolation is the caller's job: `attempt(task, i)` must build a fresh environment (database,
fakes, clock) for every call and return the final state; the runner never reuses state between
attempts. Running attempts in parallel is out of scope.
"""

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent_toolkit.evals.metrics import pass_at_k, pass_hat_k


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class Grade(_Frozen):
    """Verdict of a grader on one final state."""

    passed: bool
    reason: str


@dataclass(frozen=True)
class EvalTask[FinalStateT]:
    """One task: its input, tags and a deterministic grader of the attempt's final state."""

    id: str
    input: Mapping[str, object]
    tags: frozenset[str]
    grade: Callable[[FinalStateT], Grade]


class TaskResult(_Frozen):
    task_id: str
    n: int
    c: int
    pass_at: dict[int, float]
    pass_hat: dict[int, float]
    failures: tuple[Grade, ...]


class SuiteReport(_Frozen):
    results: tuple[TaskResult, ...]

    def summary(self) -> str:
        """Short human-readable table, one row per task plus the mean pass^n."""
        rows = [("task", "c/n", "pass@1", "pass^n")]
        for r in self.results:
            rows.append(
                (r.task_id, f"{r.c}/{r.n}", f"{r.pass_at[1]:.3f}", f"{r.pass_hat[r.n]:.3f}")
            )
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        lines = [
            "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in rows
        ]
        if self.results:
            mean = sum(r.pass_hat[r.n] for r in self.results) / len(self.results)
            lines.append(
                f"mean pass^n: {mean:.3f}"
                if len({r.n for r in self.results}) > 1
                else f"mean pass^{self.results[0].n}: {mean:.3f}"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        """Deterministic JSON (sorted keys)."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)


def _failed(exc: Exception) -> Grade:
    return Grade(passed=False, reason=f"error: {type(exc).__name__}: {exc}")


def _run_attempt[S](
    task: EvalTask[S], attempt: Callable[[EvalTask[S], int], S], i: int
) -> tuple[Grade, object]:
    try:
        state = attempt(task, i)
    except Exception as exc:
        return _failed(exc), None
    try:
        return task.grade(state), state
    except Exception as exc:
        return _failed(exc), state


def _serialize(record: Mapping[str, object]) -> str:
    try:
        return json.dumps(record, default=repr)
    except Exception:  # circular references, non-str keys, failing __repr__ of nested values
        safe = {**record, "final_state": _safe_repr(record["final_state"])}
        return json.dumps(safe, default=repr)


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except Exception as exc:
        return f"<unrepresentable {type(value).__name__}: {type(exc).__name__}>"


def _transcript_path(directory: Path, task_id: str) -> Path:
    return directory / (re.sub(r"[^A-Za-z0-9._-]", "_", task_id) + ".jsonl")


def run_suite[S](
    tasks: Sequence[EvalTask[S]],
    attempt: Callable[[EvalTask[S], int], S],
    n: int,
    transcripts_dir: Path | None = None,
    ks: Iterable[int] = (),
) -> SuiteReport:
    """Run every task n times and report pass@k and pass^k for k in {1, n, *ks}.

    An exception in `attempt` or in the grader fails that attempt (reason `error: <Type>:
    <message>`) without aborting the suite; `KeyboardInterrupt` and `SystemExit` propagate.
    With `transcripts_dir`, one JSONL file per task gets a line per attempt.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    k_values = sorted({1, n, *ks})
    if transcripts_dir is not None:
        transcripts_dir.mkdir(parents=True, exist_ok=True)
    results: list[TaskResult] = []
    for task in tasks:
        grades: list[Grade] = []
        lines: list[str] = []
        for i in range(n):
            grade, state = _run_attempt(task, attempt, i)
            grades.append(grade)
            lines.append(
                _serialize(
                    {
                        "task_id": task.id,
                        "attempt": i,
                        "passed": grade.passed,
                        "reason": grade.reason,
                        "final_state": state,
                    }
                )
            )
        if transcripts_dir is not None:
            _transcript_path(transcripts_dir, task.id).write_text("\n".join(lines) + "\n")
        c = sum(g.passed for g in grades)
        results.append(
            TaskResult(
                task_id=task.id,
                n=n,
                c=c,
                pass_at={k: pass_at_k(n, c, k) for k in k_values},
                pass_hat={k: pass_hat_k(n, c, k) for k in k_values},
                failures=tuple(g for g in grades if not g.passed),
            )
        )
    return SuiteReport(results=tuple(results))
