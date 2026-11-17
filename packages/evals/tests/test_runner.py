import json
from pathlib import Path

import pytest

from agent_toolkit.evals import EvalTask, Grade, SuiteReport, run_suite


def _grade_ok(state: int) -> Grade:
    return Grade(passed=state % 2 == 0, reason=f"state={state}")


def _task(task_id: str = "even") -> EvalTask[int]:
    return EvalTask(id=task_id, input={"q": 1}, tags=frozenset({"t"}), grade=_grade_ok)


def _attempt(task: EvalTask[int], i: int) -> int:
    return i


def test_counts_passes_on_even_attempts() -> None:
    report = run_suite([_task()], _attempt, n=5)
    (result,) = report.results
    assert (result.task_id, result.n, result.c) == ("even", 5, 3)
    assert result.pass_at[1] == pytest.approx(0.6)
    assert result.pass_hat[5] == 0.0
    assert result.pass_hat[1] == pytest.approx(0.6)
    assert [g.reason for g in result.failures] == ["state=1", "state=3"]


def test_extra_ks_are_reported() -> None:
    (result,) = run_suite([_task()], _attempt, n=5, ks=[2]).results
    assert sorted(result.pass_at) == [1, 2, 5]
    assert sorted(result.pass_hat) == [1, 2, 5]


def test_invalid_n_and_ks_raise() -> None:
    with pytest.raises(ValueError, match="n must be"):
        run_suite([_task()], _attempt, n=0)
    with pytest.raises(ValueError, match="require"):
        run_suite([_task()], _attempt, n=2, ks=[3])


def test_attempt_exception_fails_the_attempt_not_the_suite() -> None:
    def attempt(task: EvalTask[int], i: int) -> int:
        if task.id == "boom" and i == 1:
            raise RuntimeError("env down")
        return 0

    report = run_suite([_task("boom"), _task("fine")], attempt, n=3)
    boom, fine = report.results
    assert (boom.c, fine.c) == (2, 3)
    assert boom.failures == (Grade(passed=False, reason="error: RuntimeError: env down"),)


def test_grade_exception_fails_the_attempt() -> None:
    def bad_grade(state: int) -> Grade:
        raise KeyError("missing")

    task = EvalTask(id="g", input={}, tags=frozenset(), grade=bad_grade)
    (result,) = run_suite([task], _attempt, n=2).results
    assert result.c == 0
    assert result.failures[0].reason == "error: KeyError: 'missing'"


def test_keyboard_interrupt_propagates() -> None:
    def attempt(task: EvalTask[int], i: int) -> int:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_suite([_task()], attempt, n=1)


def test_transcripts_one_jsonl_per_task(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "t"
    run_suite([_task("a/b")], _attempt, n=3, transcripts_dir=out)
    (path,) = out.iterdir()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["attempt"] for line in lines] == [0, 1, 2]
    assert lines[0] == {
        "task_id": "a/b",
        "attempt": 0,
        "passed": True,
        "reason": "state=0",
        "final_state": 0,
    }
    assert lines[1]["passed"] is False


def test_transcripts_survive_unserializable_state(tmp_path: Path) -> None:
    class Odd:
        def __repr__(self) -> str:
            return "<Odd>"

    circular: list[object] = []
    circular.append(circular)

    def attempt(task: EvalTask[object], i: int) -> object:
        return Odd() if i == 0 else circular

    task: EvalTask[object] = EvalTask(
        id="odd", input={}, tags=frozenset(), grade=lambda s: Grade(passed=True, reason="ok")
    )
    report = run_suite([task], attempt, n=2, transcripts_dir=tmp_path)
    assert report.results[0].c == 2
    lines = [json.loads(x) for x in (tmp_path / "odd.jsonl").read_text().splitlines()]
    assert lines[0]["final_state"] == "<Odd>"
    assert isinstance(lines[1]["final_state"], str)


def test_errored_attempt_transcript_has_null_state(tmp_path: Path) -> None:
    def attempt(task: EvalTask[int], i: int) -> int:
        raise ValueError("x")

    run_suite([_task("e")], attempt, n=1, transcripts_dir=tmp_path)
    (line,) = (tmp_path / "e.jsonl").read_text().splitlines()
    assert json.loads(line)["final_state"] is None


def test_summary_and_json() -> None:
    report = run_suite([_task("even")], _attempt, n=5)
    text = report.summary()
    assert "even" in text
    assert "3/5" in text
    assert "0.600" in text
    assert "mean pass^5" in text
    assert report.to_json() == report.to_json()
    data = json.loads(report.to_json())
    assert data["results"][0]["c"] == 3
    assert report.to_json().index('"c"') < report.to_json().index('"task_id"')
    assert SuiteReport(results=()).summary()
