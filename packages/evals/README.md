# agent-toolkit-evals

A minimal eval runner: tasks graded by deterministic functions over the final state, k isolated
attempts per task, and exact pass@k and pass^k.

**Why.** Anthropic's "Demystifying evals for AI agents": grade the final state, prefer
deterministic graders, report pass^k (all k attempts pass) next to pass@k (at least one passes),
build suites of 20 to 50 tasks from real failures, and read the transcripts.

```
pip install "agent-toolkit-evals[pytest]"   # the extra adds the --eval-n option and fixture
```

## Example

```python
from agent_toolkit.evals import EvalTask, Grade, pass_at_k, pass_hat_k, run_suite


def grade(final_state: dict[str, int]) -> Grade:
    ok = final_state["charges"] == 1
    return Grade(
        passed=ok, reason="one charge issued" if ok else f"{final_state['charges']} charges"
    )


task = EvalTask(
    id="charge-within-mandate", input={"amount": 20_000}, tags=frozenset({"mandate"}), grade=grade
)


def attempt(task: EvalTask[dict[str, int]], i: int) -> dict[str, int]:
    # A real attempt builds a fresh environment, runs the agent and returns the final state.
    return {"charges": 1 if i != 3 else 2}


report = run_suite([task], attempt, n=4)
result = report.results[0]
assert (result.c, result.n) == (3, 4)
assert result.pass_at[1] == pass_at_k(4, 3, 1) == 0.75
assert result.pass_hat[4] == pass_hat_k(4, 3, 4) == 0.0
print(report.summary())
```

## Contracts

- `run_suite(tasks, attempt, n, transcripts_dir=None, ks=())` runs each task `n` times and reports
  pass@k and pass^k for k in {1, n, *ks}, computed exactly with `math.comb`. An exception in the
  attempt or the grader fails that attempt without aborting the suite.
- With `transcripts_dir`, one JSONL file per task gets a line per attempt (the final state is
  serialized with `repr` where it is not JSON). Read them.
- The pytest plugin only adds `--eval-n` (default 5) and the `eval_n` fixture.

## Caller duties

- `attempt(task, i)` must build a fresh environment (database, fakes, clock) per call and return
  the final state. The runner never reuses state between attempts.
- Keep transcripts out of version control when final states hold customer data.

## Limits

Sequential only (no parallel attempts), no dashboard, no model-graded scoring. The suites
themselves live in the application.
