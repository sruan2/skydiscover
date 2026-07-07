# Math Circle Packing

AES adaptation of SkyDiscover's 26-circle packing benchmark, pinned to upstream
revision `4734f324c4c3909eb3c8a4bc72a4ca34e6e679ed`.

The task is to return exactly 26 non-overlapping circles inside the unit square
while maximizing the sum of their radii. The normalized metric is:

```text
combined_score = sum_of_radii / 2.635
```

## Compatibility

This adaptation aligns the benchmark-level conditions used by SkyDiscover:

- the same ring-based initial constructor;
- executable `run_packing()` candidate programs;
- NumPy and SciPy availability;
- exactly 26 centers and radii;
- the same `1e-6` boundary and overlap tolerance;
- the same target and normalized score;
- the effective upstream 360-second evaluator timeout;
- the upstream 60,000-character solution limit;
- Claude Code with `claude-sonnet-4-6` in both profiles;
- the same geometric reference material under `workspace/reference/`.

Invalid or failed candidates receive score zero, matching upstream evaluation
semantics. Candidate code runs in a separate process.

The search frameworks are still structurally different: AES uses an
advisor-worker pair per candidate, while SkyDiscover performs one generation
step per candidate. For a resource-normalized framework comparison, report
model calls, tokens, wall time, and number of evaluated candidates in addition
to the final score.

## Comparison profiles

Two profiles make the resource budget explicit. `config.toml` runs 50 AES
iterations: 100 top-level advisor/worker invocations and 50 candidates.

| Profile | AES iterations | Agent invocations | Candidates |
|---|---:|---:|---:|
| `config.toml` | 50 | 100 | 50 |
| `config.candidate-normalized.toml` | 100 | 200 | 100 |

Claude Code may make multiple internal model turns, so report provider tokens
or cost, wall time, successful generations, evaluations, and final score.
Iteration count alone is not a fair resource comparison.

The Claude Code commands can inspect reference files. Run SkyDiscover with `--agentic`
for matching tool/reference access and disable human feedback for unattended
comparisons. Also set `evaluator.cascade_evaluation: false`; otherwise
SkyDiscover executes qualifying candidates twice while AES evaluates once.
Match the model snapshot, reasoning settings, provider, hardware,
Python and dependency versions, and number of independent trials.

## Run

```bash
uv sync --extra math-circle-packing
uv run aes validate benchmarks/math-circle-packing/config.toml
uv run aes run benchmarks/math-circle-packing/config.toml

uv run aes validate \
  benchmarks/math-circle-packing/config.candidate-normalized.toml
uv run aes run benchmarks/math-circle-packing/config.candidate-normalized.toml

cd benchmarks/math-circle-packing/workspace
uv run --extra math-circle-packing python evaluate.py submission.py
```

The evaluator recomputes the radius sum and does not trust the candidate's
reported sum.

Reset `workspace/submission.py` to the pinned seed before each independent run.
