# GPU MODE VectorAdd Benchmark

This application optimizes `workspace/submission.py` for float16 vector
addition. Evaluation runs remotely on a Modal H100, checks correctness, and
minimizes geometric mean latency across four matrix sizes.

The search starts from the checked-in `workspace/submission.py`, which currently
implements vector addition with `torch.add`. AES evaluates it as experiment
`0000` before asking the advisor for the first proposal.

Requirements:

- Python 3.11+
- A Modal account with H100 access
- An agent command compatible with the configured `claude -p` invocation

Sync the benchmark dependency and authenticate Modal:

```bash
uv sync --extra gpu-mode-vectoradd
uv run --extra gpu-mode-vectoradd modal setup
```

Deploy the evaluator once from the repository root:

```bash
uv run --extra gpu-mode-vectoradd modal deploy benchmarks/gpu-mode-vectoradd/modal_eval.py
```

The deployment is named `aes-gpu-mode-vectoradd-eval`. The local
`workspace/evaluate.py` client looks up its `evaluate_kernel` function, sends
the current `submission.py` source, and prints a metric that AES records.

Run the search:

```bash
uv run --extra gpu-mode-vectoradd aes run benchmarks/gpu-mode-vectoradd/config.toml
```

Evaluate one candidate without starting AES:

```bash
cd benchmarks/gpu-mode-vectoradd/workspace
uv run --extra gpu-mode-vectoradd python evaluate.py submission.py --output results.json
```

Change the `[advisor].command` and `[worker].command` arrays to use another
headless coding agent. AES sends each prompt on standard input and uses standard
output as the agent's plan or report. The included Claude Code worker uses
`--permission-mode acceptEdits` so it can update `submission.py` during a
non-interactive run. Agent output and run status are streamed to the terminal
and also saved under the run's `experiments/` directory.

The role prompts are adapted from
[`isaac9000/vectoradd-advisor`](https://github.com/isaac9000/vectoradd-advisor).
They preserve its task context, comparison guidance, structured advisor output,
and worker implementation contract. Evaluation, result logging, and candidate
restoration are handled by AES rather than delegated to the worker.
