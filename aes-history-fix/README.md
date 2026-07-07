# Agentic Evolutionary Search

AES is a minimal advisor-worker loop for code optimization:

```text
evaluate starting point
  -> advisor proposes one direction
  -> worker edits the candidate
  -> evaluator measures it
  -> keep improvements, restore rejections
  -> repeat
```

The implementation is intentionally concentrated in
[`aes/evolve.py`](aes/evolve.py). Benchmark-specific prompts, starting points,
and evaluators live under [`benchmarks/`](benchmarks/).

## Setup

Requires Python 3.11+ and a headless agent command. The included configurations
use `claude -p`; edit the `command` arrays in a benchmark's `config.toml` to use
another agent.

Install [`uv`](https://docs.astral.sh/uv/), then create the project environment:

```bash
uv sync
```

## Starting Points

AES uses the current contents of each benchmark's `editable_files` as the
starting point. It evaluates those files before the first advisor call and
records the result as experiment `0000`.

### GPU MODE VectorAdd

The included starting kernel is
[`benchmarks/gpu-mode-vectoradd/workspace/submission.py`](benchmarks/gpu-mode-vectoradd/workspace/submission.py):

```python
import torch


def custom_kernel(data):
    a, b = data
    return torch.add(a, b)
```

### NanoGPT

NanoGPT uses the current `train.py` from the external workspace passed with
`--workspace`. This repository does not bundle an upstream training checkout;
AES edits its `train.py` during the run and restores rejected changes.

### Anthropic Performance Take-home

The Anthropic performance take-home benchmark is vendored in
[`benchmarks/anthropic-performance-takehome/workspace`](benchmarks/anthropic-performance-takehome/workspace).
AES edits only `perf_takehome.py`; evaluation checks the integrity of the
upstream tests and simulator before measuring simulated machine cycles.

## GPU MODE VectorAdd

Evaluation runs on a Modal H100.

```bash
uv sync --extra gpu-mode-vectoradd
uv run --extra gpu-mode-vectoradd modal setup
uv run --extra gpu-mode-vectoradd modal deploy benchmarks/gpu-mode-vectoradd/modal_eval.py
```

Validate the local configuration:

```bash
uv run --extra gpu-mode-vectoradd aes validate benchmarks/gpu-mode-vectoradd/config.toml
```

Run the search:

```bash
uv run --extra gpu-mode-vectoradd aes run benchmarks/gpu-mode-vectoradd/config.toml
```

Evaluate the current kernel without running the agent loop:

```bash
cd benchmarks/gpu-mode-vectoradd/workspace
uv run --extra gpu-mode-vectoradd python evaluate.py submission.py --output results.json
```

See [the benchmark README](benchmarks/gpu-mode-vectoradd/README.md) for details.

## NanoGPT

Provide a compatible Karpathy autoresearch/NanoGPT checkout containing
`train.py`. The default config expects `uv run train.py` to print:

```text
val_bpb: <number>
```

Validate:

```bash
uv run aes validate benchmarks/nanogpt/config.toml \
  --workspace /absolute/path/to/autoresearch
```

Run:

```bash
uv run aes run benchmarks/nanogpt/config.toml \
  --workspace /absolute/path/to/autoresearch
```

The current `train.py` is the starting point and the only editable file.

## Anthropic Performance Take-home

Validate:

```bash
uv run aes validate benchmarks/anthropic-performance-takehome/config.toml
```

Run:

```bash
uv run aes run benchmarks/anthropic-performance-takehome/config.toml
```

Evaluate the current candidate without running the agent loop:

```bash
cd benchmarks/anthropic-performance-takehome/workspace
uv run python evaluate.py
```

## Configuration

Each benchmark has one `config.toml` describing:

- advisor and worker commands and prompts
- number of iterations
- application workspace and editable files
- evaluation command
- metric regex and optimization direction
- improvement tolerance and timeouts

## Run Output

Runs are written to `runs/<timestamp>_<benchmark>/`:

```text
config.toml
events.jsonl
summary.md
best/                         # accepted editable files
experiments/
  0000/                       # starting point
  0001/
    advisor_state.json
    advisor.txt
    worker.txt
    candidate.patch
    candidate/
    evaluation.json
```

Every candidate is preserved, including rejected and failed attempts. Before
each iteration, a deterministic reducer gives the stateless advisor a bounded
state containing exact metrics, classified errors, and the eight most recent
experiments. The advisor can request a specific experiment, candidate diff, or
the current best candidate through bounded read-only tools. Raw history remains
an on-disk artifact and is never injected into the advisor conversation in full.

## Current Scope

This initial version is synchronous and uses one advisor and one worker. It does
not yet implement worker swarms, distributed scheduling, regime-local rewards,
or statistical confirmation.
