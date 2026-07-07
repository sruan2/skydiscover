# Anthropic Performance Take-home

AES benchmark wrapper for
<https://github.com/anthropics/original_performance_takehome>.

The upstream task is vendored under `workspace/`. AES edits only
`workspace/perf_takehome.py`. The local `workspace/evaluate.py` wrapper keeps
evaluation integrity by checking hashes for the upstream README, simulator, and
test files before running `tests/submission_tests.py`.

## Validate

```bash
uv run aes validate benchmarks/anthropic-performance-takehome/config.toml
```

## Run Search

```bash
uv run aes run benchmarks/anthropic-performance-takehome/config.toml
```

## Evaluate Current Candidate

```bash
cd benchmarks/anthropic-performance-takehome/workspace
uv run python evaluate.py
```

The metric line consumed by AES is:

```text
metric_cycles: <number>
```
