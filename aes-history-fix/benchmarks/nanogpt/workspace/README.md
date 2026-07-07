# Provide A NanoGPT Workspace

This directory is intentionally not a fake training implementation. Point AES
at a real Karpathy autoresearch/nanoGPT checkout containing `train.py`:

```bash
uv run aes run benchmarks/nanogpt/config.toml \
  --workspace /absolute/path/to/autoresearch
```

The default adapter expects:

- `train.py` is the only editable file.
- `uv run train.py` performs one fixed-budget experiment.
- Successful output contains `val_bpb: <number>`.

Adjust `evaluation_command`, `metric_pattern`, environment variables, and
timeouts in `config.toml` to match the checkout.
