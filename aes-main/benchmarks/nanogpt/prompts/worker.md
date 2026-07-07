# NanoGPT Worker

Implement one targeted change to `train.py` based on the advisor plan. Preserve
the existing training entry point and its output contract, including a line:

```text
val_bpb: <number>
```

Do not modify data preparation, evaluation code, dependency files, or the
metric. Do not install packages or write checkpoints. The AES engine runs the
training command and restores `train.py` when the candidate is rejected.
