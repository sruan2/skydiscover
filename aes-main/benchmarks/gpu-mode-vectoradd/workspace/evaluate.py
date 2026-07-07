"""Submit the GPU MODE VectorAdd candidate to the deployed Modal H100 evaluator."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import modal


APP_NAME = "aes-gpu-mode-vectoradd-eval"
FUNCTION_NAME = "evaluate_kernel"
MODAL_TIMEOUT_SECONDS = 360


def format_result(result: dict) -> str:
    lines = [
        f"modal_status: {'success' if result.get('success') else 'failure'}",
        f"gpu: {result.get('gpu_name', 'unknown')}",
        f"platform: {result.get('platform', 'modal')}",
        f"torch: {result.get('torch_version', 'unknown')}",
        f"correctness: {result.get('tests_passed', 0)}/{result.get('tests_total', 0)}",
    ]

    for test in result.get("test_details", []):
        status = "pass" if test.get("passed") else "fail"
        lines.append(
            f"test N={test.get('N')} seed={test.get('seed')} status={status}"
            + (f" error={test['error']}" if test.get("error") else "")
        )

    for benchmark in result.get("benchmark_details", []):
        lines.append(
            f"N={benchmark['N']} seed={benchmark['seed']} "
            f"mean_us={benchmark['mean_us']} err_us={benchmark['err_us']} "
            f"runs={benchmark['runs']}"
        )

    if result.get("benchmark"):
        lines.append(f"metric_us: {result['benchmark']['geomean_us']}")
        lines.append(f"score: {result['benchmark']['score']}")
    if result.get("failure_stage"):
        lines.append(f"failure_stage: {result['failure_stage']}")
    if result.get("error"):
        lines.append(f"error: {result['error']}")
    return "\n".join(lines)


def call_modal(kernel_code: str, mode: str) -> dict:
    function = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    result: list[str | None] = [None]
    error: list[Exception | None] = [None]

    def invoke() -> None:
        try:
            result[0] = function.remote(kernel_code, mode=mode)
        except Exception as exc:  # Modal exceptions vary by client version.
            error[0] = exc

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(timeout=MODAL_TIMEOUT_SECONDS)
    if thread.is_alive():
        raise TimeoutError(f"Modal call timed out after {MODAL_TIMEOUT_SECONDS}s")
    if error[0] is not None:
        raise RuntimeError(f"Modal call failed: {error[0]}")
    if result[0] is None:
        raise RuntimeError("Modal call returned no result")
    return json.loads(result[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate GPU MODE VectorAdd on a Modal H100")
    parser.add_argument("submission", nargs="?", default="submission.py")
    parser.add_argument("--output", "-o", default="results.json")
    parser.add_argument("--mode", choices=("test", "leaderboard"), default="leaderboard")
    args = parser.parse_args()

    submission = Path(args.submission)
    if not submission.is_file():
        print(f"submission not found: {submission}", file=sys.stderr)
        return 1

    try:
        result = call_modal(submission.read_text(), args.mode)
    except (json.JSONDecodeError, RuntimeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(format_result(result))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
