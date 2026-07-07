"""Evaluate the SkyDiscover-compatible 26-circle packing contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

TARGET = 2.635
EXPECTED_CIRCLES = 26
TOLERANCE = 1e-6
CANDIDATE_TIMEOUT_SECONDS = 360
MAX_SOLUTION_CHARACTERS = 60_000
RESULT_PREFIX = "CIRCLE_PACKING_RESULT="


def validate_packing(centers: Any, radii: Any) -> tuple[bool, str]:
    """Apply the same shape and geometric checks as the upstream evaluator."""
    centers_array = np.asarray(centers)
    radii_array = np.asarray(radii)

    if centers_array.shape != (EXPECTED_CIRCLES, 2):
        return False, f"invalid centers shape {centers_array.shape}"
    if radii_array.shape != (EXPECTED_CIRCLES,):
        return False, f"invalid radii shape {radii_array.shape}"

    try:
        if np.isnan(centers_array).any() or np.isnan(radii_array).any():
            return False, "NaN values in output"
    except TypeError:
        return False, "centers and radii must be numeric"

    for index in range(EXPECTED_CIRCLES):
        if radii_array[index] < 0:
            return False, f"circle {index} has negative radius"

    for index in range(EXPECTED_CIRCLES):
        x, y = centers_array[index]
        radius = radii_array[index]
        if (
            x - radius < -TOLERANCE
            or x + radius > 1.0 + TOLERANCE
            or y - radius < -TOLERANCE
            or y + radius > 1.0 + TOLERANCE
        ):
            return False, f"circle {index} is outside the unit square"

    for left in range(EXPECTED_CIRCLES):
        for right in range(left + 1, EXPECTED_CIRCLES):
            distance = np.sqrt(
                np.sum((centers_array[left] - centers_array[right]) ** 2)
            )
            if distance < radii_array[left] + radii_array[right] - TOLERANCE:
                return False, f"circles {left} and {right} overlap"

    return True, ""


def run_candidate(path: Path) -> tuple[Any, Any, Any, float]:
    """Execute candidate code separately, as the upstream evaluator does."""
    try:
        solution_length = len(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise RuntimeError("candidate must be UTF-8 Python source") from exc
    if solution_length > MAX_SOLUTION_CHARACTERS:
        raise RuntimeError(
            "candidate exceeds upstream solution-length limit "
            f"({solution_length} > {MAX_SOLUTION_CHARACTERS} characters)"
        )

    runner = Path(__file__).with_name("run_candidate.py")
    start = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-I", str(runner), str(path.resolve())],
        cwd=Path(__file__).parent,
        text=True,
        capture_output=True,
        timeout=CANDIDATE_TIMEOUT_SECONDS,
        check=False,
    )
    elapsed = time.monotonic() - start

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"candidate exited with status {completed.returncode}: {detail}"
        )

    payload_line = next(
        (
            line[len(RESULT_PREFIX) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    if payload_line is None:
        raise RuntimeError("candidate did not return a result")

    payload = json.loads(payload_line)
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["centers"], payload["radii"], payload["reported_sum"], elapsed


def score_candidate(path: Path) -> dict[str, float | str]:
    try:
        centers, radii, reported_sum, elapsed = run_candidate(path)
        valid, reason = validate_packing(centers, radii)
        radii_array = np.asarray(radii)
        sum_radii = float(np.sum(radii_array)) if valid else 0.0

        warning = ""
        if valid:
            try:
                if abs(sum_radii - float(reported_sum)) > TOLERANCE:
                    warning = (
                        f"reported sum {reported_sum} does not match "
                        f"calculated sum {sum_radii}"
                    )
            except (TypeError, ValueError, OverflowError):
                warning = f"reported sum {reported_sum!r} is not numeric"

        score = sum_radii / TARGET if valid else 0.0
        return {
            "validity": 1.0 if valid else 0.0,
            "sum_radii": sum_radii,
            "target_ratio": score,
            "combined_score": score,
            "eval_time": elapsed,
            "error": reason,
            "warning": warning,
        }
    except subprocess.TimeoutExpired:
        return {
            "validity": 0.0,
            "sum_radii": 0.0,
            "target_ratio": 0.0,
            "combined_score": 0.0,
            "eval_time": float(CANDIDATE_TIMEOUT_SECONDS),
            "error": f"candidate timed out after {CANDIDATE_TIMEOUT_SECONDS}s",
            "warning": "",
        }
    except Exception as exc:
        return {
            "validity": 0.0,
            "sum_radii": 0.0,
            "target_ratio": 0.0,
            "combined_score": 0.0,
            "eval_time": 0.0,
            "error": str(exc),
            "warning": "",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", default="submission.py")
    args = parser.parse_args()

    result = score_candidate(Path(args.submission))
    if result["error"]:
        print(f"evaluation_error: {result['error']}", file=sys.stderr)
    if result["warning"]:
        print(f"evaluation_warning: {result['warning']}", file=sys.stderr)

    print(f"validity: {result['validity']:.0f}")
    print(f"circle_count: {EXPECTED_CIRCLES}")
    print(f"sum_radii: {result['sum_radii']:.12f}")
    print(f"target: {TARGET:.12f}")
    print(f"target_ratio: {result['target_ratio']:.12f}")
    print(f"combined_score: {result['combined_score']:.12f}")
    print(f"eval_time: {result['eval_time']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
