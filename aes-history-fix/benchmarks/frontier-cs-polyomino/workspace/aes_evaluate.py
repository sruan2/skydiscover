"""AES evaluation wrapper for Frontier-CS problem 0 (Pack the Polyominoes).

Bridges AES's "run a command, parse a metric from stdout" contract to the
SkyDiscover Frontier-CS evaluator, which compiles the C++ submission and judges
it against the local Docker judge server (default http://localhost:8081).

Prints ``combined_score: <x>`` for AES to parse (direction = maximize).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# SkyDiscover Frontier-CS checkout (provides the evaluator + frontier_cs pkg).
FRONTIER_EVAL_DIR = Path(
    "/Users/sherryruan/github/skydiscover/benchmarks/frontier-cs-eval"
)
os.environ.setdefault("FRONTIER_CS_SRC", str(FRONTIER_EVAL_DIR / "Frontier-CS" / "src"))
os.environ.setdefault("FRONTIER_CS_PROBLEM", "0")
os.environ.setdefault("JUDGE_URLS", "http://localhost:8081")
sys.path.insert(0, str(FRONTIER_EVAL_DIR))

from evaluator import evaluate  # noqa: E402  (path set above)


def main() -> int:
    submission = sys.argv[1] if len(sys.argv) > 1 else "submission.cpp"
    submission_path = str(Path(submission).resolve())

    result = evaluate(submission_path, "0")

    score = result.get("combined_score", 0.0)
    unbounded = result.get("score_unbounded", score)
    status = result.get("status", "unknown")
    message = str(result.get("message", ""))[:300]

    if status != "success":
        print(f"evaluation_error: {status}: {message}", file=sys.stderr)

    print(f"status: {status}")
    print(f"score_unbounded: {unbounded}")
    print(f"combined_score: {float(score):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
