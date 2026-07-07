from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

LOCKED_FILES = {
    "Readme.md": "398566c82976fb1cd72b4d63d718936e34ca98d122f6ffff6093915331dba0d6",
    "problem.py": "fadb0f0858e2259f5759077a5544b9906dad3ceee80d37b4f0aa77da730c93c9",
    "tests/frozen_problem.py": "fadb0f0858e2259f5759077a5544b9906dad3ceee80d37b4f0aa77da730c93c9",
    "tests/submission_tests.py": "11c57cc999da93acb41201191073cd657ddffa87635359b3157c6e177c18ea0a",
}

FORBIDDEN_FILES = {
    "trace.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_integrity() -> list[str]:
    failures = []
    for relative, expected in LOCKED_FILES.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"{relative}: missing")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"{relative}: sha256 {actual} != {expected}")

    for relative in FORBIDDEN_FILES:
        if (ROOT / relative).exists():
            failures.append(f"{relative}: generated artifact is present")

    return failures


def main() -> int:
    failures = check_integrity()
    if failures:
        print(json.dumps({"success": False, "integrity_failures": failures}, indent=2))
        return 2

    completed = subprocess.run(
        [sys.executable, "tests/submission_tests.py", "CorrectnessTests"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)

    if completed.returncode != 0:
        return completed.returncode

    matches = re.findall(r"CYCLES:\s*([0-9]+(?:\.[0-9]+)?)", completed.stdout)
    if not matches:
        print("metric_cycles: unavailable")
        return 3

    metric = float(matches[-1])
    print(f"metric_cycles: {metric:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
