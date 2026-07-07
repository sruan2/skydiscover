"""Isolated protocol runner for a hexagon-packing candidate.

Executes the candidate in a separate, isolated interpreter (``python -I``) and
prints a single JSON payload line so the evaluator never imports candidate code
into its own process.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any

RESULT_PREFIX = "HEXAGON_PACKING_RESULT="


def json_value(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def main() -> int:
    path = Path(sys.argv[1]).resolve()
    try:
        spec = importlib.util.spec_from_file_location("hexagon_packing_submission", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import candidate: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inner_hex_data, outer_hex_data, outer_hex_side_length = module.run_packing()
        payload = {
            "inner_hex_data": json_value(inner_hex_data),
            "outer_hex_data": json_value(outer_hex_data),
            "outer_hex_side_length": json_value(outer_hex_side_length),
        }
    except BaseException as exc:  # noqa: BLE001 - report any candidate failure
        payload = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    print(f"{RESULT_PREFIX}{json.dumps(payload, allow_nan=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
