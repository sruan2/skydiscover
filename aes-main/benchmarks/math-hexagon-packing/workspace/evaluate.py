"""Evaluate the SkyDiscover-compatible 11-hexagon packing contract.

Pack 11 unit regular hexagons inside a regular hexagon, maximizing
``1/outer_hex_side_length``. The geometric checks (Separating Axis Theorem
disjointness and vertex containment) and the normalized score reproduce the
pinned upstream SkyDiscover/AlphaEvolve evaluator semantics so AES and
SkyDiscover scores are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

N_HEX = 11
BENCHMARK = 1.0 / 3.930092  # current SOTA inv_outer_hex_side_length ~= 0.2544
TOLERANCE = 1e-6
CANDIDATE_TIMEOUT_SECONDS = 600
MAX_SOLUTION_CHARACTERS = 60_000
RESULT_PREFIX = "HEXAGON_PACKING_RESULT="


# ---------------------------------------------------------------------------
# Upstream geometry (adapted from google-deepmind/alphaevolve_results, Apache-2.0)
# ---------------------------------------------------------------------------


def hexagon_vertices(cx: float, cy: float, side: float, angle_degrees: float) -> list[tuple[float, float]]:
    vertices = []
    angle_radians = math.radians(angle_degrees)
    for i in range(6):
        angle = angle_radians + 2 * math.pi * i / 6
        vertices.append((cx + side * math.cos(angle), cy + side * math.sin(angle)))
    return vertices


def _normalize(v: tuple[float, float]) -> tuple[float, float]:
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2)
    return (v[0] / mag, v[1] / mag) if mag != 0 else (0.0, 0.0)


def _normals(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normals = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normals.append(_normalize((-edge[1], edge[0])))
    return normals


def _project(vertices: list[tuple[float, float]], axis: tuple[float, float]) -> tuple[float, float]:
    projections = [vx * axis[0] + vy * axis[1] for vx, vy in vertices]
    return min(projections), max(projections)


def _overlap_1d(min1: float, max1: float, min2: float, max2: float, tol: float = TOLERANCE) -> bool:
    return max1 >= min2 - tol and max2 >= min1 - tol


def _polygons_intersect(v1: list, v2: list, tol: float = TOLERANCE) -> bool:
    for axis in _normals(v1) + _normals(v2):
        min1, max1 = _project(v1, axis)
        min2, max2 = _project(v2, axis)
        if not _overlap_1d(min1, max1, min2, max2, tol):
            return False
    return True


def _hexagons_disjoint(hex1, hex2, tol: float = TOLERANCE) -> bool:
    return not _polygons_intersect(hexagon_vertices(*hex1), hexagon_vertices(*hex2), tol)


def _point_inside_hexagon(point, hex_params, tol: float = TOLERANCE) -> bool:
    verts = hexagon_vertices(*hex_params)
    for i in range(len(verts)):
        p1 = verts[i]
        p2 = verts[(i + 1) % len(verts)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        pv = (point[0] - p1[0], point[1] - p1[1])
        cross = edge[0] * pv[1] - edge[1] * pv[0]
        if cross < -tol:
            return False
    return True


def validate_packing(inner_hex_data: Any, outer_hex_data: Any, outer_side: Any) -> tuple[bool, str]:
    inner = np.asarray(inner_hex_data, dtype=float)
    outer = np.asarray(outer_hex_data, dtype=float)

    if inner.shape != (N_HEX, 3):
        return False, f"invalid inner_hex_data shape {inner.shape}, expected {(N_HEX, 3)}"
    if outer.shape != (3,):
        return False, f"invalid outer_hex_data shape {outer.shape}, expected {(3,)}"
    try:
        side = float(outer_side)
    except (TypeError, ValueError):
        return False, "outer_hex_side_length is not numeric"
    if not math.isfinite(side) or side <= 0:
        return False, f"invalid outer_hex_side_length {outer_side}"
    if np.isnan(inner).any() or np.isnan(outer).any():
        return False, "NaN values in output"

    inner_params = [(x, y, 1.0, angle) for x, y, angle in inner]
    outer_params = (outer[0], outer[1], side, outer[2])

    for i in range(N_HEX):
        for j in range(i + 1, N_HEX):
            if not _hexagons_disjoint(inner_params[i], inner_params[j]):
                return False, f"inner hexagons {i + 1} and {j + 1} intersect"

    for idx, params in enumerate(inner_params):
        for vertex in hexagon_vertices(*params):
            if not _point_inside_hexagon(vertex, outer_params):
                return False, f"inner hexagon {idx + 1} is not contained in the outer hexagon"

    return True, ""


# ---------------------------------------------------------------------------
# Candidate execution
# ---------------------------------------------------------------------------


def run_candidate(path: Path) -> tuple[Any, Any, Any, float]:
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
        raise RuntimeError(f"candidate exited with status {completed.returncode}: {detail}")

    payload_line = next(
        (
            line[len(RESULT_PREFIX):]
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
    return (
        payload["inner_hex_data"],
        payload["outer_hex_data"],
        payload["outer_hex_side_length"],
        elapsed,
    )


def score_candidate(path: Path) -> dict[str, float | str]:
    try:
        inner, outer, outer_side, elapsed = run_candidate(path)
        valid, reason = validate_packing(inner, outer, outer_side)
        if valid:
            inv_side = 1.0 / float(outer_side)
            combined = inv_side / BENCHMARK
        else:
            inv_side = 0.0
            combined = 0.0
        return {
            "validity": 1.0 if valid else 0.0,
            "outer_hex_side_length": float(outer_side) if valid else 0.0,
            "inv_outer_hex_side_length": inv_side,
            "combined_score": combined,
            "eval_time": elapsed,
            "error": reason,
        }
    except subprocess.TimeoutExpired:
        return {
            "validity": 0.0,
            "outer_hex_side_length": 0.0,
            "inv_outer_hex_side_length": 0.0,
            "combined_score": 0.0,
            "eval_time": float(CANDIDATE_TIMEOUT_SECONDS),
            "error": f"candidate timed out after {CANDIDATE_TIMEOUT_SECONDS}s",
        }
    except Exception as exc:  # noqa: BLE001 - any failure scores zero, like upstream
        return {
            "validity": 0.0,
            "outer_hex_side_length": 0.0,
            "inv_outer_hex_side_length": 0.0,
            "combined_score": 0.0,
            "eval_time": 0.0,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", default="submission.py")
    args = parser.parse_args()

    result = score_candidate(Path(args.submission))
    if result["error"]:
        print(f"evaluation_error: {result['error']}", file=sys.stderr)

    print(f"validity: {result['validity']:.0f}")
    print(f"hex_count: {N_HEX}")
    print(f"outer_hex_side_length: {result['outer_hex_side_length']:.12f}")
    print(f"benchmark: {BENCHMARK:.12f}")
    print(f"inv_outer_hex_side_length: {result['inv_outer_hex_side_length']:.12f}")
    print(f"combined_score: {result['combined_score']:.12f}")
    print(f"eval_time: {result['eval_time']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
