from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluate import (
    MAX_SOLUTION_CHARACTERS,
    TARGET,
    score_candidate,
    validate_packing,
)

ROOT = Path(__file__).resolve().parent


class EvaluatorTests(unittest.TestCase):
    def test_upstream_seed_score(self) -> None:
        result = score_candidate(ROOT / "submission.py")
        self.assertEqual(result["validity"], 1.0)
        self.assertAlmostEqual(result["sum_radii"], 0.959764216996, places=12)
        self.assertAlmostEqual(
            result["combined_score"],
            0.959764216996 / TARGET,
            places=12,
        )

    def test_upstream_boundary_tolerance_is_preserved(self) -> None:
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        centers[0] = [-5e-7, 0.0]
        valid, _ = validate_packing(centers, radii)
        self.assertTrue(valid)

    def test_upstream_overlap_tolerance_is_preserved(self) -> None:
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        centers[0] = [0.1, 0.1]
        centers[1] = [0.2999995, 0.1]
        radii[0:2] = 0.1
        valid, _ = validate_packing(centers, radii)
        self.assertTrue(valid)

    def test_rejects_overlap_beyond_upstream_tolerance(self) -> None:
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        centers[0] = [0.1, 0.1]
        centers[1] = [0.299998, 0.1]
        radii[0:2] = 0.1
        valid, reason = validate_packing(centers, radii)
        self.assertFalse(valid)
        self.assertIn("overlap", reason)

    def test_rejects_wrong_shapes(self) -> None:
        valid, reason = validate_packing(np.zeros((25, 2)), np.zeros(25))
        self.assertFalse(valid)
        self.assertIn("shape", reason)

    def test_rejects_solution_over_upstream_length_limit(self) -> None:
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            candidate.write_text("#" * (MAX_SOLUTION_CHARACTERS + 1))
            result = score_candidate(candidate)

        self.assertEqual(result["validity"], 0.0)
        self.assertEqual(result["combined_score"], 0.0)
        self.assertIn("solution-length limit", result["error"])


if __name__ == "__main__":
    unittest.main()
