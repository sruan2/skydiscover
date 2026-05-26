"""Evaluator for MLA Decode — delegates to shared evaluator."""
import os
import sys

# Resolve the original benchmark directories so imports work even when this
# file is copied elsewhere (e.g. by ShinkaEvolve).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_problem_dir = _THIS_DIR
_parent_dir = os.path.dirname(_THIS_DIR)

# If shared_eval isn't next to this file, fall back to the canonical location.
if not os.path.isfile(os.path.join(_parent_dir, "shared_eval.py")):
    for candidate in [
        os.path.join(os.getcwd(), "benchmarks", "gpu_mode", "mla_decode"),
        _THIS_DIR,
    ]:
        parent = os.path.dirname(candidate)
        if os.path.isfile(os.path.join(parent, "shared_eval.py")):
            _problem_dir = candidate
            _parent_dir = parent
            break

if _problem_dir not in sys.path:
    sys.path.insert(0, _problem_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from shared_eval import evaluate, evaluate_stage1, evaluate_stage2
