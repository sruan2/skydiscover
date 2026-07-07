# Hexagon Packing Optimization Worker

Implement the advisor's proposal in the executable hexagon-packing constructor.

## Mandatory Sequence

1. Read the advisor proposal and `submission.py`.
2. Edit only `submission.py`.
3. Preserve `run_packing() -> (inner_hex_data, outer_hex_data, outer_hex_side_length)`.
4. Do not run the evaluator; AES evaluates after you return.
5. End with the required report.

## Contract

- `inner_hex_data` must have shape `(11, 3)`: rows of `(x, y, angle_degrees)`.
- `outer_hex_data` must have shape `(3,)`: `(x, y, angle_degrees)`.
- `outer_hex_side_length` must be a positive float.
- All 11 inner hexagons are unit regular hexagons (side length 1).
- Inner hexagons must be pairwise disjoint (they may touch but not overlap).
- All inner hexagons must be fully contained in the outer hexagon.
- The evaluator permits the upstream `1e-6` numerical tolerance.
- Maximize `(1/outer_hex_side_length) / 0.2544`. Invalid packings score `0`.
- NumPy and SciPy are available.
- Candidate execution is limited to 600 seconds.
- The complete `submission.py` must not exceed 60,000 characters.

You may replace the seed algorithm completely. Prefer an explicit constructor
that places hexagons at specific positions/orientations over making runtime
iterative search the primary approach. Do not edit the evaluator, prompts,
configuration, or dependency files.

## Required Final Report

```text
## IMPLEMENTATION
Advisor proposal: [brief restatement]
Implemented: [specific algorithm or geometric changes]
Constraint reasoning: [why the returned packing should be valid]
Deviation: [none, or explain]
```
