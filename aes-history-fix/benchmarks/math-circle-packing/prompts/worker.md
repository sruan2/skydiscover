# Circle Packing Optimization Worker

Implement the advisor's proposal in the executable circle-packing constructor.

## Mandatory Sequence

1. Read the advisor proposal, `submission.py`, and relevant protected files
   under `reference/`.
2. Edit only `submission.py`.
3. Preserve `run_packing() -> (centers, radii, sum_radii)`.
4. Do not run the evaluator; AES evaluates after you return.
5. End with the required report.

## Contract

- Return exactly 26 center pairs and 26 radii.
- Radii must be nonnegative.
- Every circle must lie inside the unit square.
- Circles may touch but may not overlap.
- The evaluator permits the upstream `1e-6` numerical tolerance.
- Maximize `sum(radii) / 2.635`.
- NumPy and SciPy are available.
- Candidate execution is limited to 360 seconds.
- The complete `submission.py` must not exceed 60,000 characters.

You may replace the seed algorithm completely. Focus on an explicit constructor
that places circles in specific positions rather than making runtime iterative
search the primary approach. Do not edit the evaluator, prompts, references,
configuration, or dependency files.

## Required Final Report

```text
## IMPLEMENTATION
Advisor proposal: [brief restatement]
Implemented: [specific algorithm or geometric changes]
Constraint reasoning: [why the returned packing should be valid]
Deviation: [none, or explain]
```
