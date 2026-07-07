# Circle Packing Optimization Advisor

Improve a constructor that directly produces a specific arrangement of exactly
26 circles in a unit square. Maximize the sum of their radii. The AlphaEvolve
paper achieved `2.635` for `n=26`.

The editable program is `submission.py` and must define:

```python
run_packing() -> (centers, radii, sum_radii)
```

The evaluator requires centers of shape `(26, 2)`, radii of shape `(26,)`,
nonnegative radii, containment in the unit square, and no overlap. It uses the
same `1e-6` geometric tolerance and `sum_of_radii / 2.635` score as the pinned
SkyDiscover evaluator.

Key geometric insights:

- Circle packings often follow hexagonal patterns in their densest regions.
- Maximum infinite-plane density is `pi/(2*sqrt(3))`.
- Edge effects make square-container packing harder.
- Circles can be placed in layers or shells when confined to a square.
- Similar-radius circles often form regular patterns, while varied radii can
  use remaining space.
- Perfect symmetry need not be optimal.

Read `submission.py`, the complete experiment history, and the protected files
under `reference/`. Propose one coherent code or geometric improvement. Candidate
code may use NumPy and SciPy and must complete within 360 seconds. Focus on an
explicit constructor that places circles in specific positions rather than
making runtime iterative search the primary approach. The complete candidate
must remain at or below 60,000 characters.

Do not edit files or run evaluations.

## Output Format

```text
## STATE
[Current quality and limiting behavior.]

## RATIONALE
[Why the proposed change should improve the measured sum.]

## PROPOSAL
[One concrete implementation proposal for the worker.]
```
