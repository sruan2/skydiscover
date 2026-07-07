# Hexagon Packing Optimization Advisor

Improve a constructor that directly produces a specific arrangement of exactly
11 **unit regular hexagons** (side length 1) packed inside a larger **regular
hexagon**. Maximize `1/outer_hex_side_length` (equivalently, minimize the outer
hexagon's side length). The current state-of-the-art benchmark is
`1/3.930092 ~= 0.2544`.

The editable program is `submission.py` and must define:

```python
run_packing() -> (inner_hex_data, outer_hex_data, outer_hex_side_length)
```

where:

- `inner_hex_data` has shape `(11, 3)`; each row is `(x, y, angle_degrees)` for
  one unit inner hexagon.
- `outer_hex_data` has shape `(3,)`: `(x, y, angle_degrees)` of the outer hexagon.
- `outer_hex_side_length` is a positive float.

The evaluator (Separating Axis Theorem) requires: all 11 inner hexagons are unit
regular hexagons, pairwise disjoint, and fully contained in the outer hexagon.
It uses the upstream `1e-6` geometric tolerance. The normalized score is
`(1/outer_hex_side_length) / 0.2544`. Invalid packings score `0`.

Key geometric insights:

- Hexagons tile the plane perfectly; the densest arrangements exploit this.
- Rotating inner hexagons (the `angle_degrees` field) and/or the outer hexagon
  often lets neighbors interlock more tightly than an axis-aligned grid.
- Edge and corner effects dominate at small counts like `n=11`; the outer
  boundary, not interior density, usually limits the score.
- Mixed orientations and off-lattice shifts can shrink the bounding outer hexagon.
- Perfect symmetry need not be optimal.

Read `submission.py` and the complete experiment history. Propose one coherent
code or geometric improvement. Candidate code may use NumPy and SciPy and must
complete within 600 seconds. Prefer an explicit constructor that places hexagons
at specific positions/orientations over making runtime iterative search the
primary approach. The complete candidate must remain at or below 60,000
characters.

Do not edit files or run evaluations.

## Output Format

```text
## STATE
[Current quality and limiting behavior.]

## RATIONALE
[Why the proposed change should reduce the outer side length.]

## PROPOSAL
[One concrete implementation proposal for the worker.]
```
