# Polyomino Packing Optimization Advisor (Frontier-CS #0)

Improve a **C++** program that solves "Pack the Polyominoes (Reflections
Allowed)". The editable program is `submission.cpp`; it must read from stdin and
write to stdout, and is judged by the Frontier-CS Docker judge over 70 test
cases. The score (maximize) is normalized against a strong reference solution.

## Problem

Given `n` polyominoes (`100 ≤ n ≤ 10000`), each with `kᵢ` cells (`1 ≤ kᵢ ≤ 10`),
place ALL of them — allowing **reflection (optional) → rotation by 0/90/180/270°
→ integer translation, in that order** — into a single axis-aligned **square**
of side `W = H`, with:

- every transformed cell on integer coordinates `0 ≤ x' < W`, `0 ≤ y' < H`,
- no two cells of distinct polyominoes overlapping,
- objective: **minimize area `A = W × H`** (ties: smaller `H`, then smaller `W`).

Per-case score = `1e5 · Σkᵢ / A`. **Any invalid case rejects the entire
submission (score 0)**, so validity is paramount — a correct loose packing beats
a tight invalid one.

## Input / Output

Input: line 1 `n`; then per polyomino: line `kᵢ`, then `kᵢ` lines `x y` (local
frame, may be negative). Output: line 1 `W H`; then `n` lines `Xᵢ Yᵢ Rᵢ Fᵢ`
where `(Xᵢ,Yᵢ)` is the translation, `Rᵢ∈{0,1,2,3}` is 90°-CW rotations, and
`Fᵢ∈{0,1}` reflects across the y-axis before rotation.

## Guidance

- Time limit 2s/case, memory 256m. `n` up to 1e4 demands near-linear heuristics.
- Strong baselines: shelf / skyline / bottom-left-fill packing of oriented
  bounding boxes, then binary-search or grow the square side `W=H`.
- Exploit rotations/reflections to pick each piece's tightest orientation.
- Small pieces (k≤2) are flexible filler for leftover gaps.
- Correctness first: guarantee no overlap and in-bounds before optimizing area.

Read `submission.cpp` and the full experiment history. Propose ONE coherent
algorithmic or implementation improvement. Do not edit files or run evaluations.

## Output Format

```text
## STATE
[Current score, validity, and the limiting behavior.]

## RATIONALE
[Why the proposed change should raise the judged score.]

## PROPOSAL
[One concrete implementation proposal for the worker.]
```
