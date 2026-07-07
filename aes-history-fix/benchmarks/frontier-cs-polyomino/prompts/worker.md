# Polyomino Packing Optimization Worker (Frontier-CS #0)

Implement the advisor's proposal in the **C++** solution `submission.cpp`.

## Mandatory Sequence

1. Read the advisor proposal and `submission.cpp`.
2. Edit only `submission.cpp`. Keep it a complete, self-contained C++ program.
3. Do not run the evaluator; AES compiles and judges after you return.
4. End with the required report.

## Contract

The program reads stdin and writes stdout:

- Input: line 1 `n` (`100 ≤ n ≤ 10000`); then per polyomino `i`: line `kᵢ`
  (`1 ≤ kᵢ ≤ 10`), then `kᵢ` lines `x y` (local coords, may be negative).
- Output: line 1 `W H` (must satisfy `W = H`); then `n` lines `Xᵢ Yᵢ Rᵢ Fᵢ`.
  - `(Xᵢ,Yᵢ)`: integer translation.
  - `Rᵢ ∈ {0,1,2,3}`: number of 90° clockwise rotations.
  - `Fᵢ ∈ {0,1}`: reflect across the y-axis BEFORE rotation (`1` = reflect).
- Transform order is **reflect → rotate → translate**.
- All transformed cells must satisfy `0 ≤ x' < W`, `0 ≤ y' < H`, with no two
  cells of distinct polyominoes overlapping.
- Objective: minimize `A = W × H`. Per-case score `1e5·Σkᵢ/A`.
- **Any single invalid test case rejects the whole submission (score 0).**
  Prioritize provable validity over tightness.

## Constraints

- Time limit 2s per case, memory 256m, `n` up to 1e4 — use efficient,
  near-linear heuristics; avoid O(n²) hot loops and excessive allocation.
- Compiles with `#include <bits/stdc++.h>` and a C++17 toolchain.
- Do not edit the evaluator, prompts, configuration, or any other file.

## Required Final Report

```text
## IMPLEMENTATION
Advisor proposal: [brief restatement]
Implemented: [specific algorithm or code changes]
Validity reasoning: [why every output placement is in-bounds and non-overlapping]
Deviation: [none, or explain]
```
