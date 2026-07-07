# EVOLVE-BLOCK-START
"""Pack 11 unit regular hexagons inside a regular hexagon of minimal side length.

Returns the SkyDiscover-compatible contract:
    inner_hex_data : np.ndarray (11, 3)  rows of (x, y, angle_degrees)
    outer_hex_data : np.ndarray (3,)     (x, y, angle_degrees)
    outer_hex_side : float               side length of the outer hexagon

The objective is to MAXIMIZE 1/outer_hex_side (equivalently minimize the outer
side length) while every inner hexagon is a unit regular hexagon, the inner
hexagons are pairwise disjoint, and all inner hexagons are contained in the
outer hexagon. The current SOTA benchmark is 1/3.930092 ~= 0.2544.
"""
import numpy as np


def construct_packing():
    # A compact same-orientation honeycomb patch. The centers are taken from
    # axial hex-lattice coordinates, with a near-minimal expansion so adjacent
    # cells remain separated under the evaluator's 1e-6 SAT tolerance.
    h = np.sqrt(3.0)
    scale = 1.000000577355
    centers = scale * np.array(
        [
            [-2.0, -h],
            [-2.0, 0.0],
            [-2.0, h],
            [-0.5, -1.5 * h],
            [-0.5, -0.5 * h],
            [-0.5, 0.5 * h],
            [1.0, -h],
            [1.0, 0.0],
            [1.0, h],
            [2.5, -0.5 * h],
            [2.5, 0.5 * h],
        ],
        dtype=float,
    )
    inner_hex_data = np.column_stack((centers, np.zeros(11)))

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = 4.000001443391

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved).
def run_packing():
    """Run the hexagon packing constructor for n=11."""
    inner_hex_data, outer_hex_data, outer_hex_side_length = construct_packing()
    return inner_hex_data, outer_hex_data, float(outer_hex_side_length)


if __name__ == "__main__":
    inner, outer, side = run_packing()
    print(f"outer_hex_side_length: {side}")
    print(f"inv_outer_hex_side_length: {1.0 / side}")
