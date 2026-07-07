# EVOLVE-BLOCK-START
"""Circle packing n=26: SLSQP + topological swaps + crossover + Fibonacci starts."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
WALL_SECONDS = 340
_I, _J = np.triu_indices(N, k=1)

_SCALES = [
    (0.010, 0.04),
    (0.015, 0.06),
    (0.008, 0.03),
    (0.040, 0.15),
    (0.012, 0.05),
    (0.020, 0.08),
    (0.008, 0.03),
    (0.060, 0.22),
    (0.100, 0.35),
]


def _hex_centers(n, margin, row_offset_frac, rng):
    usable = 1.0 - 2 * margin
    cols = max(2, int(np.ceil(np.sqrt(n * 2 / np.sqrt(3)))))
    rows = max(2, int(np.ceil(n / cols)) + 2)
    dx = usable / (cols - 1)
    dy = usable / max(rows - 1, 1)
    pts = []
    for row in range(rows):
        for col in range(cols):
            if len(pts) >= n:
                break
            x = margin + col * dx + (row_offset_frac * dx if row % 2 else 0.0)
            y = margin + row * dy
            pts.append([np.clip(x, margin, 1.0 - margin),
                         np.clip(y, margin, 1.0 - margin)])
    pts = np.array(pts[:n], dtype=float)
    pts += rng.uniform(-dx * 0.05, dx * 0.05, pts.shape)
    return np.clip(pts, margin, 1.0 - margin)


def _fibonacci_centers(n, rng):
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    pts = []
    for i in range(n):
        r = np.sqrt((i + 0.5) / n) * 0.43
        theta = 2.0 * np.pi * golden * i
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        pts.append([np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)])
    pts = np.array(pts, dtype=float)
    pts += rng.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.005, 0.995)


def _is_feasible(centers, radii, tol=2e-6):
    if np.any(centers - radii[:, None] < -tol):
        return False
    if np.any(centers + radii[:, None] > 1.0 + tol):
        return False
    diff = centers[_I] - centers[_J]
    dists = np.sqrt((diff * diff).sum(1))
    return not np.any(dists < radii[_I] + radii[_J] - tol)


def _slsqp(ci, ri, maxiter=2000):
    n = N
    x0 = np.concatenate([ci.ravel(), ri])
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n

    def obj(x):
        return -x[2 * n:].sum()

    def cons(x):
        p = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        diff = p[_I] - p[_J]
        nonover = np.sqrt((diff * diff).sum(1)) - r[_I] - r[_J]
        bnd = np.stack([p[:, 0] - r, 1.0 - p[:, 0] - r,
                         p[:, 1] - r, 1.0 - p[:, 1] - r], axis=1).ravel()
        return np.concatenate([nonover, bnd])

    res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                   constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": maxiter, "ftol": 1e-11, "disp": False})
    centers = res.x[:2 * n].reshape(n, 2)
    radii = np.maximum(res.x[2 * n:], 0.0)
    return centers, radii, radii.sum()


def construct_packing():
    t0 = time.monotonic()
    best_c, best_r, best_s = None, None, 0.0
    pool = []

    def elapsed():
        return time.monotonic() - t0

    def register(c, r, s):
        nonlocal best_s, best_c, best_r
        if not _is_feasible(c, r):
            return
        if s > best_s:
            best_s, best_c, best_r = s, c.copy(), r.copy()
        pool.append((s, c.copy(), r.copy()))
        pool.sort(key=lambda x: -x[0])
        del pool[5:]

    hex_cfgs = [
        (0.08,  0.5,   0.090, 42),
        (0.075, 0.5,   0.088, 123),
        (0.09,  0.5,   0.087, 456),
        (0.08,  0.5,   0.092, 789),
        (0.07,  0.5,   0.091, 1337),
        (0.085, 0.5,   0.086, 2024),
        (0.08,  0.45,  0.089, 31337),
        (0.075, 0.55,  0.091, 99999),
        (0.09,  0.48,  0.088, 7777),
        (0.08,  0.5,   0.090, 54321),
        (0.078, 0.5,   0.091, 11111),
        (0.082, 0.52,  0.089, 22222),
    ]
    for margin, row_off, r0, seed in hex_cfgs:
        if elapsed() > WALL_SECONDS:
            break
        rng = np.random.default_rng(seed)
        ci = _hex_centers(N, margin, row_off, rng)
        ri = np.clip(np.full(N, r0) + rng.uniform(-0.005, 0.005, N), 0.01, 0.3)
        try:
            c, r, s = _slsqp(ci, ri)
        except Exception:
            continue
        register(c, r, s)

    for seed in [111, 222, 333, 444]:
        if elapsed() > WALL_SECONDS:
            break
        rng = np.random.default_rng(seed)
        ci = _fibonacci_centers(N, rng)
        ri = np.clip(np.full(N, 0.089) + rng.uniform(-0.005, 0.005, N), 0.01, 0.3)
        try:
            c, r, s = _slsqp(ci, ri)
        except Exception:
            continue
        register(c, r, s)

    rng2 = np.random.default_rng(888888)
    k = 0
    while elapsed() < WALL_SECONDS - 5 and pool:
        sc, sr = _SCALES[k % len(_SCALES)]
        pool_idx = (k // len(_SCALES)) % len(pool)
        base_c, base_r = pool[pool_idx][1], pool[pool_idx][2]
        move_type = k % 6

        if move_type < 2:
            ci = np.clip(base_c + rng2.normal(0, sc, base_c.shape), 0.005, 0.995)
            ri = np.clip(base_r * np.exp(rng2.normal(0, sr, base_r.shape)), 1e-6, 0.5)
        elif move_type < 4:
            n_mov = int(rng2.integers(3, 8))
            idx = rng2.choice(N, size=n_mov, replace=False)
            ci = base_c.copy()
            ri = base_r.copy()
            ci[idx] += rng2.normal(0, sc * 2.5, (n_mov, 2))
            ri[idx] *= np.exp(rng2.normal(0, sr * 2.0, n_mov))
            ci = np.clip(ci, 0.005, 0.995)
            ri = np.clip(ri, 1e-6, 0.5)
        elif move_type == 4:
            n_sw = int(rng2.integers(2, 5))
            all_idx = rng2.choice(N, size=n_sw * 2, replace=False)
            idxa, idxb = all_idx[:n_sw], all_idx[n_sw:]
            ci = base_c.copy()
            ri = base_r.copy()
            ci[idxa], ci[idxb] = ci[idxb].copy(), ci[idxa].copy()
            ci += rng2.normal(0, 0.003, ci.shape)
            ci = np.clip(ci, 0.005, 0.995)
        else:
            if len(pool) >= 2:
                p2_c = pool[1][1]
                p2_r = pool[1][2]
                mask = rng2.random(N) > 0.5
                ci = np.where(mask[:, None], base_c, p2_c)
                ri = np.where(mask, base_r, p2_r)
            else:
                ci = base_c.copy()
                ri = base_r.copy()
            ci = ci + rng2.normal(0, 0.005, ci.shape)
            ci = np.clip(ci, 0.005, 0.995)
            ri = np.clip(ri, 1e-6, 0.5)

        try:
            c, r, s = _slsqp(ci, ri)
        except Exception:
            k += 1
            continue
        register(c, r, s)
        k += 1

    if best_c is None:
        xs = np.linspace(0.1, 0.9, 5)
        cx, cy = np.meshgrid(xs, xs)
        pts = np.column_stack([cx.ravel(), cy.ravel()])
        best_c = np.vstack([pts, [[0.5, 0.1]]])[:N]
        best_r = np.full(N, 0.09)

    best_r = np.maximum(best_r, 0.0)
    return best_c, best_r, float(best_r.sum())


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
