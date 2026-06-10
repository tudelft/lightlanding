import numpy as np
from itertools import combinations

def cross_ratio_1d(a, b, c, d):
    den = (a - d) * (b - c)
    if abs(den) < 1e-12:
        return np.inf
    return ((a - c) * (b - d)) / den

def filter_circles_same_line_similar_radius(
    circles: np.ndarray,
    radius_tol: float = 0.1,
    line_tol: float = 5.0,
    min_group_size: int = 4,
    cross_ratio_tol: float = 0.01
) -> np.ndarray:
    """
    Keep circles that belong to a group of circles lying approximately on the
    same line and having similar radii.

    Parameters
    ----------
    circles : np.ndarray
        Array of shape (N, 3), where each row is [x, y, r].
    radius_tol : float
        Allowed relative radius difference.
        Example: 0.25 means radii may differ by up to 25% from the group mean.
    line_tol : float
        Maximum perpendicular distance (in pixels) from the fitted line for a
        circle to be considered on that line.
    min_group_size : int
        Minimum number of circles needed to form a valid line group.
    cross_ratio_tol : float
        Tolerance for the cross-ratio test to identify equally spaced points.

    Returns
    -------
    np.ndarray
        Filtered array of circles, shape (M, 3).
    """
    circles = np.asarray(circles)

    if circles.ndim != 2 or circles.shape[1] != 3:
        raise ValueError("circles must have shape (N, 3)")

    n = len(circles)
    if n < min_group_size:
        return np.empty((0, 3), dtype=circles.dtype)

    best_groups = []

    # Try every pair of circles as a candidate line
    for i in range(n):
        x1, y1, r1 = circles[i]
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]

            dx = x2 - x1
            dy = y2 - y1

            norm = np.hypot(dx, dy)
            if norm < 1e-6:
                continue

            ux = dx / norm
            uy = dy / norm
        
            group_indices = [i, j]
            current_radii = [r1, r2]

            # Line equation based on pair (i, j)
            for k in range(n):
                if k == i or k == j:
                    continue

                x, y, r = circles[k]

                # Perpendicular distance from point to line through (x1,y1)-(x2,y2)
                dist = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norm

                mean_r = np.mean([r1, r2])
                radius_ok = abs(r - mean_r) <= radius_tol * mean_r
                line_ok = dist <= line_tol
                # print('line_ok, radius_ok', line_ok, radius_ok)
                if line_ok and radius_ok:
                    group_indices.append(k)
                    current_radii.append(r)

            if len(group_indices) >= min_group_size:
                valid = False
                for quad in set(combinations(group_indices, 4)):
                    projections = []
                    for idx in quad:
                        x, y, _ = circles[idx]
                        t = (x - x1) * ux + (y - y1) * uy
                        projections.append(t)

                    projections = np.sort(np.asarray(projections))
                    a, b, c, d = projections

                    cr = cross_ratio_1d(a, b, c, d)
                    if np.isfinite(cr) and abs(cr - 4/3) <= cross_ratio_tol:
                        # print('cross_ratio_tol_ok:', 'True')
                        valid = True
                        best_groups.extend(quad)
                        break

                if not valid:
                    # print('cross_ratio_tol_ok:', 'False')
                    continue

    best_groups = sorted(set(best_groups))
    print('len(best_groups)', len(best_groups))

#    if (len(best_groups)%4) != 0:
#        return np.empty((0, 3), dtype=circles.dtype)

    return circles[best_groups]