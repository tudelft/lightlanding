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
    cross_ratio_tol: float = 0.01,
    right_angle_tol_deg: float | None = None,
) -> np.ndarray:
    """
    Keep circles only when they form the marker's two line groups: each group
    has four approximately collinear, similarly sized, equally spaced LEDs,
    and the two groups share exactly one endpoint (the amber corner LED).

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
    right_angle_tol_deg : float | None
        Maximum deviation from 90 degrees between the two line groups. Set to
        ``None`` to disable the image-space right-angle check.

    Returns
    -------
    np.ndarray
        The seven circles belonging to the two marker arms, or an empty array
        when no such pair of groups is found.
    """
    circles = np.asarray(circles)

    if circles.ndim != 2 or circles.shape[1] != 3:
        raise ValueError("circles must have shape (N, 3)")
    if right_angle_tol_deg is not None and not 0.0 <= right_angle_tol_deg < 90.0:
        raise ValueError("right_angle_tol_deg must be in [0, 90), or None")

    n = len(circles)
    if n < 7:
        return np.empty((0, 3), dtype=circles.dtype)

    valid_groups = {}

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
                    quad_radii = circles[list(quad), 2]
                    quad_mean_radius = float(np.mean(quad_radii))
                    if (
                        quad_mean_radius <= 0.0
                        or np.any(np.abs(quad_radii - quad_mean_radius) > radius_tol * quad_mean_radius)
                    ):
                        continue

                    projections = []
                    for idx in quad:
                        x, y, _ = circles[idx]
                        t = (x - x1) * ux + (y - y1) * uy
                        projections.append(t)

                    ordered = np.argsort(projections)
                    projections = np.asarray(projections)[ordered]
                    a, b, c, d = projections

                    cr = cross_ratio_1d(a, b, c, d)
                    if np.isfinite(cr) and abs(cr - 4/3) <= cross_ratio_tol:
                        valid = True
                        group_key = frozenset(quad)
                        valid_groups[group_key] = {
                            "indices": group_key,
                            "endpoints": frozenset((quad[ordered[0]], quad[ordered[-1]])),
                            "direction": np.array([ux, uy]),
                        }

                if not valid:
                    # print('cross_ratio_tol_ok:', 'False')
                    continue

    marker_candidates = set()
    groups = list(valid_groups.values())
    for first, second in combinations(groups, 2):
        shared = first["indices"] & second["indices"]
        if len(shared) != 1:
            continue

        shared_index = next(iter(shared))
        if shared_index not in first["endpoints"] or shared_index not in second["endpoints"]:
            continue

        if right_angle_tol_deg is not None:
            cosine = abs(float(np.dot(first["direction"], second["direction"])))
            image_angle_deg = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
            if abs(90.0 - image_angle_deg) > right_angle_tol_deg:
                continue

        selected_indices = first["indices"] | second["indices"]
        if len(selected_indices) != 7:
            continue

        marker_candidates.add(frozenset(selected_indices))

    if len(marker_candidates) != 1:
        return np.empty((0, 3), dtype=circles.dtype)

    return circles[sorted(next(iter(marker_candidates)))]
