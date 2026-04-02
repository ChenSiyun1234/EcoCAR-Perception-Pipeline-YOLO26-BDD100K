
"""Cross-frame lane association and temporal smoothing helpers."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def curve_distance_np(a: np.ndarray, b: np.ndarray) -> float:
    def point_to_poly(points, poly):
        if len(poly) < 2:
            return np.linalg.norm(points - poly[:1], axis=1)
        seg_a = poly[:-1]; seg_b = poly[1:]; ab = seg_b - seg_a
        out = []
        for p in points:
            ap = p[None,:] - seg_a
            denom = np.sum(ab*ab, axis=1)
            denom = np.clip(denom, 1e-9, None)
            t = np.sum(ap * ab, axis=1) / denom
            t = np.clip(t, 0.0, 1.0)
            proj = seg_a + t[:,None] * ab
            d = np.sqrt(np.sum((proj - p[None,:])**2, axis=1)).min()
            out.append(d)
        return np.asarray(out)
    if len(a) == 0 or len(b) == 0:
        return 1e9
    return float(point_to_poly(a, b).mean() + point_to_poly(b, a).mean())


def associate_lanes(prev_lanes, curr_lanes, dist_thresh_px=40.0):
    if len(prev_lanes) == 0 or len(curr_lanes) == 0:
        return [], list(range(len(curr_lanes)))
    cost = np.zeros((len(prev_lanes), len(curr_lanes)), dtype=np.float32)
    for i, pl in enumerate(prev_lanes):
        for j, cl in enumerate(curr_lanes):
            cost[i,j] = curve_distance_np(pl['points'], cl['points'])
    pi, ci = linear_sum_assignment(cost)
    matches = []
    unmatched_curr = set(range(len(curr_lanes)))
    for i,j in zip(pi,ci):
        if cost[i,j] <= dist_thresh_px:
            matches.append((i,j,float(cost[i,j]))); unmatched_curr.discard(j)
    return matches, sorted(unmatched_curr)


def smooth_lane_points(prev_points, curr_points, alpha=0.65):
    n = min(len(prev_points), len(curr_points))
    out = curr_points.copy()
    out[:n] = alpha * curr_points[:n] + (1.0 - alpha) * prev_points[:n]
    return out
