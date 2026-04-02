"""
Temporal utilities for lane association and smoothing.

These helpers are designed for video inference and future temporal training.
Current scope:
- cross-frame lane association via Hungarian matching on curve geometry
- temporal smoothing of matched lane polylines
- reusable hooks for future temporal consistency loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class LaneTrack:
    track_id: int
    points: np.ndarray          # (N, 2) normalized
    score: float
    age: int = 0
    hits: int = 1


def _resample_curve(points: np.ndarray, n: int = 96) -> np.ndarray:
    if points.shape[0] == n:
        return points.astype(np.float32)
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0) if len(pts) == 1 else np.zeros((n, 2), dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    sample = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float32)
    for i, d in enumerate(sample):
        idx = np.searchsorted(cum, d, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_len = float(seg[idx])
        if seg_len < 1e-6:
            out[i] = pts[idx]
        else:
            t = (d - cum[idx]) / seg_len
            out[i] = (1 - t) * pts[idx] + t * pts[idx + 1]
    return out


def _curve_direction(points: np.ndarray) -> np.ndarray:
    pts = _resample_curve(points)
    tang = np.diff(pts, axis=0)
    norms = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.clip(norms, 1e-6, None)
    return tang


def curve_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Symmetric Chamfer-like distance between two polylines."""
    a = _resample_curve(points_a)
    b = _resample_curve(points_b)
    dmat = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(dmat.min(axis=1).mean() + dmat.min(axis=0).mean()) * 0.5


def curve_direction_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    da = _curve_direction(points_a)
    db = _curve_direction(points_b)
    m = min(len(da), len(db))
    if m == 0:
        return 1.0
    cos = (da[:m] * db[:m]).sum(axis=-1)
    return float(1.0 - cos.mean())


class LaneAssociator:
    def __init__(self, dist_weight: float = 1.0, dir_weight: float = 0.35,
                 max_cost: float = 0.10):
        self.dist_weight = dist_weight
        self.dir_weight = dir_weight
        self.max_cost = max_cost
        self.next_id = 0
        self.tracks: List[LaneTrack] = []

    def associate(self, lanes: List[Dict]) -> List[Dict]:
        """Assign stable IDs to current-frame lanes.

        Each lane dict must contain:
        - points: (N,2) normalized numpy array
        - score: float
        """
        if not lanes:
            for tr in self.tracks:
                tr.age += 1
            self.tracks = [tr for tr in self.tracks if tr.age <= 5]
            return []

        if not self.tracks:
            out = []
            for lane in lanes:
                tr = LaneTrack(track_id=self.next_id, points=lane["points"], score=float(lane.get("score", 1.0)))
                self.tracks.append(tr)
                lane = dict(lane)
                lane["track_id"] = self.next_id
                out.append(lane)
                self.next_id += 1
            return out

        cost = np.zeros((len(self.tracks), len(lanes)), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            for j, lane in enumerate(lanes):
                d = curve_distance(tr.points, lane["points"])
                dd = curve_direction_distance(tr.points, lane["points"])
                cost[i, j] = self.dist_weight * d + self.dir_weight * dd

        row, col = linear_sum_assignment(cost)
        matched_tracks = set()
        matched_lanes = set()
        out = []

        for i, j in zip(row.tolist(), col.tolist()):
            if cost[i, j] > self.max_cost:
                continue
            tr = self.tracks[i]
            lane = dict(lanes[j])
            tr.points = lane["points"]
            tr.score = float(lane.get("score", tr.score))
            tr.age = 0
            tr.hits += 1
            lane["track_id"] = tr.track_id
            out.append(lane)
            matched_tracks.add(i)
            matched_lanes.add(j)

        for i, tr in enumerate(self.tracks):
            if i not in matched_tracks:
                tr.age += 1

        for j, lane in enumerate(lanes):
            if j in matched_lanes:
                continue
            tr = LaneTrack(track_id=self.next_id, points=lane["points"], score=float(lane.get("score", 1.0)))
            self.tracks.append(tr)
            lane = dict(lane)
            lane["track_id"] = self.next_id
            out.append(lane)
            self.next_id += 1

        self.tracks = [tr for tr in self.tracks if tr.age <= 5]
        return out


def temporal_curve_smoothing(current_points: np.ndarray, prev_points: Optional[np.ndarray],
                             alpha: float = 0.70) -> np.ndarray:
    """Simple EMA smoothing in polyline space."""
    curr = _resample_curve(current_points)
    if prev_points is None:
        return curr
    prev = _resample_curve(prev_points, n=curr.shape[0])
    return (alpha * curr + (1.0 - alpha) * prev).astype(np.float32)
