"""
BDD100K poly2d → structured lane targets.

Improved over the initial version by:
- supporting Scalabel poly2d dicts with types / Bezier hints
- resampling by arc length
- ranking lanes by vertical span / geometric usefulness
- generating visibility from in-image points
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import BDD_IMG_W, BDD_IMG_H, LANE_CAT_TO_ID


def _cubic_bezier(p0, p1, p2, p3, n: int = 32) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    mt = 1.0 - t
    pts = (
        (mt ** 3)[:, None] * p0[None] +
        (3 * mt * mt * t)[:, None] * p1[None] +
        (3 * mt * t * t)[:, None] * p2[None] +
        (t ** 3)[:, None] * p3[None]
    )
    return pts


def parse_poly2d(poly2d_field) -> List[np.ndarray]:
    if poly2d_field is None:
        return []
    out: List[np.ndarray] = []
    for item in poly2d_field:
        if isinstance(item, dict):
            verts = np.asarray(item.get("vertices", []), dtype=np.float64)
            types = item.get("types", "L" * len(verts))
            if len(verts) < 2:
                continue
            pieces = []
            i = 0
            while i < len(verts):
                if i + 3 < len(verts) and i + 1 < len(types) and types[i + 1] == "C":
                    curve = _cubic_bezier(verts[i], verts[i + 1], verts[i + 2], verts[i + 3])
                    if pieces:
                        curve = curve[1:]
                    pieces.append(curve)
                    i += 3
                else:
                    pt = verts[i:i + 2]
                    if len(pt) == 2:
                        pieces.append(pt if not pieces else pt[1:])
                    i += 1
            poly = np.concatenate(pieces, axis=0) if pieces else verts
            if len(poly) >= 2:
                out.append(poly)
        elif isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[0], (list, tuple)):
            arr = np.asarray(item, dtype=np.float64)
            if len(arr) >= 2:
                out.append(arr)
    return out


def resample_polyline(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        if len(pts) == 1:
            single = np.repeat(pts, n, axis=0)
            return single, np.ones(n, dtype=bool)
        return np.zeros((n, 2), dtype=np.float64), np.zeros(n, dtype=bool)

    y_span = np.ptp(pts[:, 1])
    if pts[-1, 1] < pts[0, 1]:
        pts = pts[::-1].copy()

    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total < 1e-6:
        out = np.repeat(pts[:1], n, axis=0)
        return out, np.ones(n, dtype=bool)

    sample = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float64)
    for i, d in enumerate(sample):
        idx = np.searchsorted(cum, d, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_len = float(seg_lens[idx])
        if seg_len < 1e-6:
            out[i] = pts[idx]
        else:
            t = (d - cum[idx]) / seg_len
            out[i] = (1.0 - t) * pts[idx] + t * pts[idx + 1]
    vis = np.ones(n, dtype=bool)
    if y_span < 2.0:
        vis[:] = False
    return out, vis


def frame_to_lane_targets(labels: List[dict], max_lanes: int = 10,
                          num_points: int = 72,
                          img_w: int = BDD_IMG_W,
                          img_h: int = BDD_IMG_H) -> Dict[str, np.ndarray]:
    existence = np.zeros(max_lanes, dtype=np.float32)
    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    lane_type = np.zeros(max_lanes, dtype=np.int64)

    candidates = []
    for label in labels:
        cat = label.get("category", "")
        if cat not in LANE_CAT_TO_ID:
            continue
        for pl in parse_poly2d(label.get("poly2d")):
            if len(pl) < 2:
                continue
            y_span = float(np.ptp(pl[:, 1]))
            length = float(np.linalg.norm(np.diff(pl, axis=0), axis=1).sum()) if len(pl) > 1 else 0.0
            score = 2.0 * y_span + 0.25 * length
            candidates.append((score, cat, pl))

    candidates.sort(key=lambda x: x[0], reverse=True)
    for lane_idx, (_, cat, pl) in enumerate(candidates[:max_lanes]):
        resampled, vis = resample_polyline(pl, num_points)
        vis = vis & (resampled[:, 0] >= 0) & (resampled[:, 0] <= img_w) & (resampled[:, 1] >= 0) & (resampled[:, 1] <= img_h)
        resampled[:, 0] /= img_w
        resampled[:, 1] /= img_h
        resampled = np.clip(resampled, 0.0, 1.0)
        existence[lane_idx] = 1.0
        points[lane_idx] = resampled.astype(np.float32)
        visibility[lane_idx] = vis.astype(np.float32)
        lane_type[lane_idx] = LANE_CAT_TO_ID[cat]

    return {
        "existence": existence,
        "points": points,
        "visibility": visibility,
        "lane_type": lane_type,
    }


class LaneLabelCache:
    def __init__(self, json_path: str, max_lanes: int = 10, num_points: int = 72):
        self.max_lanes = max_lanes
        self.num_points = num_points
        self._cache: Dict[str, List[dict]] = {}
        if json_path and os.path.isfile(json_path):
            print(f"Loading lane labels from {json_path} ...")
            with open(json_path, "r") as f:
                data = json.load(f)
            for frame in data:
                name = frame.get("name", "")
                labels = frame.get("labels") or []
                lane_labels = [l for l in labels if l.get("category", "") in LANE_CAT_TO_ID]
                if lane_labels:
                    self._cache[name] = lane_labels
            print(f"  Cached lane labels for {len(self._cache)} frames")
        else:
            print(f"  No lane labels found at: {json_path}")

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        labels = self._cache.get(image_name)
        if labels is None:
            return None
        return frame_to_lane_targets(labels, self.max_lanes, self.num_points)

    def __len__(self):
        return len(self._cache)

    def has_lanes(self, image_name: str) -> bool:
        return image_name in self._cache
