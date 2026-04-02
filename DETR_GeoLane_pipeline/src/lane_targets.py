"""
BDD100K poly2d → structured lane targets.

Parses the original BDD100K lane annotations (poly2d format) and converts
them into fixed-length ordered point sequences for query-based lane training.

This file now mirrors the more complete poly2d handling from the older YOLO26
pipeline, including support for Bezier control points in BDD100K annotations.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import BDD_IMG_W, BDD_IMG_H, LANE_CAT_TO_ID


def _bezier_curve(p0, p1, p2, p3, num_points: int = 30) -> List[Tuple[float, float]]:
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        mt = 1.0 - t
        x = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
        y = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
        pts.append((x, y))
    return pts


def _poly2d_dict_to_points(vertices: List[List[float]], types: str) -> np.ndarray:
    """Convert a Scalabel poly2d dict to a dense point list.

    Borrowed in spirit from the old src/lane_utils.py. Supports cubic-Bezier
    control points where the types string marks control vertices with 'C'.
    """
    if not vertices or len(vertices) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    points: List[Tuple[float, float]] = []
    i = 0
    n = len(vertices)
    types = types or ('L' * n)

    while i < n:
        vx, vy = float(vertices[i][0]), float(vertices[i][1])
        if i + 3 < n and i + 1 < len(types) and types[i + 1] == 'C':
            p0 = (vx, vy)
            p1 = (float(vertices[i + 1][0]), float(vertices[i + 1][1]))
            p2 = (float(vertices[i + 2][0]), float(vertices[i + 2][1]))
            p3 = (float(vertices[i + 3][0]), float(vertices[i + 3][1]))
            curve_pts = _bezier_curve(p0, p1, p2, p3, num_points=20)
            if points and curve_pts:
                curve_pts = curve_pts[1:]
            points.extend(curve_pts)
            i += 4
        else:
            points.append((vx, vy))
            i += 1

    return np.asarray(points, dtype=np.float64)


def parse_poly2d(poly2d_field) -> List[np.ndarray]:
    """Extract dense polylines from the BDD100K poly2d field.

    Supports the same variants accepted in the old pipeline:
      - list of dicts with vertices/types
      - list of points
      - list of lists of points
    """
    if poly2d_field is None or not isinstance(poly2d_field, list):
        return []

    polygons = []
    if len(poly2d_field) > 0:
        first = poly2d_field[0]
        if isinstance(first, dict):
            polygons = poly2d_field
        elif isinstance(first, (list, tuple)):
            if len(first) >= 2 and isinstance(first[0], (int, float)):
                polygons = [{"vertices": poly2d_field, "types": 'L' * len(poly2d_field)}]
            elif len(first) > 0 and isinstance(first[0], (list, tuple)):
                polygons = [{"vertices": p, "types": 'L' * len(p)} for p in poly2d_field if isinstance(p, (list, tuple))]

    out = []
    for poly in polygons:
        verts = poly.get('vertices', []) if isinstance(poly, dict) else []
        types = poly.get('types', 'L' * len(verts)) if isinstance(poly, dict) else ''
        pts = _poly2d_dict_to_points(verts, types)
        if pts.shape[0] >= 2:
            out.append(pts)
    return out


def resample_polyline(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if pts[-1, 1] < pts[0, 1]:
        pts = pts[::-1].copy()

    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]

    if total < 1e-6:
        out = np.tile(pts[0], (n, 1))
        vis = ((out[:, 0] >= 0) & (out[:, 0] < BDD_IMG_W) & (out[:, 1] >= 0) & (out[:, 1] < BDD_IMG_H))
        return out, vis.astype(bool)

    sample_dists = np.linspace(0.0, total, n)
    resampled = np.zeros((n, 2), dtype=np.float64)
    for i, d in enumerate(sample_dists):
        idx = np.searchsorted(cum_len, d, side='right') - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_start = cum_len[idx]
        seg_len = seg_lens[idx]
        if seg_len < 1e-9:
            resampled[i] = pts[idx]
        else:
            t = (d - seg_start) / seg_len
            resampled[i] = pts[idx] * (1 - t) + pts[idx + 1] * t

    vis = ((resampled[:, 0] >= 0) & (resampled[:, 0] < BDD_IMG_W) & (resampled[:, 1] >= 0) & (resampled[:, 1] < BDD_IMG_H))
    return resampled, vis.astype(bool)


def frame_to_lane_targets(labels: List[dict], max_lanes: int = 10, num_points: int = 72,
                          img_w: int = BDD_IMG_W, img_h: int = BDD_IMG_H) -> Dict[str, np.ndarray]:
    existence = np.zeros(max_lanes, dtype=np.float32)
    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    lane_type = np.zeros(max_lanes, dtype=np.int64)

    collected = []
    for label in labels:
        cat = label.get('category', '')
        if cat not in LANE_CAT_TO_ID:
            continue
        polylines = parse_poly2d(label.get('poly2d'))
        for pl in polylines:
            if pl.shape[0] < 2:
                continue
            y_span = float(pl[:, 1].max() - pl[:, 1].min())
            if y_span < 5.0:
                continue
            collected.append((y_span, cat, pl))

    collected.sort(key=lambda x: x[0], reverse=True)
    for lane_idx, (_, cat, pl) in enumerate(collected[:max_lanes]):
        resampled, vis = resample_polyline(pl, num_points)
        clipped = resampled.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, img_w - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, img_h - 1)
        norm = clipped.copy()
        norm[:, 0] /= float(img_w)
        norm[:, 1] /= float(img_h)

        existence[lane_idx] = 1.0
        points[lane_idx] = norm.astype(np.float32)
        visibility[lane_idx] = vis.astype(np.float32)
        lane_type[lane_idx] = LANE_CAT_TO_ID[cat]

    return {
        'existence': existence,
        'points': points,
        'visibility': visibility,
        'lane_type': lane_type,
    }


class LaneLabelCache:
    def __init__(self, json_path: str, max_lanes: int = 10, num_points: int = 72):
        self.max_lanes = max_lanes
        self.num_points = num_points
        self._cache: Dict[str, List[dict]] = {}

        if json_path and os.path.isfile(json_path):
            print(f"Loading lane labels from {json_path} ...")
            with open(json_path, 'r') as f:
                data = json.load(f)
            for frame in data:
                name = frame.get('name', '')
                # Support both consolidated {name, labels} and scalabel-ish frame dicts.
                labels = frame.get('labels') or []
                lane_labels = [l for l in labels if l.get('category', '').startswith('lane/')]
                if lane_labels:
                    self._cache[name] = lane_labels
            print(f"  Cached lane labels for {len(self._cache)} frames")
        else:
            print(f"  No lane labels found at: {json_path}")

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        labels = self._cache.get(image_name)
        if labels is None:
            stem = os.path.splitext(image_name)[0]
            labels = self._cache.get(stem)
        if labels is None:
            return None
        return frame_to_lane_targets(labels, self.max_lanes, self.num_points)

    def __len__(self):
        return len(self._cache)

    def has_lanes(self, image_name: str) -> bool:
        return image_name in self._cache or os.path.splitext(image_name)[0] in self._cache
