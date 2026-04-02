
"""BDD100K poly2d parsing utilities and lane target cache."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import BDD_IMG_W, BDD_IMG_H, LANE_CAT_TO_ID


def _bezier_curve(p0, p1, p2, p3, num_points=30):
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        mt = 1 - t
        x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append([float(x), float(y)])
    return pts


def _poly2d_to_polygon_dicts(poly2d_data):
    polys = []
    if poly2d_data is None:
        return polys
    if isinstance(poly2d_data, dict):
        return [poly2d_data]
    if not isinstance(poly2d_data, list) or len(poly2d_data) == 0:
        return polys
    first = poly2d_data[0]
    if isinstance(first, dict):
        return poly2d_data
    if isinstance(first, (list, tuple)):
        if len(first) >= 2 and isinstance(first[0], (int, float)):
            return [{'vertices': poly2d_data}]
        if len(first) > 0 and isinstance(first[0], (list, tuple)):
            return [{'vertices': p} for p in poly2d_data if isinstance(p, list) and len(p) > 0]
    return polys


def _normalize_vertices_and_types(poly):
    if isinstance(poly, dict):
        verts = poly.get('vertices', []) or []
        types = poly.get('types', '') or ('L' * len(verts))
    elif isinstance(poly, list):
        verts = poly
        types = 'L' * len(verts)
    else:
        return [], ''
    out_verts, out_types = [], []
    for idx, v in enumerate(verts):
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        try:
            x, y = float(v[0]), float(v[1])
        except Exception:
            continue
        out_verts.append([x, y])
        if len(v) >= 3 and isinstance(v[2], str) and v[2]:
            out_types.append(v[2][0])
        elif idx < len(types) and isinstance(types[idx], str) and types[idx]:
            out_types.append(types[idx][0])
        else:
            out_types.append('L')
    return out_verts, ''.join(out_types)


def poly2d_to_dense_points(poly2d_field, bezier_points=30) -> List[np.ndarray]:
    dense = []
    for poly in _poly2d_to_polygon_dicts(poly2d_field):
        verts, types = _normalize_vertices_and_types(poly)
        if len(verts) < 2:
            continue
        pts = []
        i = 0
        n = len(verts)
        while i < n:
            v = verts[i]
            if i + 3 < n and i + 1 < len(types) and types[i + 1] == 'C':
                p0, p1, p2, p3 = verts[i], verts[i+1], verts[i+2], verts[i+3]
                curve = _bezier_curve(p0, p1, p2, p3, num_points=bezier_points)
                if pts and curve:
                    pts.extend(curve[1:])
                else:
                    pts.extend(curve)
                i += 4
            else:
                pts.append(v)
                i += 1
        if len(pts) >= 2:
            dense.append(np.asarray(pts, dtype=np.float64))
    return dense


def resample_polyline(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        out = np.repeat(pts[:1], n, axis=0) if pts.shape[0] == 1 else np.zeros((n,2),dtype=np.float64)
        return out, np.zeros(n, dtype=bool)
    if pts[-1,1] < pts[0,1]:
        pts = pts[::-1].copy()
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total < 1e-6:
        out = np.repeat(pts[:1], n, axis=0)
        return out, np.zeros(n, dtype=bool)
    sample_d = np.linspace(0.0, total, n)
    out = np.zeros((n,2), dtype=np.float64)
    for i, d in enumerate(sample_d):
        idx = np.searchsorted(cum, d, side='right') - 1
        idx = np.clip(idx, 0, len(pts)-2)
        seg_start = cum[idx]
        seg_len = max(seg_lens[idx], 1e-9)
        t = (d - seg_start) / seg_len
        out[i] = pts[idx]*(1-t) + pts[idx+1]*t
    vis = (out[:,0] >= 0) & (out[:,0] <= BDD_IMG_W - 1) & (out[:,1] >= 0) & (out[:,1] <= BDD_IMG_H - 1)
    out[:,0] = np.clip(out[:,0], 0, BDD_IMG_W - 1)
    out[:,1] = np.clip(out[:,1], 0, BDD_IMG_H - 1)
    return out, vis


def frame_to_lane_targets(labels: List[dict], max_lanes: int = 10, num_points: int = 72,
                          img_w: int = BDD_IMG_W, img_h: int = BDD_IMG_H) -> Dict[str, np.ndarray]:
    existence = np.zeros(max_lanes, dtype=np.float32)
    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    lane_type = np.zeros(max_lanes, dtype=np.int64)

    lane_items = []
    for label in labels:
        cat = label.get('category', '')
        if cat not in LANE_CAT_TO_ID:
            continue
        for pl in poly2d_to_dense_points(label.get('poly2d')):
            if pl.shape[0] < 2:
                continue
            y_span = float(pl[:,1].max() - pl[:,1].min())
            arc = float(np.linalg.norm(np.diff(pl, axis=0), axis=1).sum())
            lane_items.append((y_span, arc, cat, pl))

    lane_items.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for lane_idx, (_, _, cat, pl) in enumerate(lane_items[:max_lanes]):
        rs, vis = resample_polyline(pl, num_points)
        rs[:,0] /= img_w; rs[:,1] /= img_h
        existence[lane_idx] = 1.0
        points[lane_idx] = np.clip(rs, 0.0, 1.0).astype(np.float32)
        visibility[lane_idx] = vis.astype(np.float32)
        lane_type[lane_idx] = LANE_CAT_TO_ID[cat]
    return {'existence': existence, 'points': points, 'visibility': visibility, 'lane_type': lane_type}


class LaneLabelCache:
    def __init__(self, json_path: Optional[str], max_lanes: int = 10, num_points: int = 72):
        self.max_lanes = max_lanes
        self.num_points = num_points
        self._cache: Dict[str, List[dict]] = {}
        self.json_path = json_path
        if json_path and os.path.isfile(json_path):
            print(f'Loading lane labels from {json_path} ...')
            with open(json_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'frames' in data:
                data = data['frames']
            for frame in data:
                name = frame.get('name') or frame.get('videoName') or ''
                if name:
                    name = Path(name).name
                labels = frame.get('labels') or frame.get('objects') or []
                lane_labels = [l for l in labels if isinstance(l.get('category',''), str) and l.get('category','').startswith('lane/')]
                if lane_labels and name:
                    self._cache[name] = lane_labels
            print(f'  Cached lane labels for {len(self._cache)} frames')
        else:
            print(f'  No lane labels found at: {json_path}')

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        name = Path(image_name).name
        labels = self._cache.get(name)
        if labels is None:
            return None
        return frame_to_lane_targets(labels, self.max_lanes, self.num_points)

    def __len__(self):
        return len(self._cache)

    def has_lanes(self, image_name: str) -> bool:
        return Path(image_name).name in self._cache


def inspect_lane_json(json_path: str, limit: int = 3) -> Dict[str, object]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'frames' in data:
        frames = data['frames']
    else:
        frames = data
    preview = []
    for frame in frames[:limit]:
        labels = frame.get('labels') or frame.get('objects') or []
        preview.append({
            'name': frame.get('name'),
            'num_labels': len(labels),
            'lane_categories': [l.get('category') for l in labels if isinstance(l.get('category'), str) and 'lane' in l.get('category','').lower()][:10],
            'poly2d_type': type((labels[0].get('poly2d') if labels else None)).__name__ if labels else None,
        })
    return {'num_frames': len(frames), 'preview': preview}
