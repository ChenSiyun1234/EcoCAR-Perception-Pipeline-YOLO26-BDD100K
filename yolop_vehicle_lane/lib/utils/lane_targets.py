"""
BDD100K lane polyline parser.
Migrated from DETR_GeoLane_pipeline/src/lane_targets.py with local constants.

Supports:
- per-image directory labels, e.g. 100k/train/*.json and 100k/val/*.json
- consolidated JSON files
- official newer Scalabel-style records with top-level `labels`
- official old-format per-image records with top-level `frames -> objects`
- poly2d stored as dicts with vertices/types/closed
- old poly2d stored as point arrays, including [x, y, "L"|"C"]
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np

# BDD100K original image dimensions
BDD_IMG_W, BDD_IMG_H = 1280, 720

# Lane categories (exclude crosswalk for training)
LANE_CATEGORIES = [
    "lane/single white",
    "lane/single yellow",
    "lane/single other",
    "lane/double white",
    "lane/double yellow",
    "lane/double other",
    "lane/road curb",
    "lane/crosswalk",
]
LANE_TRAIN_CATS = [c for c in LANE_CATEGORIES if "crosswalk" not in c]
LANE_CAT_TO_ID = {c: i for i, c in enumerate(LANE_TRAIN_CATS)}

KNOWN_LANE_SUBTYPES = {
    "crosswalk",
    "double other",
    "double white",
    "double yellow",
    "road curb",
    "single other",
    "single white",
    "single yellow",
}

def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)]

def _record_image_name(record: dict, fallback_json_path: Optional[str] = None) -> str:
    for key in ["name", "image", "imageName", "filename", "id"]:
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            base = os.path.basename(v)
            if "." not in base:
                base += ".jpg"
            return base
    if fallback_json_path:
        return Path(fallback_json_path).stem + ".jpg"
    return ""

def _normalize_lane_category(label: dict) -> Optional[str]:
    cat = str(label.get("category", "") or "").strip().lower()
    attrs = label.get("attributes") or {}

    lane_types = []
    for key in [
        "laneTypes", "laneType", "lane_type",
        "type", "types", "subtype", "subtypes"
    ]:
        lane_types.extend(_as_list(attrs.get(key)))
        if key in label:
            lane_types.extend(_as_list(label.get(key)))

    if cat.startswith("lane/"):
        norm = cat
    elif cat == "lane":
        norm = None
        for t in lane_types:
            tt = t.strip().lower()
            if tt in KNOWN_LANE_SUBTYPES:
                norm = f"lane/{tt}"
                break
    elif cat in KNOWN_LANE_SUBTYPES:
        norm = f"lane/{cat}"
    else:
        norm = None

    if norm is None:
        text_bits = [cat]
        for _, v in attrs.items():
            text_bits.extend(_as_list(v))
        joined = " ".join(x.lower() for x in text_bits)
        for subtype in KNOWN_LANE_SUBTYPES:
            if subtype in joined:
                norm = f"lane/{subtype}"
                break

    return norm if norm in LANE_CAT_TO_ID else None

def _iter_candidate_objects(record: Any) -> Iterable[dict]:
    """Yield label/object dicts from both official BDD100K schemas."""
    if not isinstance(record, dict):
        return

    labels = record.get("labels")
    if isinstance(labels, list):
        for lab in labels:
            if isinstance(lab, dict):
                yield lab

    frames = record.get("frames")
    if isinstance(frames, list):
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            objs = fr.get("objects")
            if isinstance(objs, list):
                for obj in objs:
                    if isinstance(obj, dict):
                        yield obj

    objs = record.get("objects")
    if isinstance(objs, list):
        for obj in objs:
            if isinstance(obj, dict):
                yield obj

def _normalize_poly2d_items(poly2d_field) -> List[Any]:
    if poly2d_field is None:
        return []
    if isinstance(poly2d_field, dict):
        return [poly2d_field]
    if not isinstance(poly2d_field, list) or len(poly2d_field) == 0:
        return []

    first = poly2d_field[0]
    if isinstance(first, dict):
        return [x for x in poly2d_field if isinstance(x, dict)]
    if isinstance(first, (list, tuple)):
        if len(first) >= 2 and isinstance(first[0], (int, float)):
            return [poly2d_field]
        if len(first) > 0 and isinstance(first[0], (list, tuple, dict)):
            return [seg for seg in poly2d_field if isinstance(seg, (list, tuple, dict))]
    return []

def _coerce_point(v):
    if isinstance(v, dict):
        if "x" in v and "y" in v:
            try:
                return float(v["x"]), float(v["y"]), str(v.get("type", v.get("types", "L")) or "L")[0]
            except Exception:
                return None
        if "point" in v and isinstance(v["point"], (list, tuple)) and len(v["point"]) >= 2:
            try:
                return float(v["point"][0]), float(v["point"][1]), str(v.get("type", "L") or "L")[0]
            except Exception:
                return None

    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            x = float(v[0])
            y = float(v[1])
        except Exception:
            return None
        t = "L"
        if len(v) >= 3 and isinstance(v[2], str) and len(v[2]) > 0:
            t = v[2][0]
        return x, y, t
    return None

def _coerce_vertices_and_types(poly) -> Tuple[np.ndarray, str, bool]:
    if isinstance(poly, dict):
        verts = poly.get("vertices", []) or []
        types = poly.get("types", "") or ""
        closed = bool(poly.get("closed", False))
    else:
        verts = poly
        types = ""
        closed = False

    out_verts = []
    out_types = []
    for i, v in enumerate(verts):
        out = _coerce_point(v)
        if out is None:
            continue
        x, y, embedded_t = out
        out_verts.append([x, y])
        t = embedded_t
        if embedded_t == "L" and i < len(types):
            if isinstance(types[i], str) and len(types[i]) > 0:
                t = types[i][0]
        if t not in ("L", "C"):
            t = "L"
        out_types.append(t)

    return np.asarray(out_verts, dtype=np.float64), "".join(out_types), closed

def _sample_quad(p0, p1, p2, n=12):
    ts = np.linspace(0.0, 1.0, int(max(2, n)))
    return np.stack([
        ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2
        for t in ts
    ], axis=0)

def _sample_cubic(p0, p1, p2, p3, n=20):
    ts = np.linspace(0.0, 1.0, int(max(2, n)))
    return np.stack([
        ((1 - t) ** 3) * p0
        + 3 * ((1 - t) ** 2) * t * p1
        + 3 * (1 - t) * (t ** 2) * p2
        + (t ** 3) * p3
        for t in ts
    ], axis=0)

def _segment_to_dense_points(poly, bezier_samples=20) -> np.ndarray:
    verts, types, closed = _coerce_vertices_and_types(poly)
    if len(verts) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    dense = [verts[0].copy()]
    i = 1
    while i < len(verts):
        cur = dense[-1]
        ti = types[i] if i < len(types) else "L"

        if ti == "L":
            dense.append(verts[i].copy())
            i += 1
            continue

        # cubic: current -> C -> C -> L
        if i + 2 < len(verts):
            t0 = types[i] if i < len(types) else "L"
            t1 = types[i + 1] if i + 1 < len(types) else "L"
            t2 = types[i + 2] if i + 2 < len(types) else "L"
            if t0 == "C" and t1 == "C" and t2 == "L":
                pts = _sample_cubic(cur, verts[i], verts[i + 1], verts[i + 2], bezier_samples)
                dense.extend(pts[1:])
                i += 3
                continue

        # quadratic: current -> C -> L
        if i + 1 < len(verts):
            t0 = types[i] if i < len(types) else "L"
            t1 = types[i + 1] if i + 1 < len(types) else "L"
            if t0 == "C" and t1 == "L":
                pts = _sample_quad(cur, verts[i], verts[i + 1], max(8, bezier_samples // 2))
                dense.extend(pts[1:])
                i += 2
                continue

        dense.append(verts[i].copy())
        i += 1

    if closed and len(dense) > 2:
        if np.linalg.norm(np.asarray(dense[0]) - np.asarray(dense[-1])) > 1e-6:
            dense.append(np.asarray(dense[0]).copy())

    pts = np.asarray(dense, dtype=np.float64)
    keep = [0]
    for j in range(1, len(pts)):
        if np.linalg.norm(pts[j] - pts[keep[-1]]) > 1e-6:
            keep.append(j)
    return pts[keep]

def parse_poly2d(poly2d_field, bezier_samples: int = 20) -> List[np.ndarray]:
    out = []
    for seg in _normalize_poly2d_items(poly2d_field):
        pts = _segment_to_dense_points(seg, bezier_samples=bezier_samples)
        if len(pts) >= 2:
            out.append(pts)
    return out

def extract_lane_labels_any(record: dict) -> List[dict]:
    """Return lane-like objects that actually contain geometric data."""
    labels = []
    for lab in _iter_candidate_objects(record):
        if not isinstance(lab, dict):
            continue
        if _normalize_lane_category(lab) is None:
            continue
        if lab.get("poly2d") is not None or lab.get("seg2d") is not None:
            labels.append(lab)
    return labels

def resample_polyline(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if pts[-1, 1] < pts[0, 1]:
        pts = pts[::-1].copy()
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]
    if total < 1e-6:
        out = np.tile(pts[0], (n, 1))
        return out, np.zeros(n, dtype=bool)
    sample_dists = np.linspace(0.0, total, n)
    resampled = np.zeros((n, 2), dtype=np.float64)
    for i, d in enumerate(sample_dists):
        idx = np.searchsorted(cum_len, d, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_start = cum_len[idx]
        seg_len = seg_lens[idx]
        if seg_len < 1e-9:
            resampled[i] = pts[idx]
        else:
            t = (d - seg_start) / seg_len
            resampled[i] = pts[idx] * (1 - t) + pts[idx + 1] * t
    visibility = (
        (resampled[:, 0] >= 0.0) & (resampled[:, 0] <= BDD_IMG_W - 1) &
        (resampled[:, 1] >= 0.0) & (resampled[:, 1] <= BDD_IMG_H - 1)
    )
    return resampled, visibility

def frame_to_lane_targets(labels: List[dict], max_lanes: int = 10,
                          num_points: int = 72,
                          img_w: int = BDD_IMG_W,
                          img_h: int = BDD_IMG_H) -> Dict[str, np.ndarray]:
    existence = np.zeros(max_lanes, dtype=np.float32)
    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    lane_type = np.zeros(max_lanes, dtype=np.int64)

    lane_items: List[Tuple[str, np.ndarray]] = []
    for label in labels:
        norm_cat = _normalize_lane_category(label)
        if norm_cat is None:
            continue
        geom_field = label.get("poly2d")
        if geom_field is None and label.get("seg2d") is not None:
            geom_field = label.get("seg2d")
        for pl in parse_poly2d(geom_field):
            if len(pl) < 2:
                continue
            span_y = float(pl[:, 1].max() - pl[:, 1].min())
            arc = float(np.sqrt(((np.diff(pl, axis=0)) ** 2).sum(axis=1)).sum())
            if arc < 2.0 or span_y < 1.0:
                continue
            lane_items.append((norm_cat, pl))

    lane_items.sort(
        key=lambda x: (
            float(x[1][:, 1].max() - x[1][:, 1].min()),
            float(np.sqrt(((np.diff(x[1], axis=0)) ** 2).sum(axis=1)).sum())
        ),
        reverse=True
    )

    lane_idx = 0
    for norm_cat, pl in lane_items:
        if lane_idx >= max_lanes:
            break
        resampled, vis = resample_polyline(pl, num_points)
        norm = resampled.copy()
        norm[:, 0] /= float(img_w)
        norm[:, 1] /= float(img_h)
        existence[lane_idx] = 1.0
        points[lane_idx] = np.clip(norm, 0.0, 1.0).astype(np.float32)
        visibility[lane_idx] = vis.astype(np.float32)
        lane_type[lane_idx] = LANE_CAT_TO_ID[norm_cat]
        lane_idx += 1

    return {
        "existence": existence,
        "points": points,
        "visibility": visibility,
        "lane_type": lane_type,
    }

def inspect_json_for_lanes(source_path: str, limit: int = 3) -> List[dict]:
    samples = []
    paths = sorted(str(p) for p in Path(source_path).glob("*.json"))[:limit] if os.path.isdir(source_path) else [source_path]
    for p in paths:
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception as e:
            samples.append({"path": p, "error": str(e)})
            continue
        rec = data[0] if isinstance(data, list) and data else data
        labels = extract_lane_labels_any(rec) if isinstance(rec, dict) else []
        sample = {
            "path": p,
            "top_type": type(data).__name__,
            "record_keys": list(rec.keys())[:20] if isinstance(rec, dict) else None,
            "num_lane_like": len(labels),
        }
        if labels:
            lab = labels[0]
            sample["first_category"] = lab.get("category")
            sample["first_attr_keys"] = list((lab.get("attributes") or {}).keys())
            geom_field = lab.get("poly2d")
            if geom_field is None and lab.get("seg2d") is not None:
                geom_field = lab.get("seg2d")
                sample["used_seg2d"] = True
            polys = _normalize_poly2d_items(geom_field)
            sample["num_polys"] = len(polys)
            if polys:
                verts, types, closed = _coerce_vertices_and_types(polys[0])
                sample["first_vertices_head"] = verts[:5].tolist() if len(verts) else []
                sample["first_types"] = types[:20]
                sample["closed"] = closed
                dense = _segment_to_dense_points(polys[0], 12)
                sample["dense_points_head"] = dense[:5].tolist() if len(dense) else []
                sample["dense_len"] = int(len(dense))
        samples.append(sample)
    return samples


class LaneLabelCache:
    def __init__(self, source_path: str, max_lanes: int = 10, num_points: int = 72):
        self.max_lanes = max_lanes
        self.num_points = num_points
        self._cache: Dict[str, List[dict]] = {}
        self._debug_samples: List[dict] = []
        if not source_path:
            print("  No lane label source path provided")
            return
        if os.path.isdir(source_path):
            self._load_from_directory(source_path)
        elif os.path.isfile(source_path):
            self._load_from_file(source_path)
        else:
            print(f"  No lane labels found at: {source_path}")

    def _cache_record(self, record: dict, fallback_path: Optional[str] = None):
        labels = extract_lane_labels_any(record)
        name = _record_image_name(record, fallback_path)
        if not name or not labels:
            return
        targets = frame_to_lane_targets(labels, self.max_lanes, self.num_points)
        if float(targets["existence"].sum()) > 0:
            self._cache[name] = labels
            if len(self._debug_samples) < 3:
                first = labels[0]
                geom_field = first.get("poly2d")
                if geom_field is None and first.get("seg2d") is not None:
                    geom_field = first.get("seg2d")
                polys = _normalize_poly2d_items(geom_field)
                verts, types, _ = _coerce_vertices_and_types(polys[0]) if polys else (np.zeros((0, 2)), "", False)
                dense = _segment_to_dense_points(polys[0], 12) if polys else np.zeros((0, 2))
                self._debug_samples.append({
                    "name": name,
                    "n_labels": len(labels),
                    "existence_sum": float(targets["existence"].sum()),
                    "first_category": first.get("category"),
                    "first_vertices_head": verts[:5].tolist() if len(verts) else [],
                    "first_types": types[:20],
                    "dense_points_head": dense[:5].tolist() if len(dense) else [],
                    "dense_len": int(len(dense)),
                })

    def _load_from_directory(self, dir_path: str):
        print(f"Loading lane labels from per-image directory {dir_path} ...")
        json_files = sorted([str(p) for p in Path(dir_path).glob("*.json")])
        for jpath in json_files:
            try:
                with open(jpath, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        self._cache_record(rec, jpath)
            elif isinstance(data, dict):
                self._cache_record(data, jpath)
        print(f"  Cached lane labels for {len(self._cache)} frames from directory")
        if self._debug_samples:
            print(f"  Debug sample: {self._debug_samples[0]}")
        elif json_files:
            print("  No usable lane polylines extracted. JSON inspection:")
            for s in inspect_json_for_lanes(dir_path, limit=2):
                print(f"   {s}")

    def _load_from_file(self, json_path: str):
        print(f"Loading lane labels from {json_path} ...")
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Failed to load lane labels: {e}")
            return
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    self._cache_record(rec, json_path)
        elif isinstance(data, dict):
            self._cache_record(data, json_path)
        print(f"  Cached lane labels for {len(self._cache)} frames")
        if self._debug_samples:
            print(f"  Debug sample: {self._debug_samples[0]}")
        else:
            print(f"  No usable lane polylines extracted. JSON inspection: {inspect_json_for_lanes(json_path, limit=1)}")

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        labels = self._cache.get(image_name)
        if labels is None:
            return None
        return frame_to_lane_targets(labels, self.max_lanes, self.num_points)

    def __len__(self):
        return len(self._cache)

    def has_lanes(self, image_name: str) -> bool:
        return image_name in self._cache
