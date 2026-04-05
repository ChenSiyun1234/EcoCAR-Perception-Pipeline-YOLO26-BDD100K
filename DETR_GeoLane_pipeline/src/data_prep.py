from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Union

VEHICLE_CATEGORIES = ["car", "truck", "bus", "motorcycle", "bicycle"]
VEHICLE_TO_ID = {name: i for i, name in enumerate(VEHICLE_CATEGORIES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_records_from_source(source: str | Path) -> Iterable[Tuple[dict, Path]]:
    source = Path(source)
    if source.is_dir():
        for p in sorted(source.rglob("*.json")):
            try:
                data = _load_json(p)
            except Exception:
                continue
            if isinstance(data, dict):
                yield data, p
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item, p
        return

    try:
        data = _load_json(source)
    except Exception:
        return
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item, source
    elif isinstance(data, dict):
        yield data, source


def _record_name(record: dict, fallback: Path) -> str:
    name = record.get("name")
    if isinstance(name, str) and name.strip():
        return Path(name).name
    return fallback.stem + ".jpg"


def _extract_object_labels(record: dict) -> List[dict]:
    labels = record.get("labels")
    if isinstance(labels, list):
        return [x for x in labels if isinstance(x, dict)]
    frames = record.get("frames")
    if isinstance(frames, list) and frames:
        objs = []
        for fr in frames:
            if isinstance(fr, dict) and isinstance(fr.get("objects"), list):
                objs.extend([x for x in fr["objects"] if isinstance(x, dict)])
        if objs:
            return objs
    objs = record.get("objects")
    if isinstance(objs, list):
        return [x for x in objs if isinstance(x, dict)]
    return []


def _has_detection_content(record: dict) -> bool:
    labels = _extract_object_labels(record)
    for lab in labels[:100]:
        cat = lab.get("category")
        if isinstance(cat, str) and cat in VEHICLE_TO_ID:
            return True
        if isinstance(lab.get("box2d"), dict):
            return True
    return False


def _classify_source(path: Path) -> Optional[str]:
    if path.is_dir():
        checked = 0
        for jp in sorted(path.rglob("*.json"))[:50]:
            try:
                data = _load_json(jp)
            except Exception:
                continue
            checked += 1
            rec = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None)
            if isinstance(rec, dict) and _has_detection_content(rec):
                return "detection"
        return None if checked else None
    return _classify_json_by_content(path)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _zip_members(zip_path: str | Path) -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.namelist()


def unzip_if_needed(zip_path: str | Path, dest_root: str | Path) -> Path:
    zip_path = Path(zip_path)
    dest_root = ensure_dir(dest_root)
    marker = dest_root / f".extracted_{zip_path.stem}"
    if marker.exists():
        return dest_root
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_root)
    marker.write_text("ok", encoding="utf-8")
    return dest_root


def _find_all_jsons(root: str | Path) -> List[Path]:
    return sorted([p for p in Path(root).rglob("*.json") if p.is_file()])


def _norm_path_str(path: str | Path) -> str:
    return str(path).lower().replace("\\", "/")


def _score_json_candidate(path: Path) -> int:
    s = _norm_path_str(path)
    score = 0
    if "train" in s:
        score += 4
    if "val" in s or "valid" in s:
        score += 4
    if "detect" in s or "det_" in s or "labels_images" in s:
        score += 3
    if "lane" in s or "seg" in s or "drivable" in s:
        score -= 4
    return score


def _classify_json_by_content(path: Path) -> Optional[str]:
    try:
        data = _load_json(path)
    except Exception:
        return None

    if isinstance(data, dict):
        return "detection" if _has_detection_content(data) else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return "detection" if _has_detection_content(data[0]) else None
    return None


def locate_detection_jsons(raw_root: str | Path) -> Tuple[Path, Path]:
    raw_root = Path(raw_root)

    preferred_train_dirs = [
        raw_root / "100k" / "train",
        raw_root / "bdd100k" / "100k" / "train",
    ]
    preferred_val_dirs = [
        raw_root / "100k" / "val",
        raw_root / "bdd100k" / "100k" / "val",
    ]

    train_dir = next((p for p in preferred_train_dirs if p.is_dir() and _classify_source(p) == "detection"), None)
    val_dir = next((p for p in preferred_val_dirs if p.is_dir() and _classify_source(p) == "detection"), None)
    if train_dir and val_dir:
        return train_dir, val_dir

    det_sources: List[Path] = []
    for p in sorted(raw_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower()==".json") and not p.is_dir():
            continue
        if p.is_dir() and p.name not in {"train", "val", "test"}:
            continue
        if _classify_source(p) == "detection":
            det_sources.append(p)

    if not det_sources:
        raise FileNotFoundError(f"No detection-style JSONs found under {raw_root}")

    train_candidates = [p for p in det_sources if "train" in _norm_path_str(p)]
    val_candidates = [p for p in det_sources if "val" in _norm_path_str(p) or "valid" in _norm_path_str(p)]

    if not train_candidates:
        train_candidates = det_sources
    if not val_candidates:
        val_candidates = det_sources

    train_json = sorted(train_candidates, key=lambda p: (-_score_json_candidate(p), str(p)))[0]
    val_json = sorted(val_candidates, key=lambda p: (-_score_json_candidate(p), str(p)))[0]

    if train_json == val_json:
        for p in det_sources:
            s = _norm_path_str(p)
            if p != train_json and ("val" in s or "valid" in s):
                val_json = p
                break

    if train_json == val_json:
        raise FileNotFoundError(
            "Found detection sources but could not distinguish train and val. "
            f"Candidates: {[str(p) for p in det_sources[:10]]}"
        )
    return train_json, val_json


def locate_image_dirs(raw_root: str | Path) -> Tuple[Path, Path]:
    raw_root = Path(raw_root)

    preferred_train = [
        raw_root / "bdd100k" / "images" / "100k" / "train",
        raw_root / "images" / "100k" / "train",
        raw_root / "100k" / "train",
    ]
    preferred_val = [
        raw_root / "bdd100k" / "images" / "100k" / "val",
        raw_root / "images" / "100k" / "val",
        raw_root / "100k" / "val",
    ]
    train_dir = next((p for p in preferred_train if p.is_dir() and any(x.suffix.lower() in IMAGE_SUFFIXES for x in p.iterdir() if x.is_file())), None)
    val_dir = next((p for p in preferred_val if p.is_dir() and any(x.suffix.lower() in IMAGE_SUFFIXES for x in p.iterdir() if x.is_file())), None)
    if train_dir and val_dir:
        return train_dir, val_dir

    train_dir = None
    val_dir = None
    for p in raw_root.rglob("*"):
        if not p.is_dir():
            continue
        s = _norm_path_str(p)
        has_images = any(x.suffix.lower() in IMAGE_SUFFIXES for x in p.iterdir() if x.is_file())
        if not has_images:
            continue
        if train_dir is None and s.endswith("/train") and ("/images/" in s or "/100k/train" in s):
            train_dir = p
        if val_dir is None and s.endswith("/val") and ("/images/" in s or "/100k/val" in s):
            val_dir = p

    if train_dir is None or val_dir is None:
        raise FileNotFoundError(f"Could not locate image train/val directories under {raw_root}")

    return train_dir, val_dir


def locate_lane_json(raw_root: str | Path) -> Optional[Path]:
    raw_root = Path(raw_root)
    for p in [raw_root / "100k" / "train", raw_root / "bdd100k" / "100k" / "train"]:
        if p.is_dir():
            return p
    candidates: List[Path] = []
    for p in raw_root.rglob("*.json"):
        s = _norm_path_str(p)
        if "lane" in s and ("train" in s or "val" in s or "labels" in s or "poly" in s):
            candidates.append(p)
    if candidates:
        return sorted(candidates, key=lambda p: (0 if "lane" in _norm_path_str(p) else 1, str(p)))[0]
    return None


def locate_seg_maps_root(raw_root: str | Path) -> Optional[Path]:
    raw_root = Path(raw_root)
    for p in raw_root.rglob("*"):
        if not p.is_dir():
            continue
        s = _norm_path_str(p)
        if "/seg/" in s or "seg_maps" in s or (s.endswith("/train") and "labels" in s):
            return p
    return None


def _extract_xywh(obj: dict) -> Optional[Tuple[float, float, float, float]]:
    box2d = obj.get("box2d")
    if not isinstance(box2d, dict):
        return None
    try:
        x1 = float(box2d["x1"])
        y1 = float(box2d["y1"])
        x2 = float(box2d["x2"])
        y2 = float(box2d["y2"])
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return xc, yc, w, h


def convert_detection_json_to_vehicle_yolo(json_path: str | Path, labels_out: str | Path) -> Dict[str, int]:
    labels_out = ensure_dir(labels_out)

    counts: Counter[str] = Counter()
    written = 0
    skipped_no_box = 0

    for item, src_path in _iter_records_from_source(json_path):
        name = _record_name(item, src_path)
        img_w = float(item.get("width", BDD_IMG_W) or BDD_IMG_W)
        img_h = float(item.get("height", BDD_IMG_H) or BDD_IMG_H)
        labels = _extract_object_labels(item)
        rows: List[str] = []

        for lab in labels:
            cat = lab.get("category")
            if cat not in VEHICLE_TO_ID:
                continue
            box = _extract_xywh(lab)
            if box is None:
                skipped_no_box += 1
                continue
            xc, yc, w, h = box
            rows.append(f"{VEHICLE_TO_ID[cat]} {xc / img_w:.6f} {yc / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")
            counts[cat] += 1

        stem = Path(name).stem if name else f"item_{written:06d}"
        (labels_out / f"{stem}.txt").write_text("\n".join(rows), encoding="utf-8")
        written += 1

    return {
        "files_written": written,
        "skipped_no_box": skipped_no_box,
        **{k: int(counts[k]) for k in VEHICLE_CATEGORIES},
    }


def _link_or_copy_images(src_dir: str | Path, dst_dir: str | Path) -> int:
    src_dir = Path(src_dir)
    dst_dir = ensure_dir(dst_dir)
    count = 0
    for p in src_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        dst = dst_dir / p.name
        if dst.exists():
            count += 1
            continue
        try:
            os.symlink(p, dst)
        except Exception:
            shutil.copy2(p, dst)
        count += 1
    return count


def write_vehicle_yaml(dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / "bdd100k_vehicle5.yaml"
    content = "\n".join(
        [
            f"path: {dataset_root}",
            "train: images/train",
            "val: images/val",
            "nc: 5",
            "names:",
            "  0: car",
            "  1: truck",
            "  2: bus",
            "  3: motorcycle",
            "  4: bicycle",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def write_paths_config(
    dataset_root: str | Path,
    raw_root: str | Path,
    lane_json: Optional[str | Path],
    seg_root: Optional[str | Path],
) -> Path:
    dataset_root = Path(dataset_root)
    p = dataset_root / "paths_config.yaml"
    lines = [f"dataset_root: {dataset_root}", f"raw_root: {Path(raw_root)}"]
    lines.append(f"lane_json: {Path(lane_json) if lane_json else ''}")
    lines.append(f"seg_maps_root: {Path(seg_root) if seg_root else ''}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def inspect_download_archives(downloads_dir: str | Path) -> Dict[str, List[str]]:
    downloads_dir = Path(downloads_dir)
    out: Dict[str, List[str]] = {}
    for name in ["bdd100k_labels.zip", "bdd100k_images_100k.zip", "bdd100k_seg_maps.zip"]:
        zp = downloads_dir / name
        out[name] = _zip_members(zp)[:30] if zp.exists() else []
    return out


def rebuild_dualpath_dataset(
    downloads_dir: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    force_reextract: bool = False,
) -> Dict[str, object]:
    downloads_dir = Path(downloads_dir)
    raw_root = ensure_dir(raw_root)
    output_root = ensure_dir(output_root)

    required = {
        "labels": downloads_dir / "bdd100k_labels.zip",
        "images": downloads_dir / "bdd100k_images_100k.zip",
        "seg": downloads_dir / "bdd100k_seg_maps.zip",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required zip files: {missing}")

    for zp in required.values():
        marker = raw_root / f".extracted_{zp.stem}"
        if force_reextract and marker.exists():
            marker.unlink()
        unzip_if_needed(zp, raw_root)

    train_json, val_json = locate_detection_jsons(raw_root)
    train_img_dir, val_img_dir = locate_image_dirs(raw_root)
    lane_json = locate_lane_json(raw_root)
    seg_root = locate_seg_maps_root(raw_root)

    img_train_out = ensure_dir(output_root / "images" / "train")
    img_val_out = ensure_dir(output_root / "images" / "val")
    lbl_train_out = ensure_dir(output_root / "labels" / "train")
    lbl_val_out = ensure_dir(output_root / "labels" / "val")

    train_img_count = _link_or_copy_images(train_img_dir, img_train_out)
    val_img_count = _link_or_copy_images(val_img_dir, img_val_out)
    train_counts = convert_detection_json_to_vehicle_yolo(train_json, lbl_train_out)
    val_counts = convert_detection_json_to_vehicle_yolo(val_json, lbl_val_out)
    yaml_path = write_vehicle_yaml(output_root)
    paths_cfg = write_paths_config(output_root, raw_root, lane_json, seg_root)

    return {
        "train_json": str(train_json),
        "val_json": str(val_json),
        "train_image_dir": str(train_img_dir),
        "val_image_dir": str(val_img_dir),
        "lane_json": str(lane_json) if lane_json else None,
        "seg_maps_root": str(seg_root) if seg_root else None,
        "train_image_count": int(train_img_count),
        "val_image_count": int(val_img_count),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "yaml_path": str(yaml_path),
        "paths_config": str(paths_cfg),
    }
