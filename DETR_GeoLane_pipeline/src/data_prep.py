from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

VEHICLE_CATEGORIES = ["car", "truck", "bus", "motorcycle", "bicycle"]
VEHICLE_TO_ID = {name: i for i, name in enumerate(VEHICLE_CATEGORIES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BDD_IMG_W = 1280.0
BDD_IMG_H = 720.0
CATEGORY_ALIASES = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
}


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
    if "/100k/train" in s or s.endswith("/train"):
        score += 6
    if "/100k/val" in s or s.endswith("/val") or "/valid" in s:
        score += 6
    if "detect" in s or "det_" in s or "labels_images" in s:
        score += 3
    if "lane" in s or "seg" in s or "drivable" in s:
        score -= 4
    return score


def _extract_object_labels(record: dict) -> List[dict]:
    if isinstance(record.get("labels"), list):
        return [x for x in record.get("labels", []) if isinstance(x, dict)]
    frames = record.get("frames")
    if isinstance(frames, list) and frames:
        first = frames[0]
        if isinstance(first, dict) and isinstance(first.get("objects"), list):
            return [x for x in first.get("objects", []) if isinstance(x, dict)]
    if isinstance(record.get("objects"), list):
        return [x for x in record.get("objects", []) if isinstance(x, dict)]
    return []


def _record_name(record: dict, src_path: Optional[Path] = None) -> str:
    name = record.get("name")
    if isinstance(name, str) and name:
        return name
    if src_path is not None:
        return src_path.with_suffix(".jpg").name
    return "unknown.jpg"


def _canonical_vehicle_category(cat: object) -> Optional[str]:
    if not isinstance(cat, str):
        return None
    key = cat.strip().lower().replace("_", " ")
    key = key.split("/")[-1]
    key = key.replace(" ", "") if key in {"road curb", "traffic light", "traffic sign"} else key
    key = {
        "roadcurb": "road curb",
    }.get(key, key)
    return CATEGORY_ALIASES.get(key)


def _record_has_detection_objects(record: dict) -> bool:
    labels = _extract_object_labels(record)
    for obj in labels[:200]:
        if _canonical_vehicle_category(obj.get("category")) is not None:
            return True
        if isinstance(obj.get("box2d"), dict):
            return True
    return False


def _classify_json_by_content(path: Path) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if isinstance(data, dict):
        return "detection" if _record_has_detection_objects(data) else None

    if isinstance(data, list) and data:
        sample = data[0]
        if isinstance(sample, dict) and _record_has_detection_objects(sample):
            return "detection"
    return None


def _dir_looks_like_detection_json_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.lower() not in {"train", "val", "valid", "validation", "test"}:
        return False
    jsons = sorted(path.glob("*.json"))
    if len(jsons) < 10:
        return False
    for p in jsons[:10]:
        if _classify_json_by_content(p) == "detection":
            return True
    return False


def locate_detection_jsons(raw_root: str | Path) -> Tuple[Path, Path]:
    raw_root = Path(raw_root)

    # Prefer per-image JSON directories, which is the actual structure in the user's BDD labels zip.
    dir_candidates = [p for p in raw_root.rglob("*") if _dir_looks_like_detection_json_dir(p)]
    train_dirs = [p for p in dir_candidates if "/train" in _norm_path_str(p)]
    val_dirs = [p for p in dir_candidates if "/val" in _norm_path_str(p) or "/valid" in _norm_path_str(p)]
    if train_dirs and val_dirs:
        train_dir = sorted(train_dirs, key=lambda p: (-_score_json_candidate(p), str(p)))[0]
        val_dir = sorted(val_dirs, key=lambda p: (-_score_json_candidate(p), str(p)))[0]
        return train_dir, val_dir

    det_jsons: List[Path] = []
    for p in _find_all_jsons(raw_root):
        if _classify_json_by_content(p) == "detection":
            det_jsons.append(p)

    if not det_jsons:
        raise FileNotFoundError(f"No detection-style JSONs found under {raw_root}")

    train_candidates = [p for p in det_jsons if "train" in _norm_path_str(p)]
    val_candidates = [p for p in det_jsons if "val" in _norm_path_str(p) or "valid" in _norm_path_str(p)]

    if not train_candidates:
        train_candidates = det_jsons
    if not val_candidates:
        val_candidates = det_jsons

    train_json = sorted(train_candidates, key=lambda p: (-_score_json_candidate(p), str(p)))[0]
    val_json = sorted(val_candidates, key=lambda p: (-_score_json_candidate(p), str(p)))[0]

    if train_json == val_json:
        for p in det_jsons:
            s = _norm_path_str(p)
            if p != train_json and ("val" in s or "valid" in s):
                val_json = p
                break

    if train_json == val_json:
        raise FileNotFoundError(
            "Found detection JSONs but could not distinguish train and val. "
            f"Candidates: {[str(p) for p in det_jsons[:10]]}"
        )
    return train_json, val_json


def _has_images(path: Path) -> bool:
    if not path.is_dir():
        return False
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            return True
    return False


def locate_image_dirs(raw_root: str | Path) -> Tuple[Path, Path]:
    raw_root = Path(raw_root)
    train_candidates: List[Path] = []
    val_candidates: List[Path] = []

    for p in raw_root.rglob("*"):
        if not p.is_dir() or not _has_images(p):
            continue
        s = _norm_path_str(p)
        # Prefer official image layouts and avoid JSON label directories.
        score = 0
        if "/images/100k/train" in s:
            score = 10
        elif "/images/100k/val" in s:
            score = 10
        elif "/100k/images/train" in s or "/100k/images/val" in s:
            score = 9
        elif "/images/train" in s or "/images/val" in s:
            score = 8
        elif s.endswith("/train") or s.endswith("/val"):
            score = 4
        if score <= 0:
            continue
        if "/train" in s:
            train_candidates.append((score, p))
        if "/val" in s or "/valid" in s:
            val_candidates.append((score, p))

    if not train_candidates or not val_candidates:
        raise FileNotFoundError(f"Could not locate image train/val directories under {raw_root}")

    train_dir = sorted(train_candidates, key=lambda x: (-x[0], str(x[1])))[0][1]
    val_dir = sorted(val_candidates, key=lambda x: (-x[0], str(x[1])))[0][1]
    return train_dir, val_dir


def locate_lane_json(raw_root: str | Path) -> Optional[Path]:
    raw_root = Path(raw_root)
    # Prefer per-image JSON directory structure used by the user's labels zip.
    for rel in [Path("100k/train"), Path("bdd100k/100k/train")]:
        cand = raw_root / rel
        if cand.is_dir() and len(list(cand.glob("*.json"))) > 10:
            return cand
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
    best: Optional[Path] = None
    best_score = -1
    for p in raw_root.rglob("*"):
        if not p.is_dir():
            continue
        s = _norm_path_str(p)
        score = 0
        if "/color_labels/train" in s:
            score = 10
        elif "/color_labels" in s:
            score = 8
        elif "seg_maps" in s:
            score = 7
        elif "/seg/" in s:
            score = 6
        if score > best_score:
            best = p
            best_score = score
    return best


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


def _iter_records_from_source(source: str | Path) -> Iterator[Tuple[dict, Optional[Path]]]:
    source = Path(source)
    if source.is_dir():
        for p in sorted(source.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if isinstance(data, dict):
                yield data, p
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item, p
        return
    with open(source, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        yield data, source
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item, source


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
            cat = _canonical_vehicle_category(lab.get("category"))
            if cat is None or cat not in VEHICLE_TO_ID:
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
