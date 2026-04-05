from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

VEHICLE_CATEGORIES = ["car", "truck", "bus", "motorcycle", "bicycle"]
VEHICLE_TO_ID = {name: i for i, name in enumerate(VEHICLE_CATEGORIES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BDD_IMG_W = 1280.0
BDD_IMG_H = 720.0

_RAW_CATEGORY_MAP = {
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


def _dir_has_files_with_suffix(path: Path, suffixes: set[str]) -> bool:
    try:
        for p in path.iterdir():
            if p.is_file() and p.suffix.lower() in suffixes:
                return True
    except Exception:
        return False
    return False


def _dir_has_jsons(path: Path) -> bool:
    return _dir_has_files_with_suffix(path, {".json"})


def _peek_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_object_labels(record: dict) -> List[dict]:
    labels = record.get("labels")
    if isinstance(labels, list):
        return [x for x in labels if isinstance(x, dict)]

    objects = record.get("objects")
    if isinstance(objects, list):
        return [x for x in objects if isinstance(x, dict)]

    frames = record.get("frames")
    if isinstance(frames, list):
        out: List[dict] = []
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            objs = fr.get("objects")
            if isinstance(objs, list):
                out.extend([x for x in objs if isinstance(x, dict)])
        return out
    return []


def _normalize_vehicle_category(cat: object) -> Optional[str]:
    if not isinstance(cat, str):
        return None
    s = cat.strip().lower()
    return _RAW_CATEGORY_MAP.get(s)


def _record_name(record: dict, src_path: Optional[Path] = None) -> str:
    name = record.get("name")
    if isinstance(name, str) and name:
        return name
    if src_path is not None:
        return src_path.with_suffix(".jpg").name
    return "unknown.jpg"


def _iter_records_from_source(source: str | Path) -> Iterator[Tuple[dict, Optional[Path]]]:
    source = Path(source)
    if source.is_dir():
        for p in sorted(source.glob("*.json")):
            try:
                data = _peek_json(p)
            except Exception:
                continue
            if isinstance(data, dict):
                yield data, p
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item, p
    else:
        data = _peek_json(source)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item, source
        elif isinstance(data, dict):
            yield data, source


def _source_looks_like_detection(source: Path) -> bool:
    try:
        for item, _src in _iter_records_from_source(source):
            labels = _extract_object_labels(item)
            if not labels:
                continue
            for lab in labels[:50]:
                if _normalize_vehicle_category(lab.get("category")) is not None:
                    return True
                if isinstance(lab.get("box2d"), dict):
                    return True
            return False
    except Exception:
        return False
    return False


def locate_detection_jsons(raw_root: str | Path) -> Tuple[Path, Path]:
    raw_root = Path(raw_root)

    # Prefer per-image label directories from the user's actual BDD labels zip structure.
    candidate_pairs = [
        (raw_root / "100k" / "train", raw_root / "100k" / "val"),
        (raw_root / "bdd100k" / "100k" / "train", raw_root / "bdd100k" / "100k" / "val"),
        (raw_root / "labels" / "100k" / "train", raw_root / "labels" / "100k" / "val"),
        (raw_root / "bdd100k" / "labels" / "100k" / "train", raw_root / "bdd100k" / "labels" / "100k" / "val"),
    ]
    for train_dir, val_dir in candidate_pairs:
        if train_dir.is_dir() and val_dir.is_dir() and _dir_has_jsons(train_dir) and _dir_has_jsons(val_dir):
            if _source_looks_like_detection(train_dir) and _source_looks_like_detection(val_dir):
                return train_dir, val_dir

    # Fallback to consolidated JSON files if present.
    det_jsons: List[Path] = []
    for p in _find_all_jsons(raw_root):
        if _source_looks_like_detection(p):
            det_jsons.append(p)

    if not det_jsons:
        raise FileNotFoundError(f"No detection-style JSONs found under {raw_root}")

    train_candidates = [p for p in det_jsons if "train" in _norm_path_str(p)]
    val_candidates = [p for p in det_jsons if "val" in _norm_path_str(p) or "valid" in _norm_path_str(p)]
    if not train_candidates:
        train_candidates = det_jsons
    if not val_candidates:
        val_candidates = det_jsons

    train_json = sorted(train_candidates, key=lambda p: str(p))[0]
    val_json = sorted(val_candidates, key=lambda p: str(p))[0]
    if train_json == val_json:
        raise FileNotFoundError(
            "Found detection JSONs but could not distinguish train and val. "
            f"Candidates: {[str(p) for p in det_jsons[:10]]}"
        )
    return train_json, val_json


def locate_image_dirs(raw_root: str | Path) -> Tuple[Path, Path]:
    raw_root = Path(raw_root)
    candidate_pairs = [
        (raw_root / "images" / "100k" / "train", raw_root / "images" / "100k" / "val"),
        (raw_root / "bdd100k" / "images" / "100k" / "train", raw_root / "bdd100k" / "images" / "100k" / "val"),
        (raw_root / "images" / "train", raw_root / "images" / "val"),
        (raw_root / "bdd100k" / "images" / "train", raw_root / "bdd100k" / "images" / "val"),
    ]
    for train_dir, val_dir in candidate_pairs:
        if train_dir.is_dir() and val_dir.is_dir() and _dir_has_files_with_suffix(train_dir, IMAGE_SUFFIXES) and _dir_has_files_with_suffix(val_dir, IMAGE_SUFFIXES):
            return train_dir, val_dir

    # Broad fallback but avoid label JSON directories.
    train_dir: Optional[Path] = None
    val_dir: Optional[Path] = None
    for p in raw_root.rglob("*"):
        if not p.is_dir() or not _dir_has_files_with_suffix(p, IMAGE_SUFFIXES):
            continue
        s = _norm_path_str(p)
        if train_dir is None and s.endswith("/train") and "/images/" in s:
            train_dir = p
        if val_dir is None and s.endswith("/val") and "/images/" in s:
            val_dir = p
    if train_dir is None or val_dir is None:
        raise FileNotFoundError(f"Could not locate image train/val directories under {raw_root}")
    return train_dir, val_dir


def locate_lane_json(raw_root: str | Path) -> Optional[Path]:
    raw_root = Path(raw_root)
    for p in [raw_root / "100k" / "train", raw_root / "bdd100k" / "100k" / "train"]:
        if p.is_dir() and _dir_has_jsons(p):
            return p
    candidates: List[Path] = []
    for p in raw_root.rglob("*.json"):
        s = _norm_path_str(p)
        if "lane" in s and ("train" in s or "val" in s or "labels" in s or "poly" in s):
            candidates.append(p)
    if candidates:
        return sorted(candidates, key=str)[0]
    return None


def locate_seg_maps_root(raw_root: str | Path) -> Optional[Path]:
    raw_root = Path(raw_root)
    candidate_dirs = [
        raw_root / "color_labels" / "train",
        raw_root / "bdd100k" / "color_labels" / "train",
        raw_root / "seg" / "color_labels" / "train",
        raw_root / "bdd100k" / "seg" / "color_labels" / "train",
    ]
    for p in candidate_dirs:
        if p.is_dir():
            return p
    for p in raw_root.rglob("*"):
        if p.is_dir():
            s = _norm_path_str(p)
            if "color_labels" in s and s.endswith("/train"):
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
            mapped_cat = _normalize_vehicle_category(lab.get("category"))
            if mapped_cat not in VEHICLE_TO_ID:
                continue
            box = _extract_xywh(lab)
            if box is None:
                skipped_no_box += 1
                continue
            xc, yc, w, h = box
            rows.append(f"{VEHICLE_TO_ID[mapped_cat]} {xc / img_w:.6f} {yc / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")
            counts[mapped_cat] += 1

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


def pack_output_to_drive(output_root: str | Path, drive_datasets_dir: str | Path) -> Dict[str, str]:
    output_root = Path(output_root)
    drive_datasets_dir = ensure_dir(drive_datasets_dir)
    drive_output_root = drive_datasets_dir / output_root.name
    if drive_output_root.exists():
        shutil.rmtree(drive_output_root)
    shutil.copytree(output_root, drive_output_root)

    tar_path = drive_datasets_dir / f"{output_root.name}.tar"
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w") as tar:
        tar.add(drive_output_root, arcname=drive_output_root.name)
    return {"drive_output_root": str(drive_output_root), "drive_tar_path": str(tar_path)}


def rebuild_dualpath_dataset(
    downloads_dir: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    force_reextract: bool = False,
    drive_datasets_dir: Optional[str | Path] = None,
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

    summary: Dict[str, object] = {
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

    if drive_datasets_dir is not None:
        summary.update(pack_output_to_drive(output_root, drive_datasets_dir))

    return summary
