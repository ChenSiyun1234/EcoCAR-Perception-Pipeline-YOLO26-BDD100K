
"""
Centralized path conventions and default configuration.

All Google Drive paths are defined here so the rest of the codebase
never hardcodes a Drive path.  On Colab the Drive is mounted at
/content/drive/MyDrive; locally the paths are unused.
"""

import os
import yaml
import zipfile
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# ── Drive path constants (match the existing EcoCAR layout) ──────────
ECOCAR_ROOT      = "/content/drive/MyDrive/EcoCAR"
DATASET_ROOT     = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_yolo")
WEIGHTS_DIR      = os.path.join(ECOCAR_ROOT, "weights")
TRAINING_RUNS    = os.path.join(ECOCAR_ROOT, "training_runs")
OUTPUTS_DIR      = os.path.join(ECOCAR_ROOT, "outputs")
VIDEO_DIR        = os.path.join(ECOCAR_ROOT, "video")
DOWNLOADS_DIR    = os.path.join(ECOCAR_ROOT, "downloads")
RAW_DATASET_ROOT = "/content/bdd100k_raw"
PATHS_CONFIG     = os.path.join(ECOCAR_ROOT, "paths_config.yaml")

# Local fast‑IO mirror (Colab local SSD — extracted from Drive tars)
LOCAL_DATASET     = "/content/bdd100k_yolo"

# BDD100K original image size
BDD_IMG_W, BDD_IMG_H = 1280, 720

# ── Vehicle detection classes ────────────────────────────────────────
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
NUM_CLASSES = len(VEHICLE_CLASSES)

BDD_FULL_CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]
BDD_TO_VEHICLE = {2: 0, 3: 1, 4: 2, 6: 3, 7: 4}
EXPANDED_CLASSES = ["car", "truck", "bus", "train", "motorcycle", "bicycle", "rider"]
BDD_TO_EXPANDED = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 1: 6}

# ── Lane categories from BDD100K ────────────────────────────────────
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
NUM_LANE_TYPES = len(LANE_TRAIN_CATS)
LANE_CAT_TO_ID = {c: i for i, c in enumerate(LANE_TRAIN_CATS)}


@dataclass
class Config:
    run_name: str = "dualpath_v1"
    device: str = "cuda"
    amp: bool = True
    seed: int = 42
    dataset_root: str = LOCAL_DATASET
    img_size: int = 640
    batch_size: int = 8
    num_workers: int = 4
    max_lanes: int = 10
    lane_points: int = 72
    use_expanded_classes: bool = False
    backbone: str = "resnet50"
    pretrained: bool = True
    fpn_channels: int = 256
    det_num_queries: int = 100
    det_enc_layers: int = 1
    det_dec_layers: int = 3
    det_dim: int = 256
    det_nhead: int = 8
    det_ffn_dim: int = 1024
    det_dropout: float = 0.0
    lane_num_queries: int = 10
    lane_enc_layers: int = 1
    lane_dec_layers: int = 3
    lane_dim: int = 256
    lane_nhead: int = 8
    lane_ffn_dim: int = 1024
    lane_dropout: float = 0.0
    cross_attn: bool = True
    cross_attn_layers: int = 1
    lr: float = 1e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    epochs: int = 50
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    det_cls_weight: float = 2.0
    det_l1_weight: float = 5.0
    det_giou_weight: float = 2.0
    lane_exist_weight: float = 2.0
    lane_pts_weight: float = 5.0
    lane_type_weight: float = 1.0
    det_task_weight: float = 1.0
    lane_task_weight: float = 1.0
    lane_overlap_weight: float = 2.0
    lane_raster_h: int = 72
    lane_raster_w: int = 128
    lane_raster_thickness: float = 0.03
    conf_thresh: float = 0.3
    nms_iou: float = 0.5
    lane_match_thresh: float = 15.0
    label_schema: str = "auto"
    auto_resume: bool = True
    resume_path: str = ""
    val_interval: int = 1
    max_val_batches: int = 0
    det_task_warmup_weight: float = 1.0
    lane_task_warmup_weight: float = 0.35
    task_warmup_epochs: int = 8
    save_dir: str = ""
    patience: int = 15

    def __post_init__(self):
        if not self.save_dir:
            self.save_dir = os.path.join(TRAINING_RUNS, self.run_name)

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _read_paths_config_for_raw_dir() -> Optional[str]:
    if not os.path.isfile(PATHS_CONFIG):
        return None
    try:
        with open(PATHS_CONFIG, "r") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            for k in ["bdd_raw_dir", "bdd100k_raw", "bdd_root", "bdd100k_root", "raw_bdd100k_dir"]:
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    except Exception:
        return None
    return None


def ensure_bdd_labels_unzipped(force: bool = False) -> Tuple[Optional[str], list]:
    """
    Make sure bdd100k_labels.zip from Drive downloads is extracted.

    Returns:
        raw_root: extracted directory candidate or None
        tried: list of checked paths
    """
    tried = []
    raw_root = _read_paths_config_for_raw_dir() or RAW_DATASET_ROOT
    candidate_zips = [
        os.path.join(DOWNLOADS_DIR, "bdd100k_labels.zip"),
        os.path.join(ECOCAR_ROOT, "downloads", "bdd100k_labels.zip"),
    ]
    for zpath in candidate_zips:
        tried.append(zpath)
        if os.path.isfile(zpath):
            stamp = os.path.join(raw_root, ".labels_unzipped")
            if force or not os.path.exists(stamp):
                os.makedirs(raw_root, exist_ok=True)
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(raw_root)
                with open(stamp, "w") as f:
                    f.write("ok\n")
            return raw_root, tried
    return None, tried


def get_lane_label_candidates(split: str = "train") -> list:
    raw_root = _read_paths_config_for_raw_dir() or RAW_DATASET_ROOT
    cands = [
        os.path.join(raw_root, "100k", split),
        os.path.join(raw_root, "bdd100k", "100k", split),
        os.path.join(raw_root, "bdd100k", "labels", f"bdd100k_labels_images_{split}.json"),
        os.path.join(raw_root, "labels", f"bdd100k_labels_images_{split}.json"),
        os.path.join(raw_root, "bdd100k", "labels", "lane", "polygons", f"lane_{split}.json"),
        os.path.join(raw_root, "labels", "lane", "polygons", f"lane_{split}.json"),
    ]
    seen = set()
    out = []
    for p in cands:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_lane_labels(split: str = "train", auto_extract: bool = True, return_tried: bool = False):
    tried = []
    if auto_extract:
        _, tried_zip = ensure_bdd_labels_unzipped(force=False)
        tried.extend(tried_zip)

    candidates = get_lane_label_candidates(split)
    tried.extend(candidates)
    for path in candidates:
        if os.path.isfile(path) or os.path.isdir(path):
            return (path, tried) if return_tried else path
    return (None, tried) if return_tried else None


def ensure_dirs(cfg: Config):
    for d in [cfg.save_dir, os.path.join(cfg.save_dir, "weights"),
              WEIGHTS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)
