"""
Centralized path conventions and default configuration.

All Google Drive paths are defined here so the rest of the codebase
never hardcodes a Drive path. On Colab the Drive is mounted at
/content/drive/MyDrive; locally the paths are unused.
"""

import os
import yaml
from dataclasses import dataclass, asdict
from typing import Optional

# ── Drive path constants (match the existing EcoCAR layout) ──────────
ECOCAR_ROOT      = "/content/drive/MyDrive/EcoCAR"
DATASET_ROOT     = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_yolo")
WEIGHTS_DIR      = os.path.join(ECOCAR_ROOT, "weights")
TRAINING_RUNS    = os.path.join(ECOCAR_ROOT, "training_runs")
OUTPUTS_DIR      = os.path.join(ECOCAR_ROOT, "outputs")
VIDEO_DIR        = os.path.join(ECOCAR_ROOT, "video")
PATHS_CONFIG     = os.path.join(ECOCAR_ROOT, "paths_config.yaml")
BDD_RAW_DIR      = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_raw")

# Local fast‑IO mirror (Colab local SSD — extracted from Drive tars)
LOCAL_DATASET = "/content/bdd100k_yolo"

# ── BDD100K annotation search paths (must include old YOLO26 conventions) ───────
BDD_LABEL_SEARCH = [
    os.path.join(BDD_RAW_DIR, "labels"),
    os.path.join(BDD_RAW_DIR, "bdd100k", "labels"),
    os.path.join(ECOCAR_ROOT, "datasets", "bdd100k", "labels"),
    os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_labels"),
    os.path.join(ECOCAR_ROOT, "datasets"),
    "/content/bdd100k_raw/labels",
    "/content/bdd100k_raw/bdd100k/labels",
    "/content/bdd100k/labels",
]

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

    conf_thresh: float = 0.3
    nms_iou: float = 0.5
    lane_match_thresh: float = 15.0

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


def _from_paths_config(split: str) -> Optional[str]:
    """Mirror the old YOLO26 notebook2 behavior: read paths_config.yaml if present."""
    if not os.path.isfile(PATHS_CONFIG):
        return None
    try:
        with open(PATHS_CONFIG, 'r') as f:
            pcfg = yaml.safe_load(f) or {}
    except Exception:
        return None
    for key, val in pcfg.items():
        if not isinstance(val, str) or not val.endswith('.json') or not os.path.isfile(val):
            continue
        key_l = key.lower()
        base = os.path.basename(val).lower()
        if split in key_l and split in base:
            return val
    return None


def find_lane_labels(split: str = "train") -> Optional[str]:
    """Search for BDD100K consolidated lane label JSON on Drive.

    This intentionally follows the older YOLO26 notebook2/8 conventions:
    1) paths_config.yaml override
    2) bdd100k_raw/labels and bdd100k_raw/bdd100k/labels
    3) older fallback datasets locations
    """
    split = split.lower().strip()
    from_cfg = _from_paths_config(split)
    if from_cfg:
        return from_cfg

    candidates = [
        f"bdd100k_labels_images_{split}.json",
        f"bdd100k_lane_marks_{split}.json",
        f"bdd100k_labels_{split}.json",
        f"det_{split}.json",
        f"lane_{split}.json",
        os.path.join(split, f"bdd100k_labels_images_{split}.json"),
    ]
    for base in BDD_LABEL_SEARCH:
        for cand in candidates:
            path = os.path.join(base, cand)
            if os.path.isfile(path):
                return path
    return None


def ensure_dirs(cfg: Config):
    for d in [cfg.save_dir, os.path.join(cfg.save_dir, "weights"), WEIGHTS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)
