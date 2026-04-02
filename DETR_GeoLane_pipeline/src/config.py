"""
Centralized path conventions and default configuration.

This module preserves the old YOLO26 Drive conventions so dataset lookup,
checkpointing, and outputs remain compatible with the previously stable
pipeline. It also adds path auto-resolution for Colab local mirrors.
"""

import os
import yaml
from dataclasses import dataclass, asdict
from typing import Optional

ECOCAR_ROOT = "/content/drive/MyDrive/EcoCAR"
DATASET_ROOT = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_yolo")
WEIGHTS_DIR = os.path.join(ECOCAR_ROOT, "weights")
TRAINING_RUNS = os.path.join(ECOCAR_ROOT, "training_runs")
OUTPUTS_DIR = os.path.join(ECOCAR_ROOT, "outputs")
VIDEO_DIR = os.path.join(ECOCAR_ROOT, "video")

LOCAL_DATASET = "/content/bdd100k_yolo"
OLD_PROJECT_LOCAL_DATASET = "/content/bdd100k_yolo"

BDD_LABEL_SEARCH = [
    os.path.join(ECOCAR_ROOT, "datasets", "bdd100k", "labels"),
    os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_labels"),
    os.path.join(ECOCAR_ROOT, "datasets"),
    "/content/drive/MyDrive/bdd100k_labels",
    "/content/bdd100k/labels",
]

BDD_IMG_W, BDD_IMG_H = 1280, 720

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
NUM_CLASSES = len(VEHICLE_CLASSES)
BDD_FULL_CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]
BDD_TO_VEHICLE = {2: 0, 3: 1, 4: 2, 6: 3, 7: 4}
EXPANDED_CLASSES = ["car", "truck", "bus", "train", "motorcycle", "bicycle", "rider"]
BDD_TO_EXPANDED = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 1: 6}

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


def resolve_dataset_root(preferred: str = LOCAL_DATASET) -> str:
    """Match the old project behavior: prefer local mirror, fallback to Drive."""
    candidates = [
        preferred,
        LOCAL_DATASET,
        OLD_PROJECT_LOCAL_DATASET,
        DATASET_ROOT,
    ]
    for root in candidates:
        if not root:
            continue
        if os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "labels")):
            return root
    return preferred


def find_lane_labels(split: str = "train") -> Optional[str]:
    candidates = [
        f"bdd100k_labels_images_{split}.json",
        f"lane_{split}.json",
        f"det_{split}.json",
        os.path.join(split, f"bdd100k_labels_images_{split}.json"),
        os.path.join(split, f"lane_{split}.json"),
    ]
    for base in BDD_LABEL_SEARCH:
        for cand in candidates:
            path = os.path.join(base, cand)
            if os.path.isfile(path):
                return path
    return None


def ensure_dirs(cfg: "Config"):
    for d in [cfg.save_dir, os.path.join(cfg.save_dir, "weights"), WEIGHTS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)


@dataclass
class Config:
    run_name: str = "dualpath_v1"
    device: str = "cuda"
    amp: bool = True
    seed: int = 42

    dataset_root: str = LOCAL_DATASET
    train_lane_json: str = ""
    val_lane_json: str = ""
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
    det_dec_layers: int = 4
    det_dim: int = 256
    det_nhead: int = 8
    det_ffn_dim: int = 1024
    det_dropout: float = 0.0

    lane_num_queries: int = 12
    lane_enc_layers: int = 1
    lane_dec_layers: int = 4
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
    lane_exist_weight: float = 1.5
    lane_pts_weight: float = 3.0
    lane_type_weight: float = 0.5
    lane_curve_weight: float = 5.0
    lane_dir_weight: float = 1.5
    lane_smooth_weight: float = 0.25
    det_task_weight: float = 1.0
    lane_task_weight: float = 1.0

    conf_thresh: float = 0.3
    nms_iou: float = 0.5
    lane_match_thresh: float = 15.0

    temporal_smoothing_alpha: float = 0.70
    temporal_assoc_max_cost: float = 0.10

    save_dir: str = ""
    patience: int = 15

    def __post_init__(self):
        self.dataset_root = resolve_dataset_root(self.dataset_root)
        if not self.train_lane_json:
            self.train_lane_json = find_lane_labels("train") or ""
        if not self.val_lane_json:
            self.val_lane_json = find_lane_labels("val") or ""
        if not self.save_dir:
            self.save_dir = os.path.join(TRAINING_RUNS, self.run_name)

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
