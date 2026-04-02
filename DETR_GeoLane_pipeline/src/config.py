
"""
Centralized path conventions and configuration.
Preserves the old YOLO26 Drive layout and adds notebook07-style lane JSON search.
"""

import os
import yaml
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

ECOCAR_ROOT = "/content/drive/MyDrive/EcoCAR"
DATASET_ROOT = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_yolo")
WEIGHTS_DIR = os.path.join(ECOCAR_ROOT, "weights")
TRAINING_RUNS = os.path.join(ECOCAR_ROOT, "training_runs")
OUTPUTS_DIR = os.path.join(ECOCAR_ROOT, "outputs")
VIDEO_DIR = os.path.join(ECOCAR_ROOT, "video")
LOCAL_DATASET = "/content/bdd100k_yolo"
PATHS_CONFIG_YAML = os.path.join(ECOCAR_ROOT, "paths_config.yaml")
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
    "lane/single white", "lane/single yellow", "lane/single other",
    "lane/double white", "lane/double yellow", "lane/double other",
    "lane/road curb", "lane/crosswalk",
]
LANE_TRAIN_CATS = [c for c in LANE_CATEGORIES if "crosswalk" not in c]
NUM_LANE_TYPES = len(LANE_TRAIN_CATS)
LANE_CAT_TO_ID = {c: i for i, c in enumerate(LANE_TRAIN_CATS)}


def _load_paths_yaml() -> dict:
    if os.path.isfile(PATHS_CONFIG_YAML):
        try:
            with open(PATHS_CONFIG_YAML, 'r') as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def resolve_bdd_raw_dir() -> Tuple[Optional[str], List[str]]:
    """Find the raw BDD directory using notebook07-style logic plus paths_config.yaml."""
    cfg = _load_paths_yaml()
    candidates = []
    for key in [
        'bdd_raw_dir', 'BDD_RAW_DIR', 'bdd100k_raw', 'raw_bdd_dir',
        'raw_dataset_dir', 'raw_dir'
    ]:
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            candidates.append(v)
    candidates += [
        os.path.join(ECOCAR_ROOT, 'datasets', 'bdd100k_raw'),
        os.path.join(ECOCAR_ROOT, 'bdd100k_raw'),
        os.path.join(ECOCAR_ROOT, 'datasets', 'bdd100k'),
        '/content/drive/MyDrive/EcoCAR/datasets/bdd100k_raw',
        '/content/drive/MyDrive/EcoCAR/datasets/bdd100k',
        '/content/bdd100k_raw',
        '/content/bdd100k',
    ]
    tried = []
    for c in candidates:
        c = os.path.normpath(c)
        if c in tried:
            continue
        tried.append(c)
        if os.path.isdir(c):
            return c, tried
    return None, tried


def lane_json_candidates(bdd_raw_dir: Optional[str], split: str) -> List[str]:
    if not split in {'train','val'}:
        raise ValueError(split)
    roots = []
    cfg = _load_paths_yaml()
    for key in ['bdd_raw_dir', 'BDD_RAW_DIR', 'bdd100k_raw', 'raw_bdd_dir']:
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            roots.append(v)
    if bdd_raw_dir:
        roots.append(bdd_raw_dir)
    raw2, _ = resolve_bdd_raw_dir()
    if raw2:
        roots.append(raw2)
    roots += [
        os.path.join(ECOCAR_ROOT, 'datasets', 'bdd100k_raw'),
        os.path.join(ECOCAR_ROOT, 'datasets', 'bdd100k'),
        '/content/bdd100k_raw',
        '/content/bdd100k',
    ]
    uniq_roots, seen = [], set()
    for r in roots:
        if not r:
            continue
        rr = os.path.normpath(r)
        if rr not in seen:
            uniq_roots.append(rr); seen.add(rr)
    cands = []
    for root in uniq_roots:
        cands += [
            os.path.join(root, 'bdd100k', 'labels', 'lane', 'polygons', f'lane_{split}.json'),
            os.path.join(root, 'labels', 'lane', 'polygons', f'lane_{split}.json'),
            os.path.join(root, 'bdd100k', 'labels', f'bdd100k_labels_images_{split}.json'),
            os.path.join(root, 'labels', f'bdd100k_labels_images_{split}.json'),
            os.path.join(root, f'bdd100k_labels_images_{split}.json'),
            os.path.join(root, 'lane', 'polygons', f'lane_{split}.json'),
        ]
    return cands


def find_lane_labels(split: str = 'train', return_tried: bool = False):
    raw, _ = resolve_bdd_raw_dir()
    tried = []
    for p in lane_json_candidates(raw, split):
        if p not in tried:
            tried.append(p)
        if os.path.isfile(p):
            return (p, tried) if return_tried else p
    return (None, tried) if return_tried else None


@dataclass
class Config:
    run_name: str = 'dualpath_v1'
    device: str = 'cuda'
    amp: bool = True
    seed: int = 42
    dataset_root: str = LOCAL_DATASET
    img_size: int = 640
    batch_size: int = 8
    num_workers: int = 4
    max_lanes: int = 10
    lane_points: int = 72
    use_expanded_classes: bool = False
    backbone: str = 'resnet50'
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
    lane_dir_weight: float = 1.5
    lane_smooth_weight: float = 0.5
    det_task_weight: float = 1.0
    lane_task_weight: float = 1.0
    conf_thresh: float = 0.3
    nms_iou: float = 0.5
    lane_match_thresh: float = 15.0
    save_dir: str = ''
    patience: int = 15

    def __post_init__(self):
        if not self.save_dir:
            self.save_dir = os.path.join(TRAINING_RUNS, self.run_name)

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def ensure_dirs(cfg: Config):
    for d in [cfg.save_dir, os.path.join(cfg.save_dir, 'weights'), WEIGHTS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)
