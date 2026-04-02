"""
BDD100K dataset for dual-path training.

Preserves the old YOLO26 dataset layout and Drive usage conventions while
adding structured lane targets from BDD100K poly2d.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config, BDD_TO_VEHICLE, BDD_TO_EXPANDED, resolve_dataset_root
from .lane_targets import LaneLabelCache


class BDD100KDualDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str,
                 lane_cache: Optional[LaneLabelCache] = None,
                 img_size: int = 640,
                 max_lanes: int = 10, lane_points: int = 72,
                 use_expanded_classes: bool = False,
                 augment: bool = False):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.lane_cache = lane_cache
        self.img_size = img_size
        self.max_lanes = max_lanes
        self.lane_points = lane_points
        self.augment = augment
        self.class_map = BDD_TO_EXPANDED if use_expanded_classes else BDD_TO_VEHICLE

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"images_dir not found: {images_dir}")
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        self.image_files = image_files
        print(f"  Dataset: {len(self.image_files)} images from {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.image_files[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(self.labels_dir, stem + ".txt")
        det_targets = self._load_det_labels(label_path)

        lane_targets = self.lane_cache.get(fname) if self.lane_cache is not None else None
        if lane_targets is None:
            lane_targets = {
                "existence": np.zeros(self.max_lanes, dtype=np.float32),
                "points": np.zeros((self.max_lanes, self.lane_points, 2), dtype=np.float32),
                "visibility": np.zeros((self.max_lanes, self.lane_points), dtype=np.float32),
                "lane_type": np.zeros(self.max_lanes, dtype=np.int64),
            }
            has_lanes = False
        else:
            lane_targets = {k: np.copy(v) for k, v in lane_targets.items()}
            has_lanes = lane_targets["existence"].sum() > 0

        do_flip = self.augment and (random.random() < 0.5)
        if do_flip:
            img = np.ascontiguousarray(img[:, ::-1, :])
            if det_targets.shape[0] > 0:
                det_targets[:, 1] = 1.0 - det_targets[:, 1]
            if has_lanes:
                lane_targets["points"][:, :, 0] = 1.0 - lane_targets["points"][:, :, 0]
                lane_targets["points"] = lane_targets["points"][:, ::-1, :]
                lane_targets["visibility"] = lane_targets["visibility"][:, ::-1]

        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        return {
            "image": img_tensor,
            "det_targets": torch.from_numpy(det_targets),
            "lane_existence": torch.from_numpy(lane_targets["existence"]),
            "lane_points": torch.from_numpy(lane_targets["points"]),
            "lane_visibility": torch.from_numpy(lane_targets["visibility"]),
            "lane_type": torch.from_numpy(lane_targets["lane_type"]),
            "has_lanes": torch.tensor(1.0 if has_lanes else 0.0),
            "image_name": fname,
        }

    def _load_det_labels(self, path: str) -> np.ndarray:
        if not os.path.isfile(path):
            return np.zeros((0, 5), dtype=np.float32)
        labels = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    bdd_cls = int(parts[0])
                    if bdd_cls not in self.class_map:
                        continue
                    cx, cy, w, h = [float(v) for v in parts[1:5]]
                except Exception:
                    continue
                if w <= 0 or h <= 0:
                    continue
                labels.append([self.class_map[bdd_cls], cx, cy, w, h])
        return np.asarray(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)


def dual_collate_fn(batch: List[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    det_list = []
    for bi, b in enumerate(batch):
        tgt = b["det_targets"]
        if tgt.shape[0] > 0:
            batch_idx = torch.full((tgt.shape[0], 1), bi, dtype=tgt.dtype)
            det_list.append(torch.cat([batch_idx, tgt], dim=1))
    det_targets = torch.cat(det_list) if det_list else torch.zeros((0, 6), dtype=torch.float32)
    return {
        "images": images,
        "det_targets": det_targets,
        "lane_existence": torch.stack([b["lane_existence"] for b in batch]),
        "lane_points": torch.stack([b["lane_points"] for b in batch]),
        "lane_visibility": torch.stack([b["lane_visibility"] for b in batch]),
        "lane_type": torch.stack([b["lane_type"] for b in batch]),
        "has_lanes": torch.stack([b["has_lanes"] for b in batch]),
        "image_names": [b["image_name"] for b in batch],
    }


def build_dataloaders(cfg: Config,
                      train_lane_json: Optional[str] = None,
                      val_lane_json: Optional[str] = None) -> Tuple[DataLoader, DataLoader]:
    root = resolve_dataset_root(cfg.dataset_root)
    train_lane_json = train_lane_json or cfg.train_lane_json or None
    val_lane_json = val_lane_json or cfg.val_lane_json or None

    train_lane_cache = LaneLabelCache(train_lane_json, cfg.max_lanes, cfg.lane_points) if train_lane_json else None
    val_lane_cache = LaneLabelCache(val_lane_json, cfg.max_lanes, cfg.lane_points) if val_lane_json else None

    train_ds = BDD100KDualDataset(
        images_dir=os.path.join(root, "images", "train"),
        labels_dir=os.path.join(root, "labels", "train"),
        lane_cache=train_lane_cache,
        img_size=cfg.img_size,
        max_lanes=cfg.max_lanes,
        lane_points=cfg.lane_points,
        use_expanded_classes=cfg.use_expanded_classes,
        augment=True,
    )
    val_ds = BDD100KDualDataset(
        images_dir=os.path.join(root, "images", "val"),
        labels_dir=os.path.join(root, "labels", "val"),
        lane_cache=val_lane_cache,
        img_size=cfg.img_size,
        max_lanes=cfg.max_lanes,
        lane_points=cfg.lane_points,
        use_expanded_classes=cfg.use_expanded_classes,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=dual_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=dual_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
