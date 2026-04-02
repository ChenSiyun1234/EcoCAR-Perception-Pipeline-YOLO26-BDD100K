"""
joint_dataset.py — PyTorch Dataset for joint detection + lane segmentation training.

Yields (image, det_targets, lane_mask) per sample.
Detection targets are in YOLO format, lane masks are binary PNGs.

Used by:
  - 08_joint_training.ipynb
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class JointBDDDataset(Dataset):
    """
    Dataset that yields images with both detection labels and lane masks.

    Directory layout expected:
      dataset_root/
        images/train/       (or val)
        labels/train/       YOLO format .txt
        masks/train/        lane mask .png (binary, 0/255)
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        masks_dir: str,
        img_size: int = 640,
        mask_height: int = 180,
        mask_width: int = 320,
        augment: bool = True,
        debug_limit: Optional[int] = None,
    ):
        """
        Args:
            images_dir:  Path to image directory.
            labels_dir:  Path to YOLO-format label directory.
            masks_dir:   Path to lane mask PNG directory.
            img_size:    Target image size (square).
            mask_height: Lane mask height.
            mask_width:  Lane mask width.
            augment:     Whether to apply augmentations.
            debug_limit: If set, use only N samples.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.augment = augment

        # Find all images that have BOTH detection labels and lane masks
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.samples = []
        for img_file in image_files:
            stem = os.path.splitext(img_file)[0]
            label_file = stem + ".txt"
            mask_file = stem + ".png"

            label_path = os.path.join(labels_dir, label_file)
            mask_path = os.path.join(masks_dir, mask_file)

            # Accept image if at least the label exists
            # (mask might be all-black if no lanes in that image, but file should exist)
            if os.path.isfile(label_path) and os.path.isfile(mask_path):
                self.samples.append({
                    "image": os.path.join(images_dir, img_file),
                    "label": label_path,
                    "mask": mask_path,
                })

        if debug_limit is not None:
            self.samples = self.samples[:debug_limit]

        print(f"✅ JointBDDDataset: {len(self.samples)} samples")
        print(f"   Images: {images_dir}")
        print(f"   Labels: {labels_dir}")
        print(f"   Masks:  {masks_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        img = cv2.imread(sample["image"])
        if img is None:
            raise FileNotFoundError(f"Cannot read: {sample['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Load detection labels (YOLO format: class xc yc w h)
        det_targets = self._load_yolo_labels(sample["label"])

        # Load lane mask
        lane_mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
        if lane_mask is None:
            lane_mask = np.zeros((self.mask_height, self.mask_width), dtype=np.uint8)

        # Augmentation: random horizontal flip
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img).copy()
            lane_mask = np.fliplr(lane_mask).copy()
            # Flip detection boxes: x_center becomes 1 - x_center
            if len(det_targets) > 0:
                det_targets[:, 1] = 1.0 - det_targets[:, 1]

        # Resize image to target size
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Resize lane mask to target mask size
        lane_mask = cv2.resize(lane_mask, (self.mask_width, self.mask_height),
                               interpolation=cv2.INTER_NEAREST)

        # Normalise image to [0, 1]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW

        # Binarise lane mask (threshold at 128)
        lane_mask = (lane_mask > 128).astype(np.float32)

        # Convert to tensors
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(lane_mask).unsqueeze(0)  # (1, H, W)

        if len(det_targets) > 0:
            det_tensor = torch.from_numpy(det_targets).float()
        else:
            det_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return {
            "image": img_tensor,
            "det_targets": det_tensor,     # (N, 5): class, xc, yc, w, h
            "lane_mask": mask_tensor,       # (1, mask_h, mask_w): binary
            "image_path": sample["image"],
        }

    def _load_yolo_labels(self, label_path: str) -> np.ndarray:
        """Load YOLO-format labels: class_id x_center y_center width height."""
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = [float(v) for v in parts[1:5]]
                        labels.append([cls_id, xc, yc, w, h])
        if labels:
            return np.array(labels, dtype=np.float32)
        return np.zeros((0, 5), dtype=np.float32)


def joint_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for JointBDDDataset.

    Detection targets have variable numbers of boxes per image,
    so we pad them and add a batch index column.
    """
    images = torch.stack([s["image"] for s in batch])
    lane_masks = torch.stack([s["lane_mask"] for s in batch])
    image_paths = [s["image_path"] for s in batch]

    # Build detection target tensor with batch index
    # Format: (batch_idx, class_id, xc, yc, w, h)
    all_targets = []
    for i, s in enumerate(batch):
        det = s["det_targets"]
        if len(det) > 0:
            batch_idx = torch.full((len(det), 1), i, dtype=torch.float32)
            all_targets.append(torch.cat([batch_idx, det], dim=1))

    if all_targets:
        det_targets = torch.cat(all_targets, dim=0)
    else:
        det_targets = torch.zeros((0, 6), dtype=torch.float32)

    return {
        "images": images,           # (B, 3, H, W)
        "det_targets": det_targets,  # (N_total, 6): batch_idx, cls, xc, yc, w, h
        "lane_masks": lane_masks,    # (B, 1, mask_h, mask_w)
        "image_paths": image_paths,
    }
