"""
Joint BDD100K dataset for detection + lane segmentation.

Yields synchronized (image, det_targets, lane_mask) with configurable
augmentation and lane target modes.
"""

import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import JointTransform


class JointBDDDataset(Dataset):
    """
    Dataset yielding images with detection labels and lane masks.

    Layout:
        dataset_root/images/{split}/
        dataset_root/labels/{split}/    YOLO .txt
        dataset_root/masks/{split}/     lane .png (binary 0/255)
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        masks_dir: str,
        transform: Optional[JointTransform] = None,
        img_size: int = 640,
        mask_height: int = 180,
        mask_width: int = 320,
        debug_limit: Optional[int] = None,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.masks_dir = masks_dir
        self.transform = transform or JointTransform(
            img_size=img_size, mask_height=mask_height, mask_width=mask_width, augment=False)

        image_files = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )

        self.samples = []
        self.stats = {"total": 0, "has_label": 0, "has_mask": 0, "has_both": 0,
                      "missing_label": 0, "missing_mask": 0, "empty_label": 0}

        for img_file in image_files:
            stem = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            mask_path = os.path.join(masks_dir, stem + ".png")

            has_label = os.path.isfile(label_path)
            has_mask = os.path.isfile(mask_path)

            self.stats["total"] += 1
            if has_label:
                self.stats["has_label"] += 1
            else:
                self.stats["missing_label"] += 1
            if has_mask:
                self.stats["has_mask"] += 1
            else:
                self.stats["missing_mask"] += 1
            if has_label and has_mask:
                self.stats["has_both"] += 1

            # Require at least label to be included
            if has_label and has_mask:
                self.samples.append({
                    "image": os.path.join(images_dir, img_file),
                    "label": label_path,
                    "mask": mask_path,
                })

        if debug_limit is not None:
            self.samples = self.samples[:debug_limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        img = cv2.imread(sample["image"])
        if img is None:
            raise FileNotFoundError(f"Cannot read: {sample['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load detection labels
        det_targets = self._load_yolo_labels(sample["label"])

        # Load lane mask
        lane_mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
        if lane_mask is None:
            lane_mask = np.zeros((self.transform.mask_height, self.transform.mask_width),
                                dtype=np.uint8)

        # Apply synchronized transforms
        img, det_targets, mask = self.transform(img, det_targets, lane_mask)

        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)

        if len(det_targets) > 0:
            det_tensor = torch.from_numpy(det_targets).float()
        else:
            det_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return {
            "image": img_tensor,
            "det_targets": det_tensor,
            "lane_mask": mask_tensor,
            "image_path": sample["image"],
        }

    def _load_yolo_labels(self, label_path: str) -> np.ndarray:
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    labels.append([cls_id, xc, yc, w, h])
        if labels:
            return np.array(labels, dtype=np.float32)
        return np.zeros((0, 5), dtype=np.float32)

    def print_summary(self) -> None:
        s = self.stats
        print(f"JointBDDDataset: {len(self.samples)} usable samples")
        print(f"  Scanned images : {s['total']}")
        print(f"  Has label+mask : {s['has_both']}")
        print(f"  Missing labels : {s['missing_label']}")
        print(f"  Missing masks  : {s['missing_mask']}")


def joint_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate adding batch index to detection targets."""
    images = torch.stack([s["image"] for s in batch])
    lane_masks = torch.stack([s["lane_mask"] for s in batch])
    image_paths = [s["image_path"] for s in batch]

    all_targets = []
    for i, s in enumerate(batch):
        det = s["det_targets"]
        if len(det) > 0:
            batch_idx = torch.full((len(det), 1), i, dtype=torch.float32)
            all_targets.append(torch.cat([batch_idx, det], dim=1))

    det_targets = torch.cat(all_targets, dim=0) if all_targets else torch.zeros((0, 6), dtype=torch.float32)

    return {
        "images": images,
        "det_targets": det_targets,
        "lane_masks": lane_masks,
        "image_paths": image_paths,
    }
