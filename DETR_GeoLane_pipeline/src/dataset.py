
"""BDD100K dataset for dual-path training."""
import os, cv2, numpy as np, torch
from torch.utils.data import Dataset
from typing import Dict, Optional
from .config import BDD_TO_VEHICLE, BDD_TO_EXPANDED
from .lane_targets import LaneLabelCache

class BDD100KDualDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, lane_cache: Optional[LaneLabelCache] = None,
                 img_size: int = 640, max_lanes: int = 10, lane_points: int = 72,
                 use_expanded_classes: bool = False, augment: bool = False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.lane_cache = lane_cache
        self.img_size = img_size
        self.max_lanes = max_lanes
        self.lane_points = lane_points
        self.augment = augment
        self.class_map = BDD_TO_EXPANDED if use_expanded_classes else BDD_TO_VEHICLE
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        print(f'  Dataset: {len(self.image_files)} images from {images_dir}')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.image_files[idx]
        stem = os.path.splitext(fname)[0]
        img = cv2.imread(os.path.join(self.images_dir, fname))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        do_flip = bool(self.augment and (np.random.rand() < 0.5))
        if do_flip:
            img = img[:, ::-1, :].copy()
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        det_targets = self._load_det_labels(os.path.join(self.labels_dir, stem + '.txt'))
        lane_targets = self.lane_cache.get(fname) if self.lane_cache is not None else None
        has_lanes = lane_targets is not None
        if lane_targets is None:
            lane_targets = {
                'existence': np.zeros(self.max_lanes, dtype=np.float32),
                'points': np.zeros((self.max_lanes, self.lane_points, 2), dtype=np.float32),
                'visibility': np.zeros((self.max_lanes, self.lane_points), dtype=np.float32),
                'lane_type': np.zeros(self.max_lanes, dtype=np.int64),
            }
        if do_flip:
            if det_targets.shape[0] > 0:
                det_targets[:,1] = 1.0 - det_targets[:,1]
            lane_targets['points'][:,:,0] = 1.0 - lane_targets['points'][:,:,0]
        return {
            'image': img_tensor,
            'det_targets': torch.from_numpy(det_targets),
            'lane_existence': torch.from_numpy(lane_targets['existence']),
            'lane_points': torch.from_numpy(lane_targets['points']),
            'lane_visibility': torch.from_numpy(lane_targets['visibility']),
            'lane_type': torch.from_numpy(lane_targets['lane_type']),
            'has_lanes': torch.tensor(1.0 if has_lanes else 0.0),
            'image_name': fname,
        }

    def _load_det_labels(self, path: str) -> np.ndarray:
        rows = []
        if not os.path.isfile(path):
            return np.zeros((0,5), dtype=np.float32)
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(float(parts[0])); cx, cy, w, h = [float(x) for x in parts[1:5]]
                except Exception:
                    continue
                if cls_id not in self.class_map or w <= 0 or h <= 0:
                    continue
                rows.append([self.class_map[cls_id], cx, cy, w, h])
        return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0,5), dtype=np.float32)
