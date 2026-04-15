"""
BDD100K dataset loader for vehicle-only detection + lane segmentation.
Adapted from YOLOP: iterates over lane mask files (not drivable-area masks),
filters to 5 vehicle classes only.
"""

import numpy as np
import json
import os

from .AutoDriveDataset import AutoDriveDataset
from .convert import id_dict, convert
from tqdm import tqdm


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        Build database from lane mask directory.
        Each record: {'image': path, 'label': np.array, 'lane': path}
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes

        # Iterate over lane mask files as primary anchor
        lane_dir = self.lane_root
        if not lane_dir.exists():
            print(f"WARNING: Lane mask directory not found: {lane_dir}")
            return gt_db

        lane_files = sorted(lane_dir.glob("*.png"))
        for lane_path in tqdm(list(lane_files)):
            lane_path_str = str(lane_path)
            stem = lane_path.stem

            image_path = str(self.img_root / (stem + ".jpg"))
            label_path = str(self.label_root / (stem + ".json"))

            if not os.path.exists(image_path):
                continue

            # Parse detection labels
            gt = np.zeros((0, 5))
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        label = json.load(f)
                    data = label['frames'][0]['objects']
                    data = self.filter_data(data)
                    gt = np.zeros((len(data), 5))
                    for idx, obj in enumerate(data):
                        category = obj['category']
                        if category in id_dict:
                            x1 = float(obj['box2d']['x1'])
                            y1 = float(obj['box2d']['y1'])
                            x2 = float(obj['box2d']['x2'])
                            y2 = float(obj['box2d']['y2'])
                            cls_id = id_dict[category]
                            gt[idx][0] = cls_id
                            box = convert((width, height), (x1, x2, y1, y2))
                            gt[idx][1:] = list(box)
                except Exception:
                    gt = np.zeros((0, 5))

            rec = [{
                'image': image_path,
                'label': gt,
                'lane': lane_path_str
            }]
            gt_db += rec

        print(f'database build finish: {len(gt_db)} samples')
        return gt_db

    def filter_data(self, data):
        """Keep only objects with box2d annotations that are vehicle classes."""
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():
                if obj['category'] in id_dict:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        pass
