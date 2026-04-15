"""
BDD100K dataset loader for vehicle-only detection + lane segmentation.

This version fixes the fragile path assumptions in the initial rebuild:
- it anchors on image files, not lane mask files
- it supports both raw per-image BDD JSON labels and YOLO txt labels
- it supports both /masks and /lane_masks directory names
"""

import json
import numpy as np
import os

from .AutoDriveDataset import AutoDriveDataset
from .convert import id_dict, convert
from tqdm import tqdm


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    @staticmethod
    def _candidate_image_files(img_root):
        files = []
        for name in sorted(os.listdir(img_root)):
            lower = name.lower()
            if lower.endswith((".jpg", ".jpeg", ".png")):
                files.append(name)
        return files

    def _resolve_lane_mask_path(self, stem):
        exact = os.path.join(self.lane_root, stem + '.png')
        if os.path.exists(exact):
            return exact
        matches = sorted([f for f in os.listdir(self.lane_root) if f.startswith(stem) and f.lower().endswith('.png')])
        if matches:
            return os.path.join(self.lane_root, matches[0])
        return None

    def _load_txt_labels(self, txt_path):
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = float(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                labels.append([cls_id, xc, yc, bw, bh])
        if not labels:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(labels, dtype=np.float32)

    def _load_json_labels(self, json_path, width, height):
        gt = []
        with open(json_path, 'r') as f:
            label = json.load(f)
        data = label.get('frames', [{}])[0].get('objects', [])
        data = self.filter_data(data)
        for obj in data:
            category = obj['category']
            if category not in id_dict:
                continue
            box2d = obj.get('box2d', None)
            if box2d is None:
                continue
            x1 = float(box2d['x1'])
            y1 = float(box2d['y1'])
            x2 = float(box2d['x2'])
            y2 = float(box2d['y2'])
            cls_id = float(id_dict[category])
            box = convert((width, height), (x1, x2, y1, y2))
            gt.append([cls_id, box[0], box[1], box[2], box[3]])
        if not gt:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(gt, dtype=np.float32)

    def _load_detection_labels(self, stem, width, height):
        label_format = str(getattr(self.cfg.DATASET, 'LABEL_FORMAT', 'auto')).lower()
        txt_path = os.path.join(self.label_root, stem + '.txt')
        json_path = os.path.join(self.label_root, stem + '.json')

        if label_format == 'txt':
            return self._load_txt_labels(txt_path) if os.path.exists(txt_path) else np.zeros((0, 5), dtype=np.float32)
        if label_format == 'json':
            return self._load_json_labels(json_path, width, height) if os.path.exists(json_path) else np.zeros((0, 5), dtype=np.float32)

        if os.path.exists(txt_path):
            return self._load_txt_labels(txt_path)
        if os.path.exists(json_path):
            return self._load_json_labels(json_path, width, height)
        return np.zeros((0, 5), dtype=np.float32)

    def _get_db(self):
        print('building database...')
        gt_db = []
        height, width = self.shapes

        if not os.path.isdir(self.img_root):
            print(f"ERROR: image directory not found: {self.img_root}")
            return gt_db
        if not os.path.isdir(self.label_root):
            print(f"ERROR: label directory not found: {self.label_root}")
            return gt_db
        if not os.path.isdir(self.lane_root):
            print(f"ERROR: lane directory not found: {self.lane_root}")
            return gt_db

        image_files = self._candidate_image_files(self.img_root)
        missing_lane = 0
        missing_label = 0

        for image_name in tqdm(image_files):
            stem, _ = os.path.splitext(image_name)
            image_path = os.path.join(self.img_root, image_name)
            lane_path = self._resolve_lane_mask_path(stem)
            if lane_path is None:
                missing_lane += 1
                continue

            gt = self._load_detection_labels(stem, width, height)
            if gt.shape[0] == 0:
                txt_path = os.path.join(self.label_root, stem + '.txt')
                json_path = os.path.join(self.label_root, stem + '.json')
                if not os.path.exists(txt_path) and not os.path.exists(json_path):
                    missing_label += 1

            gt_db.append({
                'image': image_path,
                'label': gt,
                'lane': lane_path,
            })

        print(f'database build finish: {len(gt_db)} samples')
        print(f'missing lane masks skipped: {missing_lane}')
        print(f'missing detection labels: {missing_label}')
        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj and obj.get('category') in id_dict:
                remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        pass
