"""
visualization_utils.py — Drawing, plotting, and feature map visualization.

Used by:
  - 00_quick_pretrained_yolo_baseline.ipynb
  - 02_bdd100k_preparation.ipynb
  - 04_inference_yolo26.ipynb
  - 05_extract_backbone_features.ipynb
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ── Colour palette for BDD100K classes ───────────────────────────────────────
CLASS_COLORS = {
    "person":        (255,  80,  80),   # red
    "rider":         (255, 160,  60),   # orange
    "car":           ( 60, 180, 255),   # blue
    "truck":         (100, 100, 255),   # purple-ish
    "bus":           (255, 220,  60),   # yellow
    "train":         (180, 100, 255),   # violet
    "motorcycle":    (100, 255, 100),   # green
    "bicycle":       (255, 100, 200),   # pink
    "traffic light": (  0, 255, 200),   # cyan
    "traffic sign":  (200, 200,   0),   # olive
}

DEFAULT_COLOR = (200, 200, 200)


def draw_bboxes(
    image: np.ndarray,
    boxes: List[Dict],
    class_names: Optional[Dict[int, str]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image:       BGR numpy array (OpenCV format).
        boxes:       List of dicts with keys: 'class_id', 'x1', 'y1', 'x2', 'y2',
                     and optionally 'confidence', 'class_name'.
        class_names: Optional mapping from class_id → name.
        thickness:   Line thickness.
        font_scale:  Text font scale.

    Returns:
        Image with drawn bounding boxes.
    """
    img = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

        # Determine class name
        name = box.get("class_name", "")
        if not name and class_names and "class_id" in box:
            name = class_names.get(box["class_id"], f"cls_{box['class_id']}")

        # Colour
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label text
        conf = box.get("confidence", None)
        label = name
        if conf is not None:
            label += f" {conf:.2f}"

        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return img


def visualize_predictions(
    image_path: str,
    results,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 8),
) -> np.ndarray:
    """
    Overlay Ultralytics YOLO prediction results on an image.

    Args:
        image_path: Path to the original image.
        results:    Ultralytics Results object (single image).
        save_path:  If set, save annotated image here.
        show:       Whether to display with matplotlib.
        figsize:    Figure size.

    Returns:
        Annotated image as BGR numpy array.
    """
    # Get the plotted result from Ultralytics
    result = results[0] if isinstance(results, list) else results
    annotated = result.plot()  # BGR numpy array

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, annotated)
        print(f"💾 Saved: {save_path}")

    if show:
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(os.path.basename(image_path))
        plt.tight_layout()
        plt.show()

    return annotated


def visualize_yolo_labels(
    image_path: str,
    label_path: str,
    class_names: Dict[int, str],
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """
    Draw YOLO-format labels (normalised) on the corresponding image.

    Args:
        image_path:  Path to image file.
        label_path:  Path to YOLO .txt label file.
        class_names: Dict mapping class_id → name.
        figsize:     Figure size.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return

    h, w = img.shape[:2]

    boxes = []
    if os.path.isfile(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, bw, bh = [float(v) for v in parts[1:5]]

                # Convert YOLO normalised → pixel coords
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                boxes.append({
                    "class_id": cls_id,
                    "class_name": class_names.get(cls_id, f"cls_{cls_id}"),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

    annotated = draw_bboxes(img, boxes, class_names)

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"{os.path.basename(image_path)} — {len(boxes)} boxes")
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(
    features: Dict[str, "torch.Tensor"],
    max_channels: int = 16,
    figsize_per_map: Tuple[int, int] = (12, 3),
) -> None:
    """
    Visualize a grid of feature map channels for each captured layer.

    Args:
        features:        OrderedDict of {name: tensor} from FeatureExtractor.
        max_channels:    Maximum number of channels to display per layer.
        figsize_per_map: Figure size per feature map row.
    """
    import torch

    for name, feat in features.items():
        if feat.dim() < 4:
            print(f"  Skipping {name}: shape {tuple(feat.shape)} (not a spatial feature map)")
            continue

        # Take first batch element
        feat_2d = feat[0].cpu()  # (C, H, W)
        n_channels = min(feat_2d.shape[0], max_channels)

        cols = min(8, n_channels)
        rows = (n_channels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_map[0], figsize_per_map[1] * rows))
        fig.suptitle(f"{name}  —  shape: {tuple(feat.shape)}", fontsize=11)

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for i in range(rows):
            for j in range(cols):
                ch_idx = i * cols + j
                if ch_idx < n_channels:
                    axes[i, j].imshow(feat_2d[ch_idx].numpy(), cmap="viridis")
                    axes[i, j].set_title(f"ch {ch_idx}", fontsize=8)
                axes[i, j].axis("off")

        plt.tight_layout()
        plt.show()


def plot_training_results(results_dir: str, figsize: Tuple[int, int] = (14, 5)) -> None:
    """
    Display training curves from Ultralytics' results.csv.

    Args:
        results_dir: Path to the training run directory (contains results.csv).
        figsize:     Figure size.
    """
    import pandas as pd

    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.isfile(csv_path):
        print(f"❌ results.csv not found in {results_dir}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Box loss
    if "train/box_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train/box_loss"], label="train")
        if "val/box_loss" in df.columns:
            axes[0].plot(df["epoch"], df["val/box_loss"], label="val")
        axes[0].set_title("Box Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

    # Class loss
    if "train/cls_loss" in df.columns:
        axes[1].plot(df["epoch"], df["train/cls_loss"], label="train")
        if "val/cls_loss" in df.columns:
            axes[1].plot(df["epoch"], df["val/cls_loss"], label="val")
        axes[1].set_title("Cls Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

    # mAP
    if "metrics/mAP50(B)" in df.columns:
        axes[2].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    if "metrics/mAP50-95(B)" in df.columns:
        axes[2].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    axes[2].set_title("mAP")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.suptitle("Training Results", fontsize=13)
    plt.tight_layout()
    plt.show()


def save_side_by_side(
    img_original: np.ndarray,
    img_predicted: np.ndarray,
    save_path: str,
    title: str = "",
) -> None:
    """Save a side-by-side comparison of original and predicted images."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(cv2.cvtColor(img_predicted, cv2.COLOR_BGR2RGB))
    ax2.set_title("Predictions")
    ax2.axis("off")
    if title:
        plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"💾 Saved: {save_path}")

def evaluate_lane_iou(
    model, 
    device, 
    val_images_dir: str, 
    val_masks_dir: str, 
    img_size: int = 640, 
    lane_thresh: float = 0.5,
    num_plot_samples: int = 3
) -> float:
    """
    Evaluate Mean Intersection over Union (mIoU) for lane segmentation on the validation set.
    
    Args:
        model: The trained multi-task YOLO model.
        device: 'cuda' or 'cpu'.
        val_images_dir: Path to directory containing validation images.
        val_masks_dir: Path to directory containing ground-truth lane masks.
        img_size: Image size for model input.
        lane_thresh: Probability threshold to binarize lane predictions.
        
    Returns:
        float: The mean IoU across all valid images.
    """
    import torch
    import cv2
    import os
    import numpy as np
    
    model.eval()
    
    if not os.path.exists(val_images_dir) or not os.path.exists(val_masks_dir):
        print(f"❌ Dataset directories not found: {val_images_dir} or {val_masks_dir}")
        return 0.0
        
    image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print("❌ No images found in validation directory.")
        return 0.0
        
    total_iou = 0.0
    valid_samples = 0
    
    print(f"⏳ Evaluating Lane IoU on {len(image_files)} images...")
    
    # Optional dependency for progress bar, fallback to loop if missing
    try:
        from tqdm.notebook import tqdm as tqdm_bar
        iterable = tqdm_bar(image_files, desc="Calculating IoU")
    except ImportError:
        iterable = image_files
        
    for img_name in iterable:
        img_path = os.path.join(val_images_dir, img_name)
        # BDD100K masks are typically .png even if images are .jpg
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(val_masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
            
        # Load and normalise image
        img = cv2.imread(img_path)
        if img is None: continue
        orig_h, orig_w = img.shape[:2]
        
        img_resized = cv2.resize(img, (img_size, img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)
        
        # Load Ground Truth Mask (binary or 255)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None: continue
        # Binarize GT mask
        gt_mask_bin = (gt_mask > 0).astype(np.uint8)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(tensor)
            
        lane_logits = outputs['lane_logits']
        lane_prob = torch.sigmoid(lane_logits)[0, 0].cpu().numpy()  # (mask_h, mask_w)
        
        # Binarize prediction at target mask resolution (180x320)
        pred_bin = (lane_prob > lane_thresh).astype(np.uint8)
        
        # Calculate IoU for this single image
        intersection = np.logical_and(pred_bin, gt_mask_bin).sum()
        union = np.logical_or(pred_bin, gt_mask_bin).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
            
        # --- Visualization for demonstration ---
        if num_plot_samples > 0 and valid_samples < num_plot_samples:
            import matplotlib.pyplot as plt
            
            # Upscale binary masks for visualization on original image
            pred_bin_vis = cv2.resize(pred_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            gt_bin_vis = cv2.resize(gt_mask_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            orig_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            overlay = orig_img_rgb.copy()
            
            # Pred = Red
            overlay[pred_bin_vis == 1] = [255, 0, 0]
            # GT = Green
            overlay[gt_bin_vis == 1] = [0, 255, 0]
            # Overlap = Yellow
            overlap_mask = np.logical_and(pred_bin_vis == 1, gt_bin_vis == 1)
            overlay[overlap_mask] = [255, 255, 0]
            
            blended = cv2.addWeighted(orig_img_rgb, 0.4, overlay, 0.6, 0)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(orig_img_rgb)
            axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(gt_bin_vis, cmap='gray')
            axes[1].set_title("Ground Truth Mask", fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(blended)
            axes[2].set_title(f"Overlay (IoU: {iou*100:.1f}%)\nGreen=GT, Red=Pred, Yellow=Match", fontsize=12)
            axes[2].axis('off')
            
            plt.suptitle(os.path.basename(img_path), fontsize=14)
            plt.tight_layout()
            plt.show()
            
            valid_samples += 1
            
        total_iou += iou
        
    mean_iou = total_iou / valid_samples if valid_samples > 0 else 0.0
    print(f"✅ Validated on {valid_samples} samples.")
    print(f"📊 Mean Intersection-over-Union (mIoU): {mean_iou * 100:.2f}%")
    
    return mean_iou

def evaluate_joint_metrics(
    model, 
    device, 
    val_images_dir: str, 
    val_labels_dir: str, 
    val_masks_dir: str, 
    img_size: int = 640, 
    lane_thresh: float = 0.5,
    conf_thresh: float = 0.001,
    nms_iou_thresh: float = 0.6,
    max_det: int = 300,
):
    """
    Evaluate BOTH Bounding Box mAP (Mean Average Precision) and 
    Lane Segmentation mIoU (Mean Intersection over Union).
    """
    import torch
    import cv2
    import os
    import numpy as np
    
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError as e:
        print(f"❌ Failed to load MeanAveragePrecision: {e}")
        print("Please ensure you have installed it: !pip install -q torchmetrics pycocotools")
        return None
        
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    model.eval()
    
    if not all(os.path.exists(d) for d in [val_images_dir, val_labels_dir, val_masks_dir]):
        print(f"❌ One or more dataset directories not found!")
        return None
        
    image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print("❌ No images found in validation directory.")
        return None
        
    map_metric_full = MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(device)
    map_metric_veh = MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(device)
    
    # Define Vehicle-only subset for BDD100K (2: car, 3: truck, 4: bus, 5: train)
    vehicle_classes = torch.tensor([2, 3, 4, 5], device=device)
    
    total_iou = 0.0
    valid_iou_samples = 0
    valid_det_samples = 0
    
    # Validation Diagnostics
    total_preds = 0
    max_preds = 0
    
    print(f"⏳ Evaluating Joint Metrics on {len(image_files)} images...")
    print(f"   ↳ mAP config: conf_thresh={conf_thresh}, nms_iou_thresh={nms_iou_thresh}, max_det={max_det}")
    
    try:
        from tqdm.notebook import tqdm as tqdm_bar
        iterable = tqdm_bar(image_files, desc="Eval")
    except ImportError:
        iterable = image_files
        
    for img_name in iterable:
        img_path = os.path.join(val_images_dir, img_name)
        
        # Load and normalise image
        img = cv2.imread(img_path)
        if img is None: continue
        orig_h, orig_w = img.shape[:2]
        
        img_resized = cv2.resize(img, (img_size, img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(tensor)
            
        # ==========================================
        # 1. Detection Evaluation (mAP)
        # ==========================================
        det_out = outputs['det_output']
        if isinstance(det_out, (tuple, list)):
            det_out = det_out[0]
            
        preds = non_max_suppression(det_out, conf_thres=conf_thresh, iou_thres=nms_iou_thresh, max_det=max_det)
        bboxes = preds[0].cpu()  # [N, 6] -> [x1, y1, x2, y2, conf, cls]
        
        num_preds = len(bboxes)
        total_preds += num_preds
        if num_preds > max_preds:
            max_preds = num_preds
        
        if len(bboxes):
            bboxes[:, [0, 2]] *= (orig_w / img_size)
            bboxes[:, [1, 3]] *= (orig_h / img_size)
            
        # Load GT labels
        label_path = os.path.join(val_labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        gt_boxes = []
        gt_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c, x_c, y_c, w, h = map(float, parts[:5])
                        x1 = (x_c - w/2) * orig_w
                        y1 = (y_c - h/2) * orig_h
                        x2 = (x_c + w/2) * orig_w
                        y2 = (y_c + h/2) * orig_h
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_labels.append(int(c))
                        
        target_dict = {
            'boxes': torch.tensor(gt_boxes, dtype=torch.float32, device=device) if gt_boxes else torch.empty((0, 4), device=device),
            'labels': torch.tensor(gt_labels, dtype=torch.int64, device=device) if gt_labels else torch.empty((0,), dtype=torch.int64, device=device)
        }
        
        pred_dict = {
            'boxes': bboxes[:, :4].to(device) if len(bboxes) else torch.empty((0, 4), device=device),
            'scores': bboxes[:, 4].to(device) if len(bboxes) else torch.empty((0,), device=device),
            'labels': bboxes[:, 5].to(torch.int64).to(device) if len(bboxes) else torch.empty((0,), dtype=torch.int64, device=device)
        }
        
        # Vehicle-only filtering
        target_dict_veh = {'boxes': torch.empty((0, 4), device=device), 'labels': torch.empty((0,), dtype=torch.int64, device=device)}
        if target_dict['labels'].shape[0] > 0:
            gt_veh_mask = torch.isin(target_dict['labels'], vehicle_classes)
            if gt_veh_mask.any():
                target_dict_veh['boxes'] = target_dict['boxes'][gt_veh_mask]
                target_dict_veh['labels'] = target_dict['labels'][gt_veh_mask]
                
        pred_dict_veh = {'boxes': torch.empty((0, 4), device=device), 'scores': torch.empty((0,), device=device), 'labels': torch.empty((0,), dtype=torch.int64, device=device)}
        if pred_dict['labels'].shape[0] > 0:
            pred_veh_mask = torch.isin(pred_dict['labels'], vehicle_classes)
            if pred_veh_mask.any():
                pred_dict_veh['boxes'] = pred_dict['boxes'][pred_veh_mask]
                pred_dict_veh['scores'] = pred_dict['scores'][pred_veh_mask]
                pred_dict_veh['labels'] = pred_dict['labels'][pred_veh_mask]
        
        map_metric_full.update([pred_dict], [target_dict])
        map_metric_veh.update([pred_dict_veh], [target_dict_veh])
        valid_det_samples += 1
        
        # ==========================================
        # 2. Lane Segmentation Evaluation (mIoU)
        # ==========================================
        mask_path = os.path.join(val_masks_dir, img_name.replace('.jpg', '.png'))
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask_bin = (gt_mask > 0).astype(np.uint8)
                
                lane_logits = outputs['lane_logits']
                lane_prob = torch.sigmoid(lane_logits)[0, 0].cpu().numpy()
                pred_bin = (lane_prob > lane_thresh).astype(np.uint8)
                
                intersection = np.logical_and(pred_bin, gt_mask_bin).sum()
                union = np.logical_or(pred_bin, gt_mask_bin).sum()
                
                iou = (intersection / union) if union > 0 else (1.0 if intersection == 0 else 0.0)
                total_iou += iou
                valid_iou_samples += 1
                
    # Compute Final Metrics
    avg_preds = total_preds / max(1, valid_det_samples)
    print(f"\\n✅ Validated on {valid_det_samples} images:")
    print(f"   - Box Diagnostics: {avg_preds:.1f} avg preds/img (Max: {max_preds})")
    print(f"   - Valid mask samples: {valid_iou_samples}")
    
    print("\\n📈 Calculating final COCO mAP integrals (this may take a moment)...")
    map_results_full = map_metric_full.compute()
    map50_95_full = map_results_full['map'].item()
    map50_full = map_results_full['map_50'].item()
    
    map_results_veh = map_metric_veh.compute()
    map50_95_veh = map_results_veh['map'].item()
    map50_veh = map_results_veh['map_50'].item()
    
    mean_iou = total_iou / valid_iou_samples if valid_iou_samples > 0 else 0.0
    
    print("="*50)
    print(" JOINT EVALUATION RESULTS")
    print("="*50)
    print(f" 🎯 Full Bounding Box mAP@50-95:         {map50_95_full * 100:>6.2f}%")
    print(f" 🎯 Full Bounding Box mAP@50:            {map50_full * 100:>6.2f}%")
    print("-" * 50)
    print(f" 🚗 Vehicle-only Bounding Box mAP@50-95: {map50_95_veh * 100:>6.2f}%")
    print(f" 🚗 Vehicle-only Bounding Box mAP@50:    {map50_veh * 100:>6.2f}%")
    print(f"    (Vehicle classes: car, truck, bus, train)")
    print("-" * 50)
    print(f" 🛣️  Lane Segmentation mIoU:             {mean_iou * 100:>6.2f}%")
    print("="*50)
    
    return {
        'mAP@50-95_full': map50_95_full,
        'mAP@50_full': map50_full,
        'mAP@50-95_veh': map50_95_veh,
        'mAP@50_veh': map50_veh,
        'mIoU': mean_iou
    }
