#!/usr/bin/env python3
"""
infer.py — Run joint inference (vehicle detection + lane segmentation).

Usage:
    python infer.py --checkpoint best_joint.pt --images /path/to/images/ --output outputs/
    python infer.py --checkpoint best_joint.pt --images img1.jpg img2.jpg
"""

import argparse
import os

import cv2
import numpy as np
import torch
import yaml

from src.utils.class_map import VEHICLE_CLASSES, NUM_VEHICLE_CLASSES

BOX_COLORS = [
    (60, 180, 255),   # car - blue
    (100, 100, 255),  # truck - purple
    (255, 220, 60),   # bus - yellow
    (100, 255, 100),  # motorcycle - green
    (255, 100, 200),  # bicycle - pink
]


def load_model(ckpt_path: str, cfg: dict, device: str):
    """Load model from checkpoint using arch_config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch_config", {})

    from src.multitask_model import build_multitask_model

    model_cfg = dict(cfg)
    if arch:
        model_cfg.setdefault("model", {}).update({
            "lane_head": {
                "type": arch.get("lane_head_type", "transformer"),
                "embed_dim": arch.get("lane_embed_dim", 128),
                "num_heads": arch.get("lane_num_heads", 4),
                "depth": arch.get("lane_depth", 2),
                "hidden_channels": arch.get("lane_hidden_channels", 64),
            },
        })
        model_cfg.setdefault("data", {}).update({
            "mask_height": arch.get("mask_height", 160),
            "mask_width": arch.get("mask_width", 160),
        })

    model = build_multitask_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    if "ema_state_dict" in ckpt:
        from src.trainers.ema import ModelEMA
        ema = ModelEMA(model, decay=0.9999)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply(model)

    model = model.to(device).eval()
    return model


def run_inference(model, image_path: str, device: str, img_size: int = 640,
                  conf_thresh: float = 0.3):
    """Run inference on a single image."""
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]

    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    det_out = outputs['det_output']

    # Handle YOLO26/v11 end2end output: eval returns (y_postprocessed, preds_dict)
    if isinstance(det_out, (tuple, list)) and len(det_out) == 2:
        y_post = det_out[0]
        if isinstance(y_post, torch.Tensor) and y_post.dim() == 3 and y_post.shape[-1] == 6:
            # Already postprocessed (x1,y1,x2,y2,conf,cls) — filter by conf
            valid = y_post[0][:, 4] > conf_thresh
            bboxes = y_post[0][valid].cpu().numpy()
        else:
            preds = non_max_suppression(y_post, conf_thres=conf_thresh, iou_thres=0.45, max_det=100)
            bboxes = preds[0].cpu().numpy()
    elif isinstance(det_out, torch.Tensor):
        preds = non_max_suppression(det_out, conf_thres=conf_thresh, iou_thres=0.45, max_det=100)
        bboxes = preds[0].cpu().numpy()
    else:
        bboxes = np.empty((0, 6))

    if len(bboxes):
        bboxes[:, [0, 2]] *= (orig_w / img_size)
        bboxes[:, [1, 3]] *= (orig_h / img_size)
        bboxes[:, :4] = bboxes[:, :4].round()

    lane_prob = torch.sigmoid(outputs['lane_logits'])[0, 0].cpu().numpy()

    return img, bboxes, lane_prob


def visualize(img, bboxes, lane_prob, save_path=None, lane_thresh=0.5):
    """Draw detections + lane overlay."""
    h, w = img.shape[:2]
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Lane overlay
    lane_mask = cv2.resize(lane_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    lane_binary = (lane_mask > lane_thresh).astype(np.uint8)
    overlay = vis.copy()
    overlay[lane_binary == 1] = (255, 0, 255)
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

    # Detection boxes
    for box in bboxes:
        x1, y1, x2, y2, conf, cls = box
        cls_id = int(cls)
        if cls_id >= NUM_VEHICLE_CLASSES:
            continue
        color = BOX_COLORS[cls_id]
        label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (int(x1), int(y1) - th - 4), (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(vis, label, (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}")

    return vis


def main():
    parser = argparse.ArgumentParser(description="EcoCAR Joint Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="outputs/inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--lane-thresh", type=float, default=0.5)
    args = parser.parse_args()

    cfg = {}
    if os.path.isfile(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    model = load_model(args.checkpoint, cfg, args.device)

    image_paths = []
    for path in args.images:
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(path, f))
        else:
            image_paths.append(path)

    os.makedirs(args.output, exist_ok=True)

    for img_path in image_paths:
        print(f"Processing: {img_path}")
        img, bboxes, lane_prob = run_inference(
            model, img_path, args.device, conf_thresh=args.conf_thresh)

        save_name = f"joint_{os.path.basename(img_path).replace('.jpg', '.png')}"
        save_path = os.path.join(args.output, save_name)
        visualize(img, bboxes, lane_prob, save_path, lane_thresh=args.lane_thresh)

        lane_pct = (lane_prob > args.lane_thresh).mean() * 100
        print(f"  Vehicles: {len(bboxes)}, Lane coverage: {lane_pct:.1f}%")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
