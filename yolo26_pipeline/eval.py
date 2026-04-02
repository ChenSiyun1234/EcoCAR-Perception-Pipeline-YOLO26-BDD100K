#!/usr/bin/env python3
"""
eval.py — Evaluate a checkpoint on the validation set.

Vehicle detection (nc=5) + lane segmentation metrics.

Usage:
    python eval.py --checkpoint training_runs/vehicle_lane_v4/weights/best_joint.pt
    python eval.py --compare training_runs/
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

from src.utils.class_map import VEHICLE_CLASSES, NUM_VEHICLE_CLASSES


def evaluate_checkpoint(ckpt_path: str, cfg: dict):
    """Load a checkpoint and evaluate on the validation set."""
    device = cfg.get("device", "cuda")
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch_config", {})
    epoch = ckpt.get("epoch", "?")
    print(f"Checkpoint: {ckpt_path}")
    print(f"  Epoch: {epoch}")

    if "metrics" in ckpt:
        saved = {k: f'{v:.4f}' if isinstance(v, float) else v
                 for k, v in ckpt['metrics'].items()}
        print(f"  Saved metrics: {json.dumps(saved, indent=4)}")

    # Build model
    from src.multitask_model import build_multitask_model

    eval_model_cfg = dict(cfg)
    if arch:
        eval_model_cfg.setdefault("model", {}).update({
            "lane_head": {
                "type": arch.get("lane_head_type", "transformer"),
                "embed_dim": arch.get("lane_embed_dim", 128),
                "num_heads": arch.get("lane_num_heads", 4),
                "depth": arch.get("lane_depth", 2),
                "hidden_channels": arch.get("lane_hidden_channels", 64),
            },
        })
        eval_model_cfg.setdefault("data", {}).update({
            "mask_height": arch.get("mask_height", 160),
            "mask_width": arch.get("mask_width", 160),
        })

    model = build_multitask_model(eval_model_cfg)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    if "ema_state_dict" in ckpt:
        from src.trainers.ema import ModelEMA
        ema = ModelEMA(model, decay=0.9999)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply(model)
        print("  Loaded EMA weights")

    model = model.to(device).eval()

    # Build val dataset
    from src.data.transforms import JointTransform
    from src.data.dataset import JointBDDDataset, joint_collate_fn
    from torch.utils.data import DataLoader

    root = data_cfg.get("dataset_root", "/content/bdd100k_yolo")
    val_transform = JointTransform(
        img_size=data_cfg.get("img_size", 640),
        mask_height=data_cfg.get("mask_height", 160),
        mask_width=data_cfg.get("mask_width", 160),
        augment=False,
    )
    val_ds = JointBDDDataset(
        images_dir=os.path.join(root, "images", "val"),
        labels_dir=os.path.join(root, "labels", "val"),
        masks_dir=os.path.join(root, "masks", "val"),
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg.get("batch_size", 16),
        shuffle=False, num_workers=data_cfg.get("num_workers", 4),
        collate_fn=joint_collate_fn, pin_memory=True,
    )

    # Evaluate
    from src.metrics.detection import DetectionMetrics
    from src.metrics.lane import LaneMetrics
    from src.utils.class_map import remap_targets_bdd_to_vehicle

    lane_metrics = LaneMetrics(
        thresholds=eval_cfg.get("lane_thresholds", [0.3, 0.4, 0.5, 0.6, 0.7]))
    det_metrics = DetectionMetrics(device=device)

    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    print(f"\nEvaluating on {len(val_ds)} samples...")

    for batch in val_loader:
        images = batch["images"].to(device)
        det_targets = batch["det_targets"].to(device)
        lane_masks = batch["lane_masks"].to(device)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=cfg.get("amp", True)):
            outputs = model(images)

        lane_metrics.update(outputs["lane_logits"], lane_masks)

        det_out = outputs["det_output"]
        conf_thresh = eval_cfg.get("conf_thresh", 0.001)

        # Handle YOLO26/v11 end2end output: eval returns (y_postprocessed, preds_dict)
        if isinstance(det_out, (tuple, list)) and len(det_out) == 2:
            y_post = det_out[0]
            if isinstance(y_post, torch.Tensor) and y_post.dim() == 3 and y_post.shape[-1] == 6:
                preds = []
                for bi in range(y_post.shape[0]):
                    valid = y_post[bi][:, 4] > conf_thresh
                    preds.append(y_post[bi][valid])
            else:
                preds = non_max_suppression(
                    y_post, conf_thres=conf_thresh,
                    iou_thres=eval_cfg.get("nms_iou_thresh", 0.6),
                    max_det=eval_cfg.get("max_det", 300),
                )
        elif isinstance(det_out, torch.Tensor):
            preds = non_max_suppression(
                det_out, conf_thres=conf_thresh,
                iou_thres=eval_cfg.get("nms_iou_thresh", 0.6),
                max_det=eval_cfg.get("max_det", 300),
            )
        else:
            preds = [torch.empty((0, 6), device=images.device)] * images.shape[0]

        # Remap GT targets to vehicle classes for metric computation
        vehicle_targets = remap_targets_bdd_to_vehicle(det_targets)

        for bi in range(images.shape[0]):
            pred_boxes = preds[bi].cpu()

            if vehicle_targets.shape[0] > 0:
                img_mask = vehicle_targets[:, 0] == bi
                img_gt = vehicle_targets[img_mask]
                if len(img_gt) > 0:
                    gt_cls = img_gt[:, 1].long().cpu()
                    gt_xywh = img_gt[:, 2:6].cpu()
                    _, _, h, w = images.shape
                    gt_boxes = torch.zeros(len(gt_xywh), 4)
                    gt_boxes[:, 0] = (gt_xywh[:, 0] - gt_xywh[:, 2] / 2) * w
                    gt_boxes[:, 1] = (gt_xywh[:, 1] - gt_xywh[:, 3] / 2) * h
                    gt_boxes[:, 2] = (gt_xywh[:, 0] + gt_xywh[:, 2] / 2) * w
                    gt_boxes[:, 3] = (gt_xywh[:, 1] + gt_xywh[:, 3] / 2) * h
                else:
                    gt_boxes = torch.empty((0, 4))
                    gt_cls = torch.empty((0,), dtype=torch.int64)
            else:
                gt_boxes = torch.empty((0, 4))
                gt_cls = torch.empty((0,), dtype=torch.int64)

            det_metrics.update(pred_boxes, gt_boxes, gt_cls)

    lane_results = lane_metrics.compute()
    det_results = det_metrics.compute()

    print(f"\n{'='*55}")
    print(f"  Evaluation Results: {os.path.basename(ckpt_path)}")
    print(f"{'='*55}")
    print(f"  Vehicle mAP@50-95: {det_results.get('det_map50_95', 0)*100:.2f}%")
    print(f"  Vehicle mAP@50:    {det_results.get('det_map50', 0)*100:.2f}%")
    for cls_name in VEHICLE_CLASSES:
        ap = det_results.get(f'ap_{cls_name}', None)
        if ap is not None:
            print(f"    AP {cls_name:<12}: {ap*100:.2f}%")
    print(f"  Lane mIoU:         {lane_results.get('lane_miou', 0)*100:.2f}%")
    print(f"  Lane F1:           {lane_results.get('lane_f1', 0)*100:.2f}%")
    print(f"  Lane best thresh:  {lane_results.get('lane_best_thresh', 0.5)}")
    print(f"  Lane best F1:      {lane_results.get('lane_best_f1', 0)*100:.2f}%")
    print(f"{'='*55}")

    results = {}
    results.update(lane_results)
    results.update(det_results)
    return results


def compare_runs(runs_dir: str):
    """Compare all training runs."""
    print(f"\n{'='*75}")
    print(f"  Run Comparison: {runs_dir}")
    print(f"{'='*75}")

    runs = []
    for name in sorted(os.listdir(runs_dir)):
        summary_path = os.path.join(runs_dir, name, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            summary["_dir"] = name
            runs.append(summary)

    if not runs:
        print("  No runs with summary.json found.")
        return

    header = f"{'Run':<30} {'Epochs':>6} {'Det mAP':>8} {'Lane mIoU':>10} {'Joint':>8}"
    print(header)
    print("-" * len(header))

    for r in runs:
        best = r.get("best_scores", {})
        print(
            f"{r.get('_dir', '?'):<30} "
            f"{r.get('epochs', '?'):>6} "
            f"{best.get('det', 0)*100:>7.1f}% "
            f"{best.get('lane', 0)*100:>9.1f}% "
            f"{best.get('joint', 0):>7.4f}"
        )
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(description="EcoCAR Joint Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--compare", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
        return

    if not args.checkpoint:
        print("Error: --checkpoint or --compare required")
        sys.exit(1)

    cfg = {}
    if os.path.isfile(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    if args.device:
        cfg["device"] = args.device

    evaluate_checkpoint(args.checkpoint, cfg)


if __name__ == "__main__":
    main()
