#!/usr/bin/env python3
"""
train.py — Main training entry point for EcoCAR joint perception.

Vehicle detection (nc=5) + lane segmentation with transformer head.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --run-name my_experiment
    python train.py --config configs/default.yaml --epochs 50 --lr 0.001
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_overrides(cfg: dict, args) -> dict:
    """Apply CLI overrides to config."""
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.epochs:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr:
        cfg.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr
    if args.device:
        cfg["device"] = args.device
    if args.warm_start:
        cfg.setdefault("training", {}).setdefault("checkpoint", {})["warm_start"] = args.warm_start
    if args.debug_limit:
        cfg.setdefault("data", {})["debug_limit"] = args.debug_limit
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="EcoCAR Joint Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warm-start", type=str, default=None)
    parser.add_argument("--debug-limit", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args)

    set_seed(cfg.get("seed", 42))

    print(f"Config: {args.config}")
    print(f"Run: {cfg.get('run_name', '?')}")
    print(f"Device: {cfg.get('device', 'cuda')}")

    # ── Build model ───────────────────────────────────────────────────────
    from src.multitask_model import build_multitask_model
    model = build_multitask_model(cfg)
    model.print_summary()

    # Warm start if configured
    warm_start = cfg.get("training", {}).get("checkpoint", {}).get("warm_start")
    if warm_start and os.path.isfile(warm_start):
        model.warm_start_from_checkpoint(warm_start, device=cfg.get("device", "cuda"))

    # ── Build dataset ─────────────────────────────────────────────────────
    from src.data.transforms import JointTransform
    from src.data.dataset import JointBDDDataset, joint_collate_fn
    from torch.utils.data import DataLoader

    data_cfg = cfg.get("data", {})
    root = data_cfg.get("dataset_root", "/content/bdd100k_yolo")

    train_transform = JointTransform(
        img_size=data_cfg.get("img_size", 640),
        mask_height=data_cfg.get("mask_height", 160),
        mask_width=data_cfg.get("mask_width", 160),
        augment=True,
        aug_cfg=data_cfg.get("augmentation", {}),
        lane_target_cfg=data_cfg.get("lane_targets", {}),
    )
    val_transform = JointTransform(
        img_size=data_cfg.get("img_size", 640),
        mask_height=data_cfg.get("mask_height", 160),
        mask_width=data_cfg.get("mask_width", 160),
        augment=False,
    )

    train_ds = JointBDDDataset(
        images_dir=os.path.join(root, "images", "train"),
        labels_dir=os.path.join(root, "labels", "train"),
        masks_dir=os.path.join(root, "masks", "train"),
        transform=train_transform,
        debug_limit=data_cfg.get("debug_limit"),
    )
    val_ds = JointBDDDataset(
        images_dir=os.path.join(root, "images", "val"),
        labels_dir=os.path.join(root, "labels", "val"),
        masks_dir=os.path.join(root, "masks", "val"),
        transform=val_transform,
        debug_limit=data_cfg.get("debug_limit"),
    )

    train_ds.print_summary()
    val_ds.print_summary()

    train_loader = DataLoader(
        train_ds, batch_size=data_cfg.get("batch_size", 16),
        shuffle=True, num_workers=data_cfg.get("num_workers", 4),
        collate_fn=joint_collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg.get("batch_size", 16),
        shuffle=False, num_workers=data_cfg.get("num_workers", 4),
        collate_fn=joint_collate_fn, pin_memory=True,
    )

    # ── Build trainer ─────────────────────────────────────────────────────
    from src.trainers.trainer import JointTrainer
    trainer = JointTrainer(cfg, model, train_loader, val_loader)

    # Resume if specified
    if args.resume and os.path.isfile(args.resume):
        epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {epoch}")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.train()


if __name__ == "__main__":
    main()
