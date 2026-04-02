"""Script to create/update notebooks 08, 09, 10, 11 for the v4 architecture."""
import json
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')


def make_nb():
    return {
        'cells': [],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12.0'}
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def md(nb, cell_id, text):
    lines = text.strip().split('\n')
    nb['cells'].append({
        'cell_type': 'markdown', 'id': cell_id, 'metadata': {},
        'source': [line + '\n' for line in lines[:-1]] + [lines[-1]],
    })


def code(nb, cell_id, text):
    lines = text.strip().split('\n')
    nb['cells'].append({
        'cell_type': 'code', 'execution_count': None, 'id': cell_id,
        'metadata': {}, 'outputs': [],
        'source': [line + '\n' for line in lines[:-1]] + [lines[-1]],
    })


def save_nb(nb, name):
    path = os.path.join(NOTEBOOKS_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Wrote {path}')


# ═══════════════════════════════════════════════════════════════════════
# NB08 — Joint Training (v4 architecture)
# ═══════════════════════════════════════════════════════════════════════
def create_nb08():
    nb = make_nb()

    md(nb, 'c0', """# 08 — Joint Training: Vehicle Detection + Lane Segmentation (v4)

**Goal:** Train a joint YOLO26-S model with:
- Native 5-class vehicle detection head (car, truck, bus, motorcycle, bicycle)
- Transformer-based lane segmentation head (SRA attention)
- Shared backbone/neck, no DualTaskNeck

**Architecture (v4):**
- Backbone: YOLO26-S (pretrained on COCO, detection head replaced with nc=5)
- Lane head: LightMUSTER transformer (embed_dim=128, depth=2, SRA)
- Loss: fixed weighting (det=1.0, lane=0.3)
- Masks: 160x160 square (matching 640x640 input)""")

    md(nb, 'c1', '## 1 — Setup')

    code(nb, 'c2', '!pip install -q ultralytics>=8.4.0 opencv-python matplotlib pyyaml tqdm\n!pip install -q "torchmetrics[detection]"')

    code(nb, 'c3', """import torch

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({mem:.1f} GB)")
else:
    print("WARNING: No GPU detected — training will be very slow.")""")

    code(nb, 'c4', """from google.colab import drive
drive.mount('/content/drive')

import os

ECOCAR_ROOT = "/content/drive/MyDrive/EcoCAR"
DATASET_DIR = "/content/bdd100k_yolo"
print(f"EcoCAR root: {ECOCAR_ROOT}")
print(f"Dataset dir: {DATASET_DIR}")""")

    code(nb, 'c5', """import sys

# Use the project directory on Drive for imports
PROJECT_DIR = os.path.join(ECOCAR_ROOT, "project")
assert os.path.isdir(PROJECT_DIR), f"Project not found at {PROJECT_DIR}"
sys.path.insert(0, PROJECT_DIR)

from src.utils.class_map import VEHICLE_CLASSES, NUM_VEHICLE_CLASSES
print(f"Vehicle classes ({NUM_VEHICLE_CLASSES}): {VEHICLE_CLASSES}")""")

    md(nb, 'c6', """## 2 — Configuration

Key design choices:
- **Native nc=5** detection head (no COCO-80 remapping)
- **Square masks** (160x160) match the 640x640 input
- **Transformer lane head** (LightMUSTER with SRA)
- **Fixed loss weights** (det=1.0, lane=0.3) — no staged weighting needed
- **SGD** with low LR for stable fine-tuning""")

    code(nb, 'c7', """import yaml

# Detection-only checkpoint from Notebook 03 (warm start)
DET_CKPT = f"{ECOCAR_ROOT}/weights/bdd100k_yolo26s_vehicle_best.pt"

cfg = {
    "run_name": "vehicle_lane_v4",
    "device": "cuda",
    "amp": True,

    "model": {
        "backbone_weights": "yolo26s.pt",
        "lane_head": {
            "type": "transformer",
            "embed_dim": 128,
            "num_heads": 4,
            "depth": 2,
            "hidden_channels": 64,
        },
    },

    "data": {
        "dataset_root": DATASET_DIR,
        "img_size": 640,
        "mask_height": 160,
        "mask_width": 160,
        "batch_size": 16,
        "num_workers": 4,
    },

    "training": {
        "epochs": 40,
        "patience": 8,
    },

    "optimizer": {
        "type": "sgd",
        "lr": 0.002,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "backbone_lr_scale": 0.1,
    },

    "scheduler": {
        "type": "cosine",
        "warmup_epochs": 3,
        "min_lr_ratio": 0.01,
    },

    "loss": {
        "lane": {
            "type": "bce_dice",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
        },
        "multitask": {
            "strategy": "fixed",
            "det_weight": 1.0,
            "lane_weight": 0.3,
        },
        "detection_preservation": {
            "enabled": False,
        },
    },

    "eval": {
        "conf_thresh": 0.001,
        "nms_iou_thresh": 0.6,
        "max_det": 300,
        "lane_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
    },
}

SAVE_DIR = f"{ECOCAR_ROOT}/training_runs/{cfg['run_name']}"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save config
cfg_path = os.path.join(SAVE_DIR, "config.yaml")
with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config saved: {cfg_path}")
print(f"Save dir:    {SAVE_DIR}")""")

    md(nb, 'c8', '## 3 — Build Model')

    code(nb, 'c9', """from src.multitask_model import build_multitask_model

model = build_multitask_model(cfg)
model.print_summary()""")

    code(nb, 'c10', """# Warm start from detection-only checkpoint
if os.path.isfile(DET_CKPT):
    loaded = model.warm_start_from_checkpoint(DET_CKPT)
    print(f"Warm start: loaded {loaded} keys from {DET_CKPT}")
else:
    print(f"No detection checkpoint found at {DET_CKPT}")
    print("Training from COCO pretrained backbone only.")""")

    code(nb, 'c11', """# Forward-pass sanity check
dummy = torch.randn(1, 3, 640, 640)
if torch.cuda.is_available():
    dummy = dummy.cuda()
    model = model.cuda()

with torch.no_grad():
    out = model(dummy)

print("Output keys:", list(out.keys()))
det = out['det_output']
if isinstance(det, torch.Tensor):
    print(f"  det_output shape: {det.shape}")
elif isinstance(det, (list, tuple)):
    print(f"  det_output: {type(det).__name__} with {len(det)} elements")
    for j, d in enumerate(det):
        if hasattr(d, 'shape'):
            print(f"    [{j}] {d.shape}")
        else:
            print(f"    [{j}] {type(d).__name__}: {d}")
else:
    print(f"  det_output: {type(det)}")
print(f"  lane_logits shape: {out['lane_logits'].shape}")
print(f"  neck_features: {len(out['neck_features'])} scales")
model = model.cpu()
torch.cuda.empty_cache()""")

    md(nb, 'c12', '## 4 — Dataset')

    code(nb, 'c13', """import tarfile

os.makedirs(DATASET_DIR, exist_ok=True)

# 1. Images + Labels from NB02 tar
NB02_TAR = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_yolo_nb02.tar")
if not os.path.isdir(os.path.join(DATASET_DIR, "images", "val")):
    assert os.path.isfile(NB02_TAR), f"NB02 tar not found: {NB02_TAR}\\nRun Notebook 02 first."
    print(f"Extracting {NB02_TAR} ...")
    with tarfile.open(NB02_TAR, "r") as tar:
        tar.extractall(DATASET_DIR, filter='data')
    print("Done.")
else:
    print("Images/labels already extracted.")

# 2. Lane masks from NB07 tar
MASKS_TAR = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_lane_masks.tar")
if not os.path.isdir(os.path.join(DATASET_DIR, "masks", "val")):
    assert os.path.isfile(MASKS_TAR), f"Masks tar not found: {MASKS_TAR}\\nRun Notebook 07 first."
    print(f"Extracting {MASKS_TAR} ...")
    with tarfile.open(MASKS_TAR, "r") as tar:
        tar.extractall(DATASET_DIR, filter='data')
    print("Done.")
else:
    print("Masks already extracted.")

# Quick check
for split in ["train", "val"]:
    n_img = len(os.listdir(os.path.join(DATASET_DIR, "images", split)))
    n_lbl = len(os.listdir(os.path.join(DATASET_DIR, "labels", split)))
    mask_dir = os.path.join(DATASET_DIR, "masks", split)
    n_mask = len(os.listdir(mask_dir)) if os.path.isdir(mask_dir) else 0
    print(f"  {split}: {n_img} images, {n_lbl} labels, {n_mask} masks")""")

    code(nb, 'c14', """from src.data.transforms import JointTransform
from src.data.dataset import JointBDDDataset, joint_collate_fn
from torch.utils.data import DataLoader

data_cfg = cfg["data"]
root = data_cfg["dataset_root"]

train_transform = JointTransform(
    img_size=data_cfg["img_size"],
    mask_height=data_cfg["mask_height"],
    mask_width=data_cfg["mask_width"],
    augment=True,
)
val_transform = JointTransform(
    img_size=data_cfg["img_size"],
    mask_height=data_cfg["mask_height"],
    mask_width=data_cfg["mask_width"],
    augment=False,
)

train_ds = JointBDDDataset(
    images_dir=os.path.join(root, "images", "train"),
    labels_dir=os.path.join(root, "labels", "train"),
    masks_dir=os.path.join(root, "masks", "train"),
    transform=train_transform,
)
val_ds = JointBDDDataset(
    images_dir=os.path.join(root, "images", "val"),
    labels_dir=os.path.join(root, "labels", "val"),
    masks_dir=os.path.join(root, "masks", "val"),
    transform=val_transform,
)

train_loader = DataLoader(
    train_ds, batch_size=data_cfg["batch_size"], shuffle=True,
    num_workers=data_cfg["num_workers"], collate_fn=joint_collate_fn,
    pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_ds, batch_size=data_cfg["batch_size"], shuffle=False,
    num_workers=data_cfg["num_workers"], collate_fn=joint_collate_fn,
    pin_memory=True,
)

print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")""")

    md(nb, 'c15', '## 5 — Sanity Check Batch')

    code(nb, 'c16', """import matplotlib.pyplot as plt
import numpy as np

batch = next(iter(train_loader))

print(f"Images:      {batch['images'].shape}")
print(f"Det targets: {batch['det_targets'].shape}")
print(f"Lane masks:  {batch['lane_masks'].shape}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(min(4, batch['images'].shape[0])):
    img = batch['images'][i].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    mask = batch['lane_masks'][i, 0].numpy()

    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Image {i}")
    axes[0, i].axis('off')

    axes[1, i].imshow(mask, cmap='magma')
    axes[1, i].set_title(f"Lane mask {i}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()""")

    md(nb, 'c17', """## 6 — Train

The `JointTrainer` handles:
- Fixed multi-task loss (det_weight=1.0, lane_weight=0.3)
- EMA-based evaluation
- Metrics logging and best checkpoint selection""")

    code(nb, 'c18', """from src.trainers.trainer import JointTrainer

trainer = JointTrainer(cfg, model, train_loader, val_loader)
trainer.save_dir = SAVE_DIR

print(f"Training for {cfg['training']['epochs']} epochs")
print(f"Save dir: {SAVE_DIR}")""")

    md(nb, 'c19', '## 7 — Training Results')

    code(nb, 'c20', """import traceback

try:
    history = trainer.train()
except Exception as e:
    print(f"\\nTraining stopped: {e}")
    traceback.print_exc(limit=3)
    history = trainer.history

print(f"\\nTraining complete. {len(history)} epochs logged.")""")

    code(nb, 'c21', """import matplotlib.pyplot as plt

if history and len(history) > 0:
    epochs_range = range(1, len(history) + 1)

    # Extract metrics from list of epoch dicts
    train_losses = [h.get('train_loss', 0) for h in history]
    det_map50 = [h.get('det_map50', 0) for h in history]
    det_map50_95 = [h.get('det_map50_95', 0) for h in history]
    lane_miou = [h.get('lane_miou', 0) for h in history]
    lane_f1 = [h.get('lane_f1', 0) for h in history]
    val_losses = [h.get('val_loss', 0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs_range, train_losses, label='Train')
    axes[0].plot(epochs_range, val_losses, label='Val')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    # Detection mAP
    axes[1].plot(epochs_range, [v*100 for v in det_map50], label='mAP@50')
    axes[1].plot(epochs_range, [v*100 for v in det_map50_95], label='mAP@50-95')
    axes[1].set_title('Detection mAP (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    # Lane metrics
    axes[2].plot(epochs_range, [v*100 for v in lane_miou], label='mIoU')
    axes[2].plot(epochs_range, [v*100 for v in lane_f1], label='F1')
    axes[2].set_title('Lane Metrics (%)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
    plt.show()
else:
    print("No training history available.")""")

    md(nb, 'c22', '## 8 — Save Checkpoint')

    code(nb, 'c23', """import shutil

# Copy best checkpoint to weights dir
weights_dir = os.path.join(SAVE_DIR, "weights")
best_src = os.path.join(weights_dir, "best_joint.pt")
backup_dir = os.path.join(ECOCAR_ROOT, "weights")
os.makedirs(backup_dir, exist_ok=True)

if os.path.isfile(best_src):
    dst = os.path.join(backup_dir, "vehicle_lane_v4_best.pt")
    shutil.copy2(best_src, dst)
    print(f"Best checkpoint copied to: {dst}")
else:
    print(f"best_joint.pt not found in {weights_dir}")
    # Try last checkpoint
    last_src = os.path.join(weights_dir, "last.pt")
    if os.path.isfile(last_src):
        dst = os.path.join(backup_dir, "vehicle_lane_v4_last.pt")
        shutil.copy2(last_src, dst)
        print(f"Last checkpoint copied to: {dst}")

# Save summary
import json
summary = {
    "run_name": cfg["run_name"],
    "epochs": cfg["training"]["epochs"],
    "architecture": "v4_native_vehicle_transformer_lane",
    "nc": 5,
    "classes": VEHICLE_CLASSES,
    "mask_size": [cfg["data"]["mask_height"], cfg["data"]["mask_width"]],
}
if history and len(history) > 0:
    best_det = max(h.get('det_map50_95', 0) for h in history)
    best_lane = max(h.get('lane_miou', 0) for h in history)
    summary["best_scores"] = {
        "det": best_det,
        "lane": best_lane,
        "joint": best_det * 0.6 + best_lane * 0.4,
    }

summary_path = os.path.join(SAVE_DIR, "summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved: {summary_path}")
print("\\n" + "=" * 55)
print(" JOINT TRAINING COMPLETE (v4)")
print(f" Classes: {', '.join(VEHICLE_CLASSES)}")
if 'best_scores' in summary:
    print(f" Best det mAP@50-95: {summary['best_scores']['det']*100:.1f}%")
    print(f" Best lane mIoU:     {summary['best_scores']['lane']*100:.1f}%")
print(f" Checkpoint: {backup_dir}")
print("=" * 55)""")

    save_nb(nb, '08_joint_training.ipynb')


# ═══════════════════════════════════════════════════════════════════════
# NB09 — Joint Inference & Evaluation (v4)
# ═══════════════════════════════════════════════════════════════════════
def create_nb09():
    nb = make_nb()

    md(nb, 'c0', """# 09 — Joint Inference & Evaluation (v4)

**Goal:** Evaluate and visualize the joint vehicle detection + lane segmentation model.

Features:
- Qualitative inference with detection boxes + lane overlay
- Full validation set evaluation (mAP + lane IoU/F1)
- Per-class AP breakdown for vehicle classes
- Lane threshold sensitivity analysis""")

    md(nb, 'c1', '## 1 — Setup')

    code(nb, 'c2', '!pip install -q ultralytics>=8.4.0 opencv-python matplotlib pyyaml\n!pip install -q "torchmetrics[detection]"')

    code(nb, 'c3', """from google.colab import drive
drive.mount('/content/drive')

import os, sys, random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

ECOCAR_ROOT = "/content/drive/MyDrive/EcoCAR"
DATASET_DIR = "/content/bdd100k_yolo"
JOINT_WEIGHTS = f"{ECOCAR_ROOT}/weights/vehicle_lane_v4_best.pt"

PROJECT_DIR = os.path.join(ECOCAR_ROOT, "project")
assert os.path.isdir(PROJECT_DIR), f"Project not found at {PROJECT_DIR}"
sys.path.insert(0, PROJECT_DIR)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Joint weights: {JOINT_WEIGHTS}")""")

    md(nb, 'c4', '## 2 — Load Model')

    code(nb, 'c5', """from src.multitask_model import build_multitask_model
from src.utils.class_map import VEHICLE_CLASSES, NUM_VEHICLE_CLASSES

assert os.path.isfile(JOINT_WEIGHTS), f"Checkpoint not found: {JOINT_WEIGHTS}"

ckpt = torch.load(JOINT_WEIGHTS, map_location='cuda', weights_only=False)
arch = ckpt.get('arch_config', {})
epoch = ckpt.get('epoch', '?')
print(f"Checkpoint epoch: {epoch}")
if 'metrics' in ckpt:
    for k, v in ckpt['metrics'].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

model = build_multitask_model(
    weights='yolo26s.pt',
    lane_head_type=arch.get('lane_head_type', 'transformer'),
    lane_embed_dim=arch.get('lane_embed_dim', 128),
    lane_num_heads=arch.get('lane_num_heads', 4),
    lane_depth=arch.get('lane_depth', 2),
    mask_height=arch.get('mask_height', 160),
    mask_width=arch.get('mask_width', 160),
)

missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

if 'ema_state_dict' in ckpt:
    from src.trainers.ema import ModelEMA
    ema = ModelEMA(model, decay=0.9999)
    ema.load_state_dict(ckpt['ema_state_dict'])
    ema.apply(model)
    print("Loaded EMA weights")

model = model.to('cuda').eval()
print("Model loaded.")""")

    md(nb, 'c6', '## 3 — Qualitative Inference')

    code(nb, 'c7', """BOX_COLORS = [
    (60, 180, 255),   # car - blue
    (100, 100, 255),  # truck - purple
    (255, 220, 60),   # bus - yellow
    (100, 255, 100),  # motorcycle - green
    (255, 100, 200),  # bicycle - pink
]

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

IMG_SIZE = 640
CONF_THRESH = 0.3
LANE_THRESH = 0.5

def run_joint_inference(model, img_bgr, device='cuda'):
    orig_h, orig_w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(tensor)

    det_out = out['det_output']

    # Handle YOLO26 end2end output: eval returns (y_postprocessed, preds_dict)
    if isinstance(det_out, (tuple, list)) and len(det_out) == 2:
        y_post = det_out[0]
        if isinstance(y_post, torch.Tensor) and y_post.dim() == 3 and y_post.shape[-1] == 6:
            valid = y_post[0][:, 4] > CONF_THRESH
            bboxes = y_post[0][valid].cpu().numpy()
        else:
            preds = non_max_suppression(y_post, conf_thres=CONF_THRESH, iou_thres=0.45, max_det=100)
            bboxes = preds[0].cpu().numpy()
    elif isinstance(det_out, torch.Tensor):
        preds = non_max_suppression(det_out, conf_thres=CONF_THRESH, iou_thres=0.45, max_det=100)
        bboxes = preds[0].cpu().numpy()
    else:
        bboxes = np.empty((0, 6))

    if len(bboxes):
        bboxes[:, [0, 2]] *= (orig_w / IMG_SIZE)
        bboxes[:, [1, 3]] *= (orig_h / IMG_SIZE)

    lane_prob = torch.sigmoid(out['lane_logits'])[0, 0].cpu().numpy()
    return bboxes, lane_prob

def draw_results(img_bgr, bboxes, lane_prob, lane_thresh=0.5):
    h, w = img_bgr.shape[:2]
    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    lane_mask = cv2.resize(lane_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    lane_binary = (lane_mask > lane_thresh).astype(np.uint8)
    overlay = vis.copy()
    overlay[lane_binary == 1] = (255, 0, 255)
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

    for box in bboxes:
        x1, y1, x2, y2, conf, cls = box
        cls_id = int(cls)
        if cls_id >= NUM_VEHICLE_CLASSES:
            continue
        color = BOX_COLORS[cls_id]
        label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(x1), int(y1)-th-4), (int(x1)+tw, int(y1)), color, -1)
        cv2.putText(vis, label, (int(x1), int(y1)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis

print("Inference functions ready.")""")

    code(nb, 'c8', """import tarfile

# Extract dataset if needed
NB02_TAR = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_yolo_nb02.tar")
if not os.path.isdir(os.path.join(DATASET_DIR, "images", "val")):
    assert os.path.isfile(NB02_TAR), f"NB02 tar not found: {NB02_TAR}"
    os.makedirs(DATASET_DIR, exist_ok=True)
    with tarfile.open(NB02_TAR, "r") as tar:
        tar.extractall(DATASET_DIR, filter='data')
    print("Dataset extracted.")
else:
    print("Dataset already present.")

# Select sample images
val_dir = os.path.join(DATASET_DIR, "images", "val")
all_images = sorted(os.listdir(val_dir))
random.seed(42)
sample_images = random.sample(all_images, min(6, len(all_images)))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for i, fname in enumerate(sample_images):
    img_path = os.path.join(val_dir, fname)
    img_bgr = cv2.imread(img_path)
    bboxes, lane_prob = run_joint_inference(model, img_bgr)
    vis = draw_results(img_bgr, bboxes, lane_prob)

    ax = axes[i // 3, i % 3]
    ax.imshow(vis)
    ax.set_title(f"{fname} ({len(bboxes)} dets)")
    ax.axis('off')

plt.suptitle("Joint Inference: Vehicle Detection + Lane Segmentation", fontsize=14)
plt.tight_layout()
plt.show()""")

    md(nb, 'c9', '## 4 — Full Evaluation')

    code(nb, 'c10', """# Extract masks if needed
MASKS_TAR = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_lane_masks.tar")
if not os.path.isdir(os.path.join(DATASET_DIR, "masks", "val")):
    assert os.path.isfile(MASKS_TAR), f"Masks tar not found: {MASKS_TAR}"
    with tarfile.open(MASKS_TAR, "r") as tar:
        tar.extractall(DATASET_DIR, filter='data')
    print("Masks extracted.")

from src.data.transforms import JointTransform
from src.data.dataset import JointBDDDataset, joint_collate_fn
from torch.utils.data import DataLoader

val_transform = JointTransform(img_size=640, mask_height=160, mask_width=160, augment=False)
val_ds = JointBDDDataset(
    images_dir=os.path.join(DATASET_DIR, "images", "val"),
    labels_dir=os.path.join(DATASET_DIR, "labels", "val"),
    masks_dir=os.path.join(DATASET_DIR, "masks", "val"),
    transform=val_transform,
)
val_loader = DataLoader(
    val_ds, batch_size=16, shuffle=False, num_workers=4,
    collate_fn=joint_collate_fn, pin_memory=True,
)
print(f"Validation: {len(val_ds)} samples, {len(val_loader)} batches")""")

    code(nb, 'c11', """from src.metrics.detection import DetectionMetrics
from src.metrics.lane import LaneMetrics
from src.utils.class_map import remap_targets_bdd_to_vehicle
from tqdm import tqdm

det_metrics = DetectionMetrics(device='cuda')
lane_metrics = LaneMetrics(thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])

print(f"Evaluating on {len(val_ds)} samples...")

for batch in tqdm(val_loader, desc="Evaluating"):
    images = batch['images'].to('cuda')
    det_targets = batch['det_targets'].to('cuda')
    lane_masks = batch['lane_masks'].to('cuda')

    with torch.no_grad(), torch.amp.autocast('cuda'):
        outputs = model(images)

    # Lane metrics
    lane_metrics.update(outputs['lane_logits'], lane_masks)

    # Detection metrics
    det_out = outputs['det_output']

    # Handle YOLO26 end2end output format
    if isinstance(det_out, (tuple, list)) and len(det_out) == 2:
        y_post = det_out[0]
        if isinstance(y_post, torch.Tensor) and y_post.dim() == 3 and y_post.shape[-1] == 6:
            preds = [y_post[bi][y_post[bi][:, 4] > 0.001] for bi in range(y_post.shape[0])]
        else:
            preds = non_max_suppression(y_post, conf_thres=0.001, iou_thres=0.6, max_det=300)
    elif isinstance(det_out, torch.Tensor):
        preds = non_max_suppression(det_out, conf_thres=0.001, iou_thres=0.6, max_det=300)
    else:
        preds = [torch.empty((0, 6), device=images.device)] * images.shape[0]

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

det_results = det_metrics.compute()
lane_results = lane_metrics.compute()""")

    code(nb, 'c12', """# Print results
print("=" * 55)
print(" DETECTION METRICS")
print("=" * 55)
print(f"  mAP@50-95: {det_results.get('det_map50_95', 0)*100:.2f}%")
print(f"  mAP@50:    {det_results.get('det_map50', 0)*100:.2f}%")
print()
for cls_name in VEHICLE_CLASSES:
    ap = det_results.get(f'ap_{cls_name}', None)
    if ap is not None:
        print(f"  AP {cls_name:<12}: {ap*100:.2f}%")

print()
print("=" * 55)
print(" LANE METRICS")
print("=" * 55)
print(f"  mIoU:         {lane_results.get('lane_miou', 0)*100:.2f}%")
print(f"  F1:           {lane_results.get('lane_f1', 0)*100:.2f}%")
print(f"  Best thresh:  {lane_results.get('lane_best_thresh', 0.5)}")
print(f"  Best F1:      {lane_results.get('lane_best_f1', 0)*100:.2f}%")
print("=" * 55)""")

    md(nb, 'c13', '## 5 — Lane Threshold Analysis')

    code(nb, 'c14', """# Plot F1 and IoU vs threshold
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
f1_vals = [lane_results.get(f'lane_f1_{t}', 0) for t in thresholds]
iou_vals = [lane_results.get(f'lane_iou_{t}', 0) for t in thresholds]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, [v*100 for v in f1_vals], 'o-', label='F1')
ax.plot(thresholds, [v*100 for v in iou_vals], 's-', label='IoU')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score (%)')
ax.set_title('Lane Segmentation: F1 and IoU vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""")

    md(nb, 'c15', '## 6 — Summary')

    code(nb, 'c16', """print("=" * 60)
print(" JOINT MODEL EVALUATION SUMMARY (v4)")
print("=" * 60)
print(f"  Architecture:      v4 native nc=5 + transformer lane head")
print(f"  Classes:           {', '.join(VEHICLE_CLASSES)}")
print(f"  Checkpoint:        {os.path.basename(JOINT_WEIGHTS)}")
print(f"  Epoch:             {epoch}")
print(f"  Vehicle mAP@50-95: {det_results.get('det_map50_95', 0)*100:.2f}%")
print(f"  Vehicle mAP@50:    {det_results.get('det_map50', 0)*100:.2f}%")
print(f"  Lane mIoU:         {lane_results.get('lane_miou', 0)*100:.2f}%")
print(f"  Lane F1:           {lane_results.get('lane_f1', 0)*100:.2f}%")
print("=" * 60)""")

    save_nb(nb, '09_joint_inference.ipynb')


# ═══════════════════════════════════════════════════════════════════════
# NB10 — Video Inference
# ═══════════════════════════════════════════════════════════════════════
def create_nb10():
    nb = make_nb()

    md(nb, 'c0', """# 10 — Video Inference: Vehicle Detection + Lane Segmentation

**Goal:** Run the trained joint model on a video from Google Drive and produce
an annotated output video with vehicle detections and lane overlays.

Input: `Drive/EcoCAR/video/*.mp4`
Output: Annotated video + per-frame stats""")

    md(nb, 'c1', '## 1 — Setup')

    code(nb, 'c2', '!pip install -q ultralytics opencv-python')

    code(nb, 'c3', """import os, sys, time, json, csv
import cv2
import numpy as np
import torch
from google.colab import drive

drive.mount('/content/drive')

ECOCAR_ROOT = '/content/drive/MyDrive/EcoCAR'
VIDEO_DIR = f'{ECOCAR_ROOT}/video'
WEIGHTS_DIR = f'{ECOCAR_ROOT}/weights'
OUTPUT_DIR = '/content/video_output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_DIR = os.path.join(ECOCAR_ROOT, 'project')
assert os.path.isdir(PROJECT_DIR), f'Project not found at {PROJECT_DIR}'
sys.path.insert(0, PROJECT_DIR)

print(f'GPU: {torch.cuda.get_device_name(0)}')""")

    md(nb, 'c4', '## 2 — Load Model')

    code(nb, 'c5', """from src.multitask_model import build_multitask_model
from src.utils.class_map import VEHICLE_CLASSES, NUM_VEHICLE_CLASSES

CKPT_PATH = os.path.join(WEIGHTS_DIR, 'vehicle_lane_v4_best.pt')
assert os.path.isfile(CKPT_PATH), f'Checkpoint not found: {CKPT_PATH}'

ckpt = torch.load(CKPT_PATH, map_location='cuda', weights_only=False)
arch = ckpt.get('arch_config', {})

model = build_multitask_model(
    weights='yolo26s.pt',
    lane_head_type=arch.get('lane_head_type', 'transformer'),
    lane_embed_dim=arch.get('lane_embed_dim', 128),
    lane_num_heads=arch.get('lane_num_heads', 4),
    lane_depth=arch.get('lane_depth', 2),
    mask_height=arch.get('mask_height', 160),
    mask_width=arch.get('mask_width', 160),
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.to('cuda').eval()
print('Model loaded.')""")

    md(nb, 'c6', '## 3 — Find Input Video')

    code(nb, 'c7', """video_files = []
if os.path.isdir(VIDEO_DIR):
    video_files = sorted([
        os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ])

if not video_files:
    print(f'No videos found in {VIDEO_DIR}')
    print('Upload a video to Drive/EcoCAR/video/ and re-run.')
else:
    for vf in video_files:
        sz = os.path.getsize(vf) / (1024**2)
        print(f'  {os.path.basename(vf)} ({sz:.1f} MB)')
    INPUT_VIDEO = video_files[0]
    print(f'\\nUsing: {INPUT_VIDEO}')""")

    md(nb, 'c8', '## 4 — Run Video Inference')

    code(nb, 'c9', """BOX_COLORS = [
    (60, 180, 255), (100, 100, 255), (255, 220, 60),
    (100, 255, 100), (255, 100, 200),
]

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

IMG_SIZE = 640
CONF_THRESH = 0.3
LANE_THRESH = 0.5

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0] + '_annotated.mp4'
out_path = os.path.join(OUTPUT_DIR, out_name)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

frame_stats = []
frame_idx = 0

print(f'Input:  {INPUT_VIDEO}')
print(f'Output: {out_path}')
print(f'Resolution: {orig_w}x{orig_h} @ {fps:.1f} FPS, {total_frames} frames')
print(f'Processing...')

torch.cuda.synchronize()
t_start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t_frame_start = time.time()

    # Preprocess
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to('cuda')

    # Inference
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(tensor)

    # Decode detections
    det_out = out['det_output']
    if isinstance(det_out, (tuple, list)) and len(det_out) == 2:
        y_post = det_out[0]
        if isinstance(y_post, torch.Tensor) and y_post.dim() == 3 and y_post.shape[-1] == 6:
            valid = y_post[0][:, 4] > CONF_THRESH
            bboxes = y_post[0][valid].cpu().numpy()
        else:
            preds = non_max_suppression(y_post, conf_thres=CONF_THRESH, iou_thres=0.45, max_det=100)
            bboxes = preds[0].cpu().numpy()
    elif isinstance(det_out, torch.Tensor):
        preds = non_max_suppression(det_out, conf_thres=CONF_THRESH, iou_thres=0.45, max_det=100)
        bboxes = preds[0].cpu().numpy()
    else:
        bboxes = np.empty((0, 6))

    # Scale boxes back
    if len(bboxes):
        bboxes[:, [0, 2]] *= (orig_w / IMG_SIZE)
        bboxes[:, [1, 3]] *= (orig_h / IMG_SIZE)

    # Lane
    lane_prob = torch.sigmoid(out['lane_logits'])[0, 0].cpu().numpy()
    lane_mask = cv2.resize(lane_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    lane_binary = (lane_mask > LANE_THRESH).astype(np.uint8)

    # Draw
    vis = frame.copy()
    overlay = vis.copy()
    overlay[lane_binary == 1] = (255, 0, 255)
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

    for box in bboxes:
        x1, y1, x2, y2, conf, cls = box
        cls_id = int(cls)
        if cls_id >= NUM_VEHICLE_CLASSES:
            continue
        color = BOX_COLORS[cls_id]
        label = f'{VEHICLE_CLASSES[cls_id]} {conf:.2f}'
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(x1), int(y1)-th-4), (int(x1)+tw, int(y1)), color, -1)
        cv2.putText(vis, label, (int(x1), int(y1)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    writer.write(vis)

    torch.cuda.synchronize()
    t_frame_end = time.time()

    frame_stats.append({
        'frame': frame_idx,
        'time_ms': (t_frame_end - t_frame_start) * 1000,
        'n_detections': len(bboxes),
        'lane_coverage': float(lane_binary.mean()),
    })

    frame_idx += 1
    if frame_idx % 100 == 0:
        elapsed = time.time() - t_start
        fps_actual = frame_idx / elapsed
        print(f'  Frame {frame_idx}/{total_frames} ({fps_actual:.1f} FPS)')

cap.release()
writer.release()

total_time = time.time() - t_start
avg_fps = frame_idx / total_time

print(f'\\nDone! {frame_idx} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)')
print(f'Output: {out_path}')""")

    md(nb, 'c10', '## 5 — Frame Statistics')

    code(nb, 'c11', """import matplotlib.pyplot as plt

times = [s['time_ms'] for s in frame_stats]
dets = [s['n_detections'] for s in frame_stats]
lanes = [s['lane_coverage'] * 100 for s in frame_stats]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(times, linewidth=0.5)
axes[0].set_title('Frame Latency (ms)')
axes[0].set_xlabel('Frame')
axes[0].axhline(np.mean(times), color='r', linestyle='--', label=f'Avg: {np.mean(times):.1f}ms')
axes[0].legend()

axes[1].plot(dets, linewidth=0.5)
axes[1].set_title('Detections per Frame')
axes[1].set_xlabel('Frame')

axes[2].plot(lanes, linewidth=0.5)
axes[2].set_title('Lane Coverage (%)')
axes[2].set_xlabel('Frame')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/frame_stats.png', dpi=150)
plt.show()

# Save stats CSV
csv_path = f'{OUTPUT_DIR}/frame_stats.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['frame', 'time_ms', 'n_detections', 'lane_coverage'])
    w.writeheader()
    w.writerows(frame_stats)
print(f'Frame stats saved: {csv_path}')""")

    code(nb, 'c12', """# Copy output to Drive
import shutil
drive_out = f'{ECOCAR_ROOT}/video/output'
os.makedirs(drive_out, exist_ok=True)
shutil.copy2(out_path, drive_out)
shutil.copy2(csv_path, drive_out)
print(f'Results copied to {drive_out}')""")

    save_nb(nb, '10_video_inference.ipynb')


# ═══════════════════════════════════════════════════════════════════════
# NB11 — GPU Utilization Profiling
# ═══════════════════════════════════════════════════════════════════════
def create_nb11():
    nb = make_nb()

    md(nb, 'c0', """# 11 — GPU Utilization Profiling (H100)

**Goal:** Measure GPU utilization during video inference and produce a per-second report.

Metrics collected per second:
- GPU utilization %
- GPU memory usage (MB)
- Inference FPS
- Frame latency (ms)

Uses NVML (via pynvml) for accurate GPU monitoring, synchronized with inference.""")

    md(nb, 'c1', '## 1 — Setup')

    code(nb, 'c2', """!pip install -q ultralytics pynvml opencv-python""")

    code(nb, 'c3', """import os, sys, time, json, csv, threading
import cv2
import numpy as np
import torch
from google.colab import drive

drive.mount('/content/drive')

ECOCAR_ROOT = '/content/drive/MyDrive/EcoCAR'
VIDEO_DIR = f'{ECOCAR_ROOT}/video'
WEIGHTS_DIR = f'{ECOCAR_ROOT}/weights'
OUTPUT_DIR = '/content/gpu_profile'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_DIR = os.path.join(ECOCAR_ROOT, 'project')
assert os.path.isdir(PROJECT_DIR), f'Project not found at {PROJECT_DIR}'
sys.path.insert(0, PROJECT_DIR)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB)')""")

    md(nb, 'c4', '## 2 — NVML GPU Monitor')

    code(nb, 'c5', """import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_info = pynvml.nvmlDeviceGetName(handle)
if isinstance(gpu_info, bytes):
    gpu_info = gpu_info.decode()
print(f'NVML device: {gpu_info}')

class GPUMonitor:
    \"\"\"Background thread that polls GPU utilization every interval_ms.\"\"\"

    def __init__(self, interval_ms=100):
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _poll(self):
        while self._running:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.samples.append({
                    'timestamp': time.time(),
                    'gpu_util': util.gpu,
                    'mem_util': util.memory,
                    'mem_used_mb': mem.used / (1024**2),
                    'mem_total_mb': mem.total / (1024**2),
                })
            except Exception:
                pass
            time.sleep(self.interval)

    def get_per_second_report(self, start_time):
        \"\"\"Aggregate samples into per-second buckets.\"\"\"
        if not self.samples:
            return []

        report = []
        current_second = 0
        bucket = []

        for s in self.samples:
            sec = int(s['timestamp'] - start_time)
            if sec < 0:
                continue
            if sec != current_second:
                if bucket:
                    report.append({
                        'second': current_second,
                        'gpu_util_avg': np.mean([b['gpu_util'] for b in bucket]),
                        'gpu_util_max': max(b['gpu_util'] for b in bucket),
                        'mem_used_avg_mb': np.mean([b['mem_used_mb'] for b in bucket]),
                        'mem_used_max_mb': max(b['mem_used_mb'] for b in bucket),
                        'n_samples': len(bucket),
                    })
                current_second = sec
                bucket = []
            bucket.append(s)

        if bucket:
            report.append({
                'second': current_second,
                'gpu_util_avg': np.mean([b['gpu_util'] for b in bucket]),
                'gpu_util_max': max(b['gpu_util'] for b in bucket),
                'mem_used_avg_mb': np.mean([b['mem_used_mb'] for b in bucket]),
                'mem_used_max_mb': max(b['mem_used_mb'] for b in bucket),
                'n_samples': len(bucket),
            })

        return report

print('GPUMonitor ready.')""")

    md(nb, 'c6', '## 3 — Load Model & Video')

    code(nb, 'c7', """from src.multitask_model import build_multitask_model
from src.utils.class_map import VEHICLE_CLASSES, NUM_VEHICLE_CLASSES

CKPT_PATH = os.path.join(WEIGHTS_DIR, 'vehicle_lane_v4_best.pt')
assert os.path.isfile(CKPT_PATH), f'Checkpoint not found: {CKPT_PATH}'

ckpt = torch.load(CKPT_PATH, map_location='cuda', weights_only=False)
arch = ckpt.get('arch_config', {})

model = build_multitask_model(
    weights='yolo26s.pt',
    lane_head_type=arch.get('lane_head_type', 'transformer'),
    lane_embed_dim=arch.get('lane_embed_dim', 128),
    lane_num_heads=arch.get('lane_num_heads', 4),
    lane_depth=arch.get('lane_depth', 2),
    mask_height=arch.get('mask_height', 160),
    mask_width=arch.get('mask_width', 160),
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.to('cuda').eval()

# Find video
video_files = sorted([
    os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
]) if os.path.isdir(VIDEO_DIR) else []
assert video_files, f'No videos in {VIDEO_DIR}'
INPUT_VIDEO = video_files[0]

cap = cv2.VideoCapture(INPUT_VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f'Video: {os.path.basename(INPUT_VIDEO)} ({total_frames} frames @ {fps_video:.0f} FPS)')""")

    md(nb, 'c8', '## 4 — Profiled Inference')

    code(nb, 'c9', """try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

IMG_SIZE = 640
CONF_THRESH = 0.3

# Warmup
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device='cuda')
for _ in range(10):
    with torch.no_grad(), torch.amp.autocast('cuda'):
        _ = model(dummy)
torch.cuda.synchronize()
print('Warmup complete.')

# Start GPU monitor
monitor = GPUMonitor(interval_ms=100)
frame_times = []

torch.cuda.synchronize()
monitor.start()
t_start = time.time()
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to('cuda')

    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(tensor)

    det_out = out['det_output']
    if isinstance(det_out, (tuple, list)) and len(det_out) == 2:
        y_post = det_out[0]
        if not (isinstance(y_post, torch.Tensor) and y_post.dim() == 3 and y_post.shape[-1] == 6):
            _ = non_max_suppression(y_post, conf_thres=CONF_THRESH, iou_thres=0.45, max_det=100)
    elif isinstance(det_out, torch.Tensor):
        _ = non_max_suppression(det_out, conf_thres=CONF_THRESH, iou_thres=0.45, max_det=100)
    _ = torch.sigmoid(out['lane_logits'])

    torch.cuda.synchronize()
    t1 = time.time()

    frame_times.append((t1 - t0) * 1000)
    frame_idx += 1

    if frame_idx % 200 == 0:
        print(f'  Frame {frame_idx}/{total_frames}')

torch.cuda.synchronize()
t_end = time.time()

monitor.stop()
cap.release()

total_time = t_end - t_start
avg_fps = frame_idx / total_time
avg_latency = np.mean(frame_times)
p95_latency = np.percentile(frame_times, 95)
p99_latency = np.percentile(frame_times, 99)

print(f'\\nProfiling complete:')
print(f'  Frames:       {frame_idx}')
print(f'  Total time:   {total_time:.1f}s')
print(f'  Avg FPS:      {avg_fps:.1f}')
print(f'  Avg latency:  {avg_latency:.2f}ms')
print(f'  P95 latency:  {p95_latency:.2f}ms')
print(f'  P99 latency:  {p99_latency:.2f}ms')""")

    md(nb, 'c10', '## 5 — Per-Second GPU Utilization Report')

    code(nb, 'c11', """report = monitor.get_per_second_report(t_start)

print(f'Per-second GPU utilization report ({len(report)} seconds):')
print(f'{\"Second\":>6} {\"GPU%\":>6} {\"GPU Max%\":>9} {\"Mem (MB)\":>10} {\"Samples\":>8}')
print('-' * 45)
for r in report:
    print(f'{r[\"second\"]:>6} {r[\"gpu_util_avg\"]:>5.1f}% {r[\"gpu_util_max\"]:>8.1f}% '
          f'{r[\"mem_used_avg_mb\"]:>9.0f} {r[\"n_samples\"]:>8}')

# Save report CSV
report_csv = f'{OUTPUT_DIR}/gpu_utilization_per_second.csv'
with open(report_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'second', 'gpu_util_avg', 'gpu_util_max',
        'mem_used_avg_mb', 'mem_used_max_mb', 'n_samples'
    ])
    w.writeheader()
    w.writerows(report)
print(f'\\nReport saved: {report_csv}')

# Save frame latencies
latency_csv = f'{OUTPUT_DIR}/frame_latencies.csv'
with open(latency_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frame', 'latency_ms'])
    for i, t in enumerate(frame_times):
        w.writerow([i, f'{t:.3f}'])
print(f'Frame latencies saved: {latency_csv}')""")

    md(nb, 'c12', '## 6 — Visualization')

    code(nb, 'c13', """import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# GPU utilization over time
seconds = [r['second'] for r in report]
gpu_avg = [r['gpu_util_avg'] for r in report]
gpu_max = [r['gpu_util_max'] for r in report]

axes[0, 0].fill_between(seconds, gpu_avg, alpha=0.3, color='blue')
axes[0, 0].plot(seconds, gpu_avg, label='Avg', color='blue')
axes[0, 0].plot(seconds, gpu_max, label='Max', color='red', alpha=0.5)
axes[0, 0].set_title('GPU Utilization per Second')
axes[0, 0].set_ylabel('GPU Utilization (%)')
axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 105)

# Memory usage
mem_avg = [r['mem_used_avg_mb'] / 1024 for r in report]
axes[0, 1].plot(seconds, mem_avg, color='green')
axes[0, 1].set_title('GPU Memory Usage per Second')
axes[0, 1].set_ylabel('Memory (GB)')
axes[0, 1].set_xlabel('Time (seconds)')

# Frame latency distribution
axes[1, 0].hist(frame_times, bins=50, color='orange', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(avg_latency, color='red', linestyle='--', label=f'Avg: {avg_latency:.1f}ms')
axes[1, 0].axvline(p95_latency, color='purple', linestyle='--', label=f'P95: {p95_latency:.1f}ms')
axes[1, 0].set_title('Frame Latency Distribution')
axes[1, 0].set_xlabel('Latency (ms)')
axes[1, 0].legend()

# FPS over time (rolling window)
window = 30
if len(frame_times) > window:
    rolling_fps = [1000.0 / np.mean(frame_times[max(0,i-window):i+1]) for i in range(len(frame_times))]
    axes[1, 1].plot(rolling_fps, linewidth=0.5, color='teal')
    axes[1, 1].axhline(avg_fps, color='red', linestyle='--', label=f'Avg: {avg_fps:.0f} FPS')
    axes[1, 1].set_title(f'Rolling FPS (window={window})')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('FPS')
    axes[1, 1].legend()

plt.suptitle(f'GPU Profiling Report — {gpu_name}', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/gpu_profile.png', dpi=150)
plt.show()""")

    md(nb, 'c14', '## 7 — Summary')

    code(nb, 'c15', """# Print final summary
avg_gpu = np.mean(gpu_avg) if gpu_avg else 0
max_gpu = max(gpu_max) if gpu_max else 0
avg_mem = np.mean(mem_avg) if mem_avg else 0

print('=' * 55)
print(f'  GPU PROFILING SUMMARY')
print('=' * 55)
print(f'  GPU:              {gpu_name}')
print(f'  Video:            {os.path.basename(INPUT_VIDEO)}')
print(f'  Frames:           {frame_idx}')
print(f'  Duration:         {total_time:.1f}s')
print(f'  Avg FPS:          {avg_fps:.1f}')
print(f'  Avg latency:      {avg_latency:.2f}ms')
print(f'  P95 latency:      {p95_latency:.2f}ms')
print(f'  P99 latency:      {p99_latency:.2f}ms')
print(f'  Avg GPU util:     {avg_gpu:.1f}%')
print(f'  Peak GPU util:    {max_gpu:.1f}%')
print(f'  Avg GPU memory:   {avg_mem:.2f} GB')
print('=' * 55)

# Copy results to Drive
import shutil
drive_out = f'{ECOCAR_ROOT}/profiling'
os.makedirs(drive_out, exist_ok=True)
for f in ['gpu_utilization_per_second.csv', 'frame_latencies.csv', 'gpu_profile.png']:
    src = f'{OUTPUT_DIR}/{f}'
    if os.path.isfile(src):
        shutil.copy2(src, drive_out)
print(f'\\nResults copied to {drive_out}')""")

    save_nb(nb, '11_gpu_profiling.ipynb')


if __name__ == '__main__':
    create_nb08()
    create_nb09()
    create_nb10()
    create_nb11()
