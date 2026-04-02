"""Generate the two pipeline notebooks as .ipynb files."""
import json, os

NB_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
os.makedirs(NB_DIR, exist_ok=True)

def make_nb():
    return {'cells': [], 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.0'}},
        'nbformat': 4, 'nbformat_minor': 5}

def md(nb, cid, text):
    lines = text.strip().split('\n')
    nb['cells'].append({'cell_type': 'markdown', 'id': cid, 'metadata': {},
        'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]})

def code(nb, cid, text):
    lines = text.strip().split('\n')
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'id': cid,
        'metadata': {}, 'outputs': [],
        'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]})

def save(nb, name):
    p = os.path.join(NB_DIR, name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Wrote {p}')


# ╔═══════════════════════════════════════════════════════════════════╗
# ║ NB00 — Full dual-path pipeline                                   ║
# ╚═══════════════════════════════════════════════════════════════════╝
def create_nb00():
    nb = make_nb()

    # ── Title ─────────────────────────────────────────────────────────
    md(nb, 'title', """# 00 — DualPath Pipeline: Vehicle Detection + Geometric Lane Prediction

**Architecture: DualPathNet**
- Shared ResNet-50 + FPN backbone
- RT-DETR-inspired transformer detection decoder (vehicle-only)
- MapTR-inspired query-based lane decoder (structured polylines, NOT masks)
- Optional weak cross-branch attention

**Dataset: BDD100K**
- Detection: YOLO-format labels, vehicle classes only
- Lanes: structured targets from original BDD100K poly2d annotations

**Key design choices:**
- Weakly-coupled dual-path reduces negative transfer between tasks
- Lane prediction is geometric (ordered point sequences), not raster masks
- Architecture is future-ready for temporal lane memory (StreamMapNet-style)""")

    # ── Setup ─────────────────────────────────────────────────────────
    md(nb, 's1', '## 1 — Environment Setup')

    code(nb, 'c1', """!pip install -q torch torchvision torchmetrics pyyaml scipy opencv-python matplotlib tqdm""")

    code(nb, 'c2', """import torch
import os, sys

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({mem:.1f} GB)")
else:
    print("WARNING: No GPU — training will be slow")""")

    code(nb, 'c3', """from google.colab import drive
drive.mount('/content/drive')

# ── Fixed Drive paths (preserved from old pipeline) ──
ECOCAR_ROOT = "/content/drive/MyDrive/EcoCAR"
DATASET_DRIVE = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_yolo")
WEIGHTS_DIR = os.path.join(ECOCAR_ROOT, "weights")
TRAINING_RUNS = os.path.join(ECOCAR_ROOT, "training_runs")

# ── Project code ──
PROJECT_DIR = os.path.join(ECOCAR_ROOT, "DETR_GeoLane_pipeline")
if not os.path.isdir(PROJECT_DIR):
    # Fallback: clone from repo
    !git clone https://github.com/ChenSiyun1234/EcoCAR-Perception-Pipeline-YOLO26-BDD100K.git /content/repo 2>/dev/null || true
    PROJECT_DIR = "/content/repo/DETR_GeoLane_pipeline"

sys.path.insert(0, PROJECT_DIR)
print(f"Project: {PROJECT_DIR}")
print(f"Dataset: {DATASET_DRIVE}")""")

    # ── Dataset ───────────────────────────────────────────────────────
    md(nb, 's2', """## 2 — Dataset Inspection

Reuses the existing BDD100K YOLO-format dataset from the old pipeline.
Detection labels are `.txt` files; lane targets are parsed from raw BDD100K JSON.""")

    code(nb, 'c4', """import tarfile

LOCAL_DS = "/content/bdd100k_yolo"
os.makedirs(LOCAL_DS, exist_ok=True)

# Extract from Drive tar if needed (faster than FUSE)
NB02_TAR = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_yolo_nb02.tar")
if not os.path.isdir(os.path.join(LOCAL_DS, "images", "val")):
    if os.path.isfile(NB02_TAR):
        print(f"Extracting {NB02_TAR} ...")
        with tarfile.open(NB02_TAR, "r") as tar:
            tar.extractall(LOCAL_DS, filter='data')
        print("Done.")
    elif os.path.isdir(os.path.join(DATASET_DRIVE, "images")):
        # Symlink from Drive
        for sub in ["images", "labels"]:
            src = os.path.join(DATASET_DRIVE, sub)
            dst = os.path.join(LOCAL_DS, sub)
            if not os.path.exists(dst) and os.path.isdir(src):
                os.symlink(src, dst)
        print("Linked dataset from Drive.")
    else:
        print(f"ERROR: No dataset found at {NB02_TAR} or {DATASET_DRIVE}")

# Count
for split in ["train", "val"]:
    img_dir = os.path.join(LOCAL_DS, "images", split)
    lbl_dir = os.path.join(LOCAL_DS, "labels", split)
    n_img = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
    n_lbl = len(os.listdir(lbl_dir)) if os.path.isdir(lbl_dir) else 0
    print(f"  {split}: {n_img} images, {n_lbl} labels")""")

    code(nb, 'c5', """# ── Detection class distribution ──
from collections import Counter

from src.config import BDD_TO_VEHICLE, VEHICLE_CLASSES

lbl_dir = os.path.join(LOCAL_DS, "labels", "train")
class_counts = Counter()
total_files = 0

if os.path.isdir(lbl_dir):
    for f in os.listdir(lbl_dir):
        if not f.endswith(".txt"):
            continue
        total_files += 1
        with open(os.path.join(lbl_dir, f)) as fh:
            for line in fh:
                cls_id = int(line.strip().split()[0])
                if cls_id in BDD_TO_VEHICLE:
                    class_counts[BDD_TO_VEHICLE[cls_id]] += 1

print(f"Vehicle detection class distribution ({total_files} files):")
for i, name in enumerate(VEHICLE_CLASSES):
    print(f"  {name:<12}: {class_counts.get(i, 0):>8,}")
print(f"  Total vehicles: {sum(class_counts.values()):>8,}")""")

    # ── Lane label parsing ────────────────────────────────────────────
    md(nb, 's3', """## 3 — Lane Annotation Parsing

Parse raw BDD100K poly2d annotations into structured lane targets.
Each lane becomes a fixed-length ordered point sequence (72 points).""")

    code(nb, 'c6', """from src.config import find_lane_labels

train_lane_json = find_lane_labels("train")
val_lane_json = find_lane_labels("val")

print(f"Train lane labels: {train_lane_json or 'NOT FOUND'}")
print(f"Val lane labels:   {val_lane_json or 'NOT FOUND'}")

if train_lane_json is None:
    print()
    print("="*60)
    print("Lane labels not found on Drive.")
    print("To enable lane training, make sure the old YOLO26 raw-label paths are present.")
    print("This notebook now searches the old YOLO26 locations too, including:")
    print("  - /content/drive/MyDrive/EcoCAR/datasets/bdd100k/labels/")
    print("  - /content/drive/MyDrive/EcoCAR/datasets/")
    print()
    print("Expected filenames:")
    print("  - bdd100k_labels_images_train.json")
    print("  - bdd100k_labels_images_val.json")
    print()
    print("Download from: https://bdd-data.berkeley.edu/")
    print("The pipeline will still train detection without lane labels.")
    print("="*60)""")

    code(nb, 'c7', """# Preview lane annotations if available
if train_lane_json:
    import json
    from src.lane_targets import LaneLabelCache, frame_to_lane_targets

    cache = LaneLabelCache(train_lane_json, max_lanes=10, num_points=72)

    # Show stats
    n_with_lanes = len(cache)
    print(f"Frames with lane annotations: {n_with_lanes}")

    # Preview one frame
    import matplotlib.pyplot as plt
    import numpy as np

    for name in list(cache._cache.keys())[:1]:
        targets = cache.get(name)
        n_lanes = int(targets["existence"].sum())
        print(f"\\nSample: {name} — {n_lanes} lanes")

        fig, ax = plt.subplots(figsize=(10, 3))
        for i in range(n_lanes):
            pts = targets["points"][i]
            ax.plot(pts[:, 0] * 1280, pts[:, 1] * 720, '-', linewidth=2, label=f"Lane {i}")
        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)
        ax.set_title(f"Lane polylines: {name}")
        ax.legend()
        plt.show()""")

    # ── Config ────────────────────────────────────────────────────────
    md(nb, 's4', '## 4 — Configuration')

    code(nb, 'c8', """from src.config import Config

cfg = Config(
    run_name="dualpath_v1",
    dataset_root=LOCAL_DS,
    img_size=640,
    batch_size=8,
    backbone="resnet50",
    pretrained=True,
    det_num_queries=100,
    det_dec_layers=3,
    lane_num_queries=10,
    lane_dec_layers=3,
    lane_points=72,
    cross_attn=True,
    epochs=50,
    lr=1e-4,
    patience=15,
)

cfg.save(os.path.join(cfg.save_dir, "config.yaml"))
print(f"Config saved to {cfg.save_dir}")
print(f"  Backbone: {cfg.backbone}")
print(f"  Det queries: {cfg.det_num_queries}")
print(f"  Lane queries: {cfg.lane_num_queries} x {cfg.lane_points} points")
print(f"  Cross-attn: {cfg.cross_attn}")""")

    # ── Model ─────────────────────────────────────────────────────────
    md(nb, 's5', '## 5 — Build Model')

    code(nb, 'c9', """from src.model import build_model

model = build_model(cfg)
model.print_summary()""")

    code(nb, 'c10', """# Sanity check forward pass
dummy = torch.randn(2, 3, cfg.img_size, cfg.img_size)
if torch.cuda.is_available():
    dummy = dummy.cuda()
    model = model.cuda()

with torch.no_grad():
    out = model(dummy)

for k, v in out.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

model = model.cpu()
torch.cuda.empty_cache()""")

    # ── Dataset + DataLoader ──────────────────────────────────────────
    md(nb, 's6', '## 6 — Dataset & DataLoader')

    code(nb, 'c11', """from src.dataset import build_dataloaders

train_loader, val_loader = build_dataloaders(
    cfg,
    train_lane_json=train_lane_json,
    val_lane_json=val_lane_json,
)

print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
print(f"Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")""")

    code(nb, 'c12', """# Visualize a batch
import matplotlib.pyplot as plt
import numpy as np

batch = next(iter(train_loader))
print(f"Images: {batch['images'].shape}")
print(f"Det targets: {batch['det_targets'].shape}")
print(f"Lane existence: {batch['lane_existence'].shape}")
print(f"Lane points: {batch['lane_points'].shape}")
print(f"Has lanes: {batch['has_lanes']}")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i in range(min(4, batch['images'].shape[0])):
    img = batch['images'][i].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    axes[i].imshow(img)
    axes[i].set_title(f"Image {i}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()""")

    # ── Training ──────────────────────────────────────────────────────
    md(nb, 's7', """## 7 — Training

Trains the DualPathNet with:
- AdamW optimizer (backbone LR x0.1)
- Cosine LR schedule with warmup
- Hungarian matching for both detection and lane tasks
- Metric-driven checkpointing (best det, best lane, best joint)""")

    code(nb, 'c13', """from src.trainer import Trainer

trainer = Trainer(cfg, model, train_loader, val_loader)
print(f"Training for {cfg.epochs} epochs")
print(f"Save dir: {cfg.save_dir}")""")

    code(nb, 'c14', """import traceback

try:
    history = trainer.train()
except Exception as e:
    print(f"\\nTraining stopped: {e}")
    traceback.print_exc(limit=5)
    history = trainer.history

print(f"\\nLogged {len(history)} epochs.")""")

    # ── Training curves ───────────────────────────────────────────────
    md(nb, 's8', '## 8 — Training Curves')

    code(nb, 'c15', """import matplotlib.pyplot as plt

if history and len(history) > 0:
    epochs = range(1, len(history) + 1)
    train_loss = [h.get("train_loss", 0) for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]
    det_map = [h.get("det_map50", 0) * 100 for h in history]
    lane_f1 = [h.get("lane_f1", 0) * 100 for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, det_map)
    axes[1].set_title("Detection mAP@50 (%)")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, lane_f1)
    axes[2].set_title("Lane F1 (%)")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "training_curves.png"), dpi=150)
    plt.show()
else:
    print("No history to plot.")""")

    # ── Qualitative results ───────────────────────────────────────────
    md(nb, 's9', '## 9 — Qualitative Inference')

    code(nb, 'c16', """import cv2
import random
from src.visualize import draw_all

model.eval()
if torch.cuda.is_available():
    model = model.cuda()

val_dir = os.path.join(LOCAL_DS, "images", "val")
all_images = sorted(os.listdir(val_dir))
random.seed(42)
sample_images = random.sample(all_images, min(6, len(all_images)))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for i, fname in enumerate(sample_images):
    img_bgr = cv2.imread(os.path.join(val_dir, fname))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (cfg.img_size, cfg.img_size))

    tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=cfg.amp):
        outputs = model(tensor)

    vis = draw_all(img_resized, outputs, conf_thresh=0.3, lane_thresh=0.5)

    ax = axes[i // 3, i % 3]
    ax.imshow(vis)
    ax.set_title(fname)
    ax.axis('off')

plt.suptitle("DualPathNet Inference", fontsize=14)
plt.tight_layout()
plt.show()""")

    # ── Save best weights ─────────────────────────────────────────────
    md(nb, 's10', '## 10 — Export Best Weights')

    code(nb, 'c17', """import shutil

best_src = os.path.join(cfg.save_dir, "weights", "best_joint.pt")
best_dst = os.path.join(WEIGHTS_DIR, f"{cfg.run_name}_best.pt")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

if os.path.isfile(best_src):
    shutil.copy2(best_src, best_dst)
    print(f"Best weights copied to: {best_dst}")
else:
    last_src = os.path.join(cfg.save_dir, "weights", "last.pt")
    if os.path.isfile(last_src):
        last_dst = os.path.join(WEIGHTS_DIR, f"{cfg.run_name}_last.pt")
        shutil.copy2(last_src, last_dst)
        print(f"Last weights copied to: {last_dst}")

print("\\n" + "="*55)
print("  TRAINING COMPLETE")
print("="*55)
print(f"  Run:        {cfg.run_name}")
print(f"  Epochs:     {len(history)}")
print(f"  Best det:   {trainer.best_scores['det']*100:.2f}%")
print(f"  Best lane:  {trainer.best_scores['lane']*100:.2f}%")
print(f"  Best joint: {trainer.best_scores['joint']:.4f}")
print(f"  Weights:    {WEIGHTS_DIR}")
print("="*55)""")

    save(nb, '00_dualpath_pipeline.ipynb')


# ╔═══════════════════════════════════════════════════════════════════╗
# ║ NB01 — H100 Video Profiling                                      ║
# ╚═══════════════════════════════════════════════════════════════════╝
def create_nb01():
    nb = make_nb()

    md(nb, 'title', """# 01 — H100 Video Profiling

**Goal:** Measure inference performance on an H100 GPU.

Metrics per second:
- GPU utilization %
- GPU memory usage (MB)
- Inference FPS
- Frame latency (ms)""")

    md(nb, 's1', '## 1 — Setup')

    code(nb, 'c1', """!pip install -q torch torchvision pynvml opencv-python matplotlib pyyaml scipy""")

    code(nb, 'c2', """import os, sys, time, csv, threading
import cv2
import numpy as np
import torch
from google.colab import drive
drive.mount('/content/drive')

ECOCAR_ROOT = "/content/drive/MyDrive/EcoCAR"
VIDEO_DIR = os.path.join(ECOCAR_ROOT, "video")
WEIGHTS_DIR = os.path.join(ECOCAR_ROOT, "weights")
OUTPUT_DIR = "/content/gpu_profile"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_DIR = os.path.join(ECOCAR_ROOT, "DETR_GeoLane_pipeline")
if not os.path.isdir(PROJECT_DIR):
    !git clone https://github.com/ChenSiyun1234/EcoCAR-Perception-Pipeline-YOLO26-BDD100K.git /content/repo 2>/dev/null || true
    PROJECT_DIR = "/content/repo/DETR_GeoLane_pipeline"
sys.path.insert(0, PROJECT_DIR)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")""")

    md(nb, 's2', '## 2 — NVML GPU Monitor')

    code(nb, 'c3', """import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

class GPUMonitor:
    def __init__(self, interval_ms=100):
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)

    def _poll(self):
        while self._running:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.samples.append({
                    'timestamp': time.time(),
                    'gpu_util': util.gpu,
                    'mem_used_mb': mem.used / (1024**2),
                })
            except Exception:
                pass
            time.sleep(self.interval)

    def per_second_report(self, t0):
        report = []
        if not self.samples:
            return report
        bucket, sec = [], 0
        for s in self.samples:
            t = int(s['timestamp'] - t0)
            if t < 0:
                continue
            if t != sec:
                if bucket:
                    report.append({
                        'second': sec,
                        'gpu_util_avg': np.mean([b['gpu_util'] for b in bucket]),
                        'gpu_util_max': max(b['gpu_util'] for b in bucket),
                        'mem_used_avg_mb': np.mean([b['mem_used_mb'] for b in bucket]),
                    })
                sec, bucket = t, []
            bucket.append(s)
        if bucket:
            report.append({
                'second': sec,
                'gpu_util_avg': np.mean([b['gpu_util'] for b in bucket]),
                'gpu_util_max': max(b['gpu_util'] for b in bucket),
                'mem_used_avg_mb': np.mean([b['mem_used_mb'] for b in bucket]),
            })
        return report

print("GPUMonitor ready.")""")

    md(nb, 's3', '## 3 — Load Model')

    code(nb, 'c4', """from src.config import Config
from src.model import build_model

CKPT_PATH = os.path.join(WEIGHTS_DIR, "dualpath_v1_best.pt")
if not os.path.isfile(CKPT_PATH):
    # Try last
    CKPT_PATH = os.path.join(WEIGHTS_DIR, "dualpath_v1_last.pt")

assert os.path.isfile(CKPT_PATH), f"No checkpoint found at {CKPT_PATH}"

ckpt = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
saved_cfg = ckpt.get("config", {})
cfg = Config.from_dict(saved_cfg) if saved_cfg else Config()

model = build_model(cfg)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model = model.to("cuda").eval()
print(f"Model loaded from epoch {ckpt.get('epoch', '?')}")""")

    md(nb, 's4', '## 4 — Find Video')

    code(nb, 'c5', """video_files = sorted([
    os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
]) if os.path.isdir(VIDEO_DIR) else []

assert video_files, f"No videos in {VIDEO_DIR}"
INPUT_VIDEO = video_files[0]

cap = cv2.VideoCapture(INPUT_VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_vid = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {os.path.basename(INPUT_VIDEO)} ({total_frames} frames @ {fps_vid:.0f} FPS)")""")

    md(nb, 's5', '## 5 — Profiled Inference')

    code(nb, 'c6', """IMG_SIZE = cfg.img_size

# Warmup
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device="cuda")
for _ in range(10):
    with torch.no_grad(), torch.amp.autocast("cuda"):
        _ = model(dummy)
torch.cuda.synchronize()
print("Warmup complete.")

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
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to("cuda")

    with torch.no_grad(), torch.amp.autocast("cuda"):
        _ = model(tensor)

    torch.cuda.synchronize()
    frame_times.append((time.time() - t0) * 1000)
    frame_idx += 1
    if frame_idx % 200 == 0:
        print(f"  Frame {frame_idx}/{total_frames}")

torch.cuda.synchronize()
t_end = time.time()
monitor.stop()
cap.release()

total_time = t_end - t_start
avg_fps = frame_idx / total_time
avg_lat = np.mean(frame_times)
p95_lat = np.percentile(frame_times, 95)
p99_lat = np.percentile(frame_times, 99)

print(f"\\nDone: {frame_idx} frames in {total_time:.1f}s")
print(f"  Avg FPS:     {avg_fps:.1f}")
print(f"  Avg latency: {avg_lat:.2f}ms")
print(f"  P95 latency: {p95_lat:.2f}ms")
print(f"  P99 latency: {p99_lat:.2f}ms")""")

    md(nb, 's6', '## 6 — Per-Second Report')

    code(nb, 'c7', """report = monitor.per_second_report(t_start)

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, "gpu_per_second.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["second", "gpu_util_avg", "gpu_util_max", "mem_used_avg_mb"])
    w.writeheader()
    w.writerows(report)
print(f"Report saved: {csv_path}")

# Save latencies
lat_path = os.path.join(OUTPUT_DIR, "frame_latencies.csv")
with open(lat_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "latency_ms"])
    for i, t in enumerate(frame_times):
        w.writerow([i, f"{t:.3f}"])""")

    md(nb, 's7', '## 7 — Visualization')

    code(nb, 'c8', """import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

seconds = [r["second"] for r in report]
gpu_avg = [r["gpu_util_avg"] for r in report]
gpu_max = [r["gpu_util_max"] for r in report]
mem_gb = [r["mem_used_avg_mb"] / 1024 for r in report]

axes[0,0].fill_between(seconds, gpu_avg, alpha=0.3, color="blue")
axes[0,0].plot(seconds, gpu_avg, label="Avg", color="blue")
axes[0,0].plot(seconds, gpu_max, label="Max", color="red", alpha=0.5)
axes[0,0].set_title("GPU Utilization per Second")
axes[0,0].set_ylabel("GPU %")
axes[0,0].legend()
axes[0,0].set_ylim(0, 105)

axes[0,1].plot(seconds, mem_gb, color="green")
axes[0,1].set_title("GPU Memory Usage")
axes[0,1].set_ylabel("GB")

axes[1,0].hist(frame_times, bins=50, color="orange", edgecolor="black", alpha=0.7)
axes[1,0].axvline(avg_lat, color="red", linestyle="--", label=f"Avg: {avg_lat:.1f}ms")
axes[1,0].axvline(p95_lat, color="purple", linestyle="--", label=f"P95: {p95_lat:.1f}ms")
axes[1,0].set_title("Frame Latency Distribution")
axes[1,0].set_xlabel("Latency (ms)")
axes[1,0].legend()

window = 30
if len(frame_times) > window:
    rolling = [1000.0 / np.mean(frame_times[max(0,i-window):i+1]) for i in range(len(frame_times))]
    axes[1,1].plot(rolling, linewidth=0.5, color="teal")
    axes[1,1].axhline(avg_fps, color="red", linestyle="--", label=f"Avg: {avg_fps:.0f} FPS")
    axes[1,1].set_title(f"Rolling FPS (window={window})")
    axes[1,1].set_ylabel("FPS")
    axes[1,1].legend()

plt.suptitle(f"GPU Profiling — {gpu_name}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gpu_profile.png"), dpi=150)
plt.show()""")

    code(nb, 'c9', """# Copy to Drive
import shutil

drive_out = os.path.join(ECOCAR_ROOT, "profiling")
os.makedirs(drive_out, exist_ok=True)
for f in ["gpu_per_second.csv", "frame_latencies.csv", "gpu_profile.png"]:
    src = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(src):
        shutil.copy2(src, drive_out)

print(f"\\n{'='*55}")
print(f"  GPU PROFILING SUMMARY")
print(f"{'='*55}")
print(f"  GPU:          {gpu_name}")
print(f"  Frames:       {frame_idx}")
print(f"  Avg FPS:      {avg_fps:.1f}")
print(f"  Avg latency:  {avg_lat:.2f}ms")
print(f"  P95 latency:  {p95_lat:.2f}ms")
print(f"  Avg GPU util: {np.mean(gpu_avg):.1f}%")
print(f"  Avg GPU mem:  {np.mean(mem_gb):.2f} GB")
print(f"  Results:      {drive_out}")
print(f"{'='*55}")""")

    save(nb, '01_h100_video_profiling.ipynb')


if __name__ == '__main__':
    create_nb00()
    create_nb01()
