"""
Default configuration for Vehicle + Lane detection pipeline.
Adapted from YOLOP config with drivable-area segmentation removed.
"""
import os
from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = (0,)
_C.WORKERS = 2
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = True
_C.NEED_AUTOANCHOR = False
_C.DEBUG = False

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Google Drive persistence (Colab)
_C.DRIVE = CN()
_C.DRIVE.ROOT = '/content/drive/MyDrive/EcoCAR'
_C.DRIVE.CHECKPOINT_DIR = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/checkpoints'
_C.DRIVE.METRICS_DIR = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/metrics'

# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
# MODEL.NAME selects the baseline: 'YOLOP' or 'YOLOPv2'. See lib/models/__init__.py.
_C.MODEL.NAME = 'YOLOPv2'
_C.MODEL.NC = 5  # number of detection classes (vehicle-only)
_C.MODEL.VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
_C.MODEL.PRETRAINED = ""
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height
_C.MODEL.EXTRA = CN(new_allowed=True)

# loss params (no DA_SEG_GAIN)
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None  # [cls, obj, iou, ll_seg, ll_iou] - 5 elements
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0
_C.LOSS.OBJ_POS_WEIGHT = 1.0
_C.LOSS.SEG_POS_WEIGHT = 1.0
_C.LOSS.BOX_GAIN = 0.05
_C.LOSS.CLS_GAIN = 0.5
_C.LOSS.OBJ_GAIN = 1.0
_C.LOSS.LL_SEG_GAIN = 1.0   # lane line segmentation loss gain (YOLOP/YOLOPv2 mask path)
_C.LOSS.LL_IOU_GAIN = 0.2   # lane line IoU loss gain
_C.LOSS.LL_FL_GAMMA = 0.0   # lane seg focal-loss γ; YOLOPv2 paper turns this on
_C.LOSS.LL_DICE_GAIN = 0.0  # hybrid focal+dice variant (YOLOPv2 §3 ablation)
# Task weights for the YOLOPv2-DETRLane Stage-2 variant only.
_C.LOSS.DET_TASK_WEIGHT = 1.0
_C.LOSS.LANE_TASK_WEIGHT = 1.0

# LANE head config — consumed by the YOLOPv2-DETRLane variant.
# Not used by the YOLOP / YOLOPv2 mask baselines.
_C.LANE = CN(new_allowed=True)
_C.LANE.NUM_QUERIES = 10
_C.LANE.NUM_POINTS = 72
_C.LANE.NUM_TYPES = 7
_C.LANE.MAX_GT = 10
_C.LANE.D_MODEL = 256
_C.LANE.NHEAD = 8
_C.LANE.FFN_DIM = 1024
_C.LANE.DEC_LAYERS = 3
_C.LANE.DROPOUT = 0.0
_C.LANE.USE_TASK_ADAPTERS = True
_C.LANE.ADAPTER_HIDDEN_RATIO = 0.5
# Per-term lane loss weights
_C.LANE.EXIST_WEIGHT = 2.0
_C.LANE.PTS_WEIGHT = 5.0
_C.LANE.TYPE_WEIGHT = 1.0
_C.LANE.TANGENT_WEIGHT = 1.0
_C.LANE.CURVATURE_WEIGHT = 0.5
_C.LANE.OVERLAP_WEIGHT = 2.0
_C.LANE.VIS_WEIGHT = 0.5
_C.LANE.AUX_WEIGHT = 0.5
_C.LANE.RASTER_H = 72
_C.LANE.RASTER_W = 128
_C.LANE.RASTER_THICKNESS = 0.03
_C.LANE.GEOM_WARMUP_SCALE = 0.70
_C.LANE.GEOM_FINAL_SCALE = 1.00
_C.LANE.RASTER_START_SCALE = 1.00
_C.LANE.RASTER_FINAL_SCALE = 0.15
# Staged training: number of epochs to run detection-only before enabling lane loss
_C.LANE.DET_ONLY_EPOCHS = 3

# DATASET related params (no MASKROOT for drivable area)
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = '/content/bdd100k/images/100k'
_C.DATASET.LABELROOT = '/content/bdd100k/labels/100k'
_C.DATASET.LANEROOT = '/content/bdd100k/lane_masks'  # pre-rendered lane masks or packaged dataset root
_C.DATASET.LANE_DIR_CANDIDATES = ['masks', 'lane_masks']
_C.DATASET.LABEL_FORMAT = 'auto'  # auto|txt|json
_C.DATASET.LANE_JSON_TRAIN = ''  # path to BDD100K lane label JSON for train
_C.DATASET.LANE_JSON_VAL = ''    # path to BDD100K lane label JSON for val
_C.DATASET.DATASET = 'BddDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.ROOT = ''  # optional packaged dataset root, e.g. /content/bdd100k_vehicle5
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [720, 1280]

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015
_C.DATASET.HSV_S = 0.7
_C.DATASET.HSV_V = 0.4

# Mosaic + MixUp augmentation gates.
# [INFERRED] YOLOPv2 does not publish training code; these are YOLOv7
# defaults that YOLOPv2 inherits by lineage. Keep OFF for a faithful
# YOLOP baseline; turn ON for the YOLOPv2-style baseline.
_C.DATASET.MOSAIC = False
_C.DATASET.MOSAIC_PROB = 1.0
_C.DATASET.MIXUP = False
_C.DATASET.MIXUP_PROB = 0.15

# train (no DRIVABLE_ONLY, SEG_ONLY, ENC_SEG_ONLY)
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001   # initial learning rate
_C.TRAIN.LRF = 0.2      # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

# Scheduler: off = cosine-annealing + linear warmup (YOLOP default, our
# paper-closest interpretation). On = torch SGDR. YOLOPv2 paper text
# mentions "warm restart" without publishing T_0/T_mult — mark [INFERRED]
# if the ablation is enabled.
_C.TRAIN.SGDR = False
_C.TRAIN.SGDR_T0 = 100          # default restart period ≈ END_EPOCH/3
_C.TRAIN.SGDR_TMULT = 1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100

_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

# Task-specific training modes
_C.TRAIN.DET_ONLY = False
_C.TRAIN.LANE_ONLY = False
_C.TRAIN.ENC_DET_ONLY = False

_C.TRAIN.PLOT = True

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 24
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True
_C.TEST.NMS_CONF_THRESHOLD = 0.001
_C.TEST.NMS_IOU_THRESHOLD = 0.6
# YOLOPv2 paper: train 640x640, test 640x384. Kept [640,640] by default
# for back-compat with YOLOP; the YOLOPv2 YAML overrides this to
# [640, 384].
_C.TEST.IMAGE_SIZE = [640, 640]   # width x height

# Lane mask render widths. YOLOPv2 paper: width 8 train, width 2 test.
_C.DATASET.LANE_TRAIN_THICKNESS = 8
_C.DATASET.LANE_TEST_THICKNESS = 2


def update_config(cfg, args):
    cfg.defrost()

    if hasattr(args, 'cfg') and args.cfg:
        cfg.merge_from_file(args.cfg)

    if hasattr(args, 'modelDir') and args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if hasattr(args, 'logDir') and args.logDir:
        cfg.LOG_DIR = args.logDir

    cfg.freeze()
