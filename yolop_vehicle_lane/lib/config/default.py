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
_C.DRIVE.CHECKPOINT_DIR = '/content/drive/MyDrive/EcoCAR/checkpoints'
_C.DRIVE.METRICS_DIR = '/content/drive/MyDrive/EcoCAR/metrics'

# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'VehicleLane'
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
_C.LOSS.LL_SEG_GAIN = 1.0   # lane line segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.2   # lane line IoU loss gain

# DATASET related params (no MASKROOT for drivable area)
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = '/content/bdd100k/images/100k'
_C.DATASET.LABELROOT = '/content/bdd100k/labels/100k'
_C.DATASET.LANEROOT = '/content/bdd100k/lane_masks'  # pre-rendered lane masks
_C.DATASET.LANE_JSON_TRAIN = ''  # path to BDD100K lane label JSON for train
_C.DATASET.LANE_JSON_VAL = ''    # path to BDD100K lane label JSON for val
_C.DATASET.DATASET = 'BddDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
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


def update_config(cfg, args):
    cfg.defrost()

    if hasattr(args, 'cfg') and args.cfg:
        cfg.merge_from_file(args.cfg)

    if hasattr(args, 'modelDir') and args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if hasattr(args, 'logDir') and args.logDir:
        cfg.LOG_DIR = args.logDir

    cfg.freeze()
