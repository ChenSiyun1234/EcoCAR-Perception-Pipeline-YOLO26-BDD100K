"""
Train / validate loops for Vehicle + Lane detection.
Adapted from YOLOP function.py with drivable-area segmentation removed.

Output structure:
  model(img) -> (det_out, ll_seg_out)    # 2-tuple, no DA
  target     -> [det_labels, lane_label]  # 2-element list
"""

import time
import math
import os
import json
import random
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from torch import amp  # torch.amp replaces torch.cuda.amp in recent PyTorch
from tqdm import tqdm

from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
from lib.core.general import (
    non_max_suppression, check_img_size, scale_coords,
    xyxy2xywh, xywh2xyxy, box_iou, coco80_to_coco91_class,
    plot_images, ap_per_class, output_to_target
)
from lib.utils.utils import time_synchronized
from lib.utils import plot_img_and_mask, plot_one_box, show_seg_result


def train(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup,
          writer_dict, logger, device, rank=-1):
    """
    Train for one epoch.

    Model outputs (2-tuple):
      output[0]: det heads  [B,3,H,W,nc+5] x3 scales
      output[1]: lane seg   [B,2,H,W]

    Target (2-element list):
      target[0]: det labels  [N,6]
      target[1]: lane masks  [B,2,H,W]
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    # Per-iteration time pivot. `end` is the timestamp of the previous
    # iteration's end; `time.time() - end` at the top of the body is the
    # data-loading wait, and at the bottom is the full iteration. The
    # original YOLOP code left `start = time.time()` outside the loop
    # and never reset it, so both data_time and batch_time accumulated
    # monotonically and made Speed look like 0.0 samples/s.
    end = time.time()
    for i, (input, target, paths, shapes) in enumerate(train_loader):
        # REPAIR (v5): YOLOP upstream uses `num_batch * (epoch - 1)` because
        # its loop is 1-indexed (epoch goes 1..END_EPOCH). Our notebook uses a
        # 0-indexed loop (epoch goes 0..END_EPOCH-1). With the old formula
        # `num_iter` was NEGATIVE for every iteration of epoch 0, which made
        # `np.interp(num_iter, [0, num_warmup], ...)` clamp LR to 0 for the
        # weight groups and to WARMUP_BIASE_LR for the bias group — i.e.
        # only biases trained during epoch 0 while weights sat frozen at
        # zero. That is almost certainly why lane IoU collapsed to 0 after
        # epoch 0 and never recovered. Using `epoch` directly lines up with
        # the 0-indexed outer loop.
        num_iter = i + num_batch * epoch

        if num_iter < num_warmup:
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(num_iter, xi,
                    [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi,
                        [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])

        data_time.update(time.time() - end)
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target

        with amp.autocast(device_type=device.type, enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, head_losses = criterion(outputs, target, shapes, model)

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if rank in [-1, 0]:
            losses.update(total_loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            if i % cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          speed=input.size(0)/max(batch_time.val, 1e-6),
                          data_time=data_time, loss=losses)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        # Reset the per-iteration pivot so the next batch's data_time /
        # batch_time are per-iteration, not cumulative.
        end = time.time()


def validate(epoch, config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, logger=None, device='cpu', rank=-1):
    """
    Validate one epoch.
    Returns: (ll_segment_result, detect_result, losses_avg, maps, times)
    No DA segment result (removed).
    """
    max_stride = 32
    weights = None

    save_dir = output_dir + os.path.sep + 'visualization'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE]
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
    training = False
    is_coco = False
    save_conf = False
    verbose = False
    save_hybrid = False
    log_imgs, wandb = min(16, 100), None

    nc = model.nc  # 5 vehicle classes (was hardcoded nc=1 in YOLOP)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    try:
        import wandb
    except ImportError:
        wandb = None
        log_imgs = 0

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    ll_metric = SegmentationMetric(2)  # lane line binary segmentation

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.

    losses = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    T_inf = AverageMeter()
    T_nms = AverageMeter()

    model.eval()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[0][1][0][0]

            t = time_synchronized()
            det_out, ll_seg_out = model(img)  # 2-tuple (no DA)
            t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf / img.size(0), img.size(0))

            inf_out, train_out = det_out

            # Lane line segment evaluation
            _, ll_predict = torch.max(ll_seg_out, 1)
            _, ll_gt = torch.max(target[1], 1)  # target[1] is lane (was target[2] in YOLOP)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()

            ll_acc_seg.update(ll_acc, img.size(0))
            ll_IoU_seg.update(ll_IoU, img.size(0))
            ll_mIoU_seg.update(ll_mIoU, img.size(0))

            # Loss computation: 2-tuple (no DA)
            total_loss, head_losses = criterion((train_out, ll_seg_out), target, shapes, model)
            losses.update(total_loss.item(), img.size(0))

            # NMS
            t = time_synchronized()
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
            output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRESHOLD,
                                         iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)
            t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms / img.size(0), img.size(0))

            if config.TEST.PLOTS:
                if batch_i == 0:
                    for i in range(test_batch_size):
                        # Lane line visualization
                        img_ll = cv2.imread(paths[i])
                        # REPAIR (v5): take argmax FIRST, then upsample with
                        # nearest — upsampling the 2-channel logits with
                        # bilinear then argmax'ing was wrong (mixes classes at
                        # boundaries). Thin lane supervision also demands
                        # nearest, never bilinear.
                        ll_seg_mask = ll_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
                        ll_seg_mask = torch.nn.functional.interpolate(
                            ll_seg_mask.unsqueeze(1).float(),
                            scale_factor=float(1.0 / max(ratio, 1e-6)),
                            mode='nearest',
                        ).squeeze(1).long()

                        ll_gt_mask = target[1][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        _, ll_gt_mask = torch.max(ll_gt_mask, 1)
                        ll_gt_mask = torch.nn.functional.interpolate(
                            ll_gt_mask.unsqueeze(1).float(),
                            scale_factor=float(1.0 / max(ratio, 1e-6)),
                            mode='nearest',
                        ).squeeze(1).long()

                        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                        ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()
                        img_ll1 = img_ll.copy()
                        _ = show_seg_result(img_ll, ll_seg_mask, i, epoch, save_dir, is_ll=True)
                        _ = show_seg_result(img_ll1, ll_gt_mask, i, epoch, save_dir, is_ll=True, is_gt=True)

                        # Detection visualization
                        img_det = cv2.imread(paths[i])
                        img_gt = img_det.copy()
                        det = output[i].clone()
                        if len(det):
                            det[:, :4] = scale_coords(img[i].shape[1:], det[:, :4], img_det.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img_det, label=label_det_pred, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir + "/batch_{}_{}_det_pred.png".format(epoch, i), img_det)

                        labels = target[0][target[0][:, 0] == i, 1:]
                        labels[:, 1:5] = xywh2xyxy(labels[:, 1:5])
                        if len(labels):
                            labels[:, 1:5] = scale_coords(img[i].shape[1:], labels[:, 1:5], img_gt.shape).round()
                        for cls, x1, y1, x2, y2 in labels:
                            label_det_gt = f'{names[int(cls)]}'
                            xyxy = (x1, y1, x2, y2)
                            plot_one_box(xyxy, img_gt, label=label_det_gt, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir + "/batch_{}_{}_det_gt.png".format(epoch, i), img_gt)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = target[0][target[0][:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            if config.TEST.SAVE_TXT:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    with open(save_dir + '/labels/' + path.stem + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if config.TEST.PLOTS and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            if config.TEST.SAVE_JSON:
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])
                box[:, :2] -= box[:, 2:] / 2
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                if config.TEST.PLOTS:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    if pi.shape[0]:
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    map70 = None
    map75 = None
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap70, ap75, ap = ap[:, 0], ap[:, 4], ap[:, 5], ap.mean(1)
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # REPAIR (v5): `t_inf` / `t_nms` here are the LAST batch's wall clocks,
    # not cumulative totals, so the old `x / seen * 1E3` produced a number
    # that scaled inversely with the number of validation batches. Use the
    # AverageMeter .avg values instead — those are per-image already.
    inf_ms = T_inf.avg * 1e3
    nms_ms = T_nms.avg * 1e3
    total_ms = inf_ms + nms_ms
    t = (inf_ms, nms_ms, total_ms, imgsz, imgsz, batch_size)
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if config.TEST.PLOTS:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(Path(save_dir).glob('test*.jpg'))]})

    # Save JSON
    if config.TEST.SAVE_JSON and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = '../coco/annotations/instances_val2017.json'
        pred_json = str(Path(save_dir) / f"{w}_predictions.json")
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in val_loader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results (no DA segment result)
    if not training:
        s = f"\n{len(list(Path(save_dir).glob('labels/*.txt')))} labels saved to {Path(save_dir) / 'labels'}" if config.TEST.SAVE_TXT else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
    detect_result = np.asarray([mp, mr, map50, map])
    t = [T_inf.avg, T_nms.avg]

    return ll_segment_result, detect_result, losses.avg, maps, t


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
