# Author: Jiahao Li
# CreatTime: 2022/12/15
# FileName:
# Description: None


# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch(beginning)

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import os
import sys
import cv2
from pathlib import Path
import math

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()  # Get absolute Path
ROOT = FILE.parents[0]  # File parent root path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

from val import process_batch # for end-of-epoch mAP
from models.yolo import Model
from utils.plots import Annotator, Colors
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, colorstr,
                           non_max_suppression, xywh2xyxy, xyxy2xywh, scale_boxes, scale_rootPoints)
from utils.dataloaders import create_dataloader
from utils.metrics import ap_per_class, compute_OKS


def output_to_target(output, max_det=300):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf, root_conf, root_x, root_y] for plotting
    targets = []
    for i, o in enumerate(output):
        box, conf, cls, root_xy, root_conf = o[:max_det].cpu().split((4, 1, 1, 2, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf, root_conf, root_xy), 1))
    return torch.cat(targets, 0).detach().numpy()


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, root_conf_thresh=0.1):
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    colors = Colors()
    max_size = 1920  # max image size
    max_subplots = 1  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width(size of training image)
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # image targets
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 9  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)
            root_conf = ti[:, 6] if labels else ti[:, 7]
            root_xy = ti[:, 7:] if labels else ti[:, 8:]

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
                    root_xy *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            root_xy[:, 0] += x
            root_xy[:, 1] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                # color = colors(cls)
                color = (255, 215, 0)
                cls = names[cls] if names else cls
                if labels or conf[j] > root_conf_thresh:
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
                    ## draw root points
                    if root_conf[j] >= 0.5:
                        annotator.circle(root_xy[j], boundary=[x, y, x + w, y + h], r=4, fill=(0, 204, 255))
    annotator.im.save(fname)  # save


def inference(
        data,
        dataloader,
        model,
        batch_size=1,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thresh=0.001,  # confidence threshold
        iou_thresh=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        root_conf=0.1,
        oks_sigma=0.1,
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        save_dir=Path(''),
        names=None
):
    device, pt, jit, engine = next(model.parameters()).device, True, False, False
    cuda = device.type != 'cpu'
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()   # number of iou : 10

    seen = 0
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    s = ('%22s' + '%11s' * 7) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95', 'OKS')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map, OKS = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()
    jdict, stats, ap, ap_class, matched_root = [], [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    with torch.no_grad():
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            with dt[0]:
                if cuda:
                    im = im.to(device)
                    targets = targets.to(device)
                im = im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                nb, _, height, width = im.shape  # batch size, channels, height, width

            # Inference
            with dt[1]:
                # when model.eval(), forward_func will return two outputs: (pred_With_sigmoid and fit to orig_image, None)
                # pred: the output after sigmoid and is fit to orig_image_size(without normalized)
                # pred.size() == [ batch_size, sum(三个layer的 num_anchor*grid_x*grid_y), outputs(xywh+conf+num_class+(root_x, root_y, root_conf)*num_kps) ]
                preds, train_out = model(im, augment=augment)

            # NMS
            targets[:, [2, 3, 4, 5, 7, 8]] *= torch.tensor((width, height, width, height, width, height), device=device)
                                            # target.size() = num_objects_in_whole_batch * 9(img_index_in_batch, category, x, y, w, h, root_conf, root_x, root_y)
            with dt[2]:
                preds = non_max_suppression(preds,
                                            conf_thresh,
                                            iou_thresh,
                                            labels=[],
                                            multi_label=False,
                                            agnostic=single_cls,
                                            max_det=max_det)

            # Plot images
            image_name = '.'.join([paths[0].split('.')[0].split('/')[-1], 'png'])
            plot_images(im, targets, None, os.path.join('./results/GroundTruth', image_name), names, root_conf_thresh=root_conf)  # labels
            plot_images(im, output_to_target(preds), None, os.path.join('./results/Prediction', image_name), names, root_conf_thresh=root_conf)



            # Metrics
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                path, shape = Path(paths[si]), shapes[si][0]  # path: image_path, shape: image_shape
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init, npr: num_pred_bbox, niou: 10(mAP from 0.5 to 0.95)
                seen += 1   # for calculate time consuming

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                        matched_root.append((torch.zeros((0, 3), device=device), torch.zeros((0, 4), device=device)))
                    continue

                # Predictions
                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                predn[:, :4] = scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # Rescale boxes (xyxy) to original_img_size
                predn[:, [6, 7]] = scale_rootPoints(im[si].shape[1:], predn[:, [6, 7]], shape, shapes[si][1]) # Rescale rootPoints  to original_img_size

                # Evaluate
                if nl:
                    ## mAP match
                    tbox = xywh2xyxy(labels[:, 1:5])  # xywh -> x_min, y_min, x_max, y_max
                    tbox = scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  #  # Rescale boxes (xyxy) to original_img_shape
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct, matches_AP50 = process_batch(predn[:, :6], labelsn, iouv)

                    ## OKS match
                    if matches_AP50.size(0):
                        troot = labels[:, 6:]
                        troot = scale_rootPoints(im[si].shape[1:], troot, shape, shapes[si][1])
                        area = (tbox[:, 2] - tbox[:, 0]) * (tbox[:, 3] - tbox[:, 1])
                        troot = torch.cat((labels[:, 5:6], troot, area[..., None]), 1)  # troot.size(): num_objects, 4(root_conf, root_x, root_y, bbox_area)
                        matched_pred_root = predn[matches_AP50[:, 1]][:, 6:] # matched_pred_root.size(): num_matched_objects, 3(root_x, root_y, root_conf)
                        matched_target_root = troot[matches_AP50[:, 0]]
                        matched_root.append((matched_pred_root, matched_target_root))
                    else:
                        matched_root.append((torch.zeros((0, 3), device=device), torch.zeros((0, 4), device=device)))
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pred_class, target_class)



    # Compute metrics
    ## mAP
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    ## OKS
    matched_root = [torch.cat(x, 0).cpu().numpy() for x in zip(*matched_root)]
    if len(matched_root) and matched_root[0].any():
        OKS = compute_OKS(*matched_root, sigma=oks_sigma, conf=root_conf)


    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 5  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, OKS))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in  testset, can not compute metrics without labels')
    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)


def main(opt):

    if not os.path.exists(opt.results):
        os.mkdir(opt.results)
        os.mkdir(os.path.join(opt.results, 'GroundTruth'))
        os.mkdir(os.path.join(opt.results, 'Prediction'))


    data_dict = check_dataset(opt.data)  # check if data is None(./DataInfo.yaml)
    test_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'Weeds'} if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(opt.model, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=None)  # Initailize the model
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as Float32
    model.load_state_dict(csd)
    model.to(device)

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is grid_size_multiply
    test_loader = create_dataloader(test_path,
                                   imgsz,
                                   batch_size=opt.batch_size,
                                   stride=gs,
                                   single_cls=True,
                                   cache=None,
                                   rect=True,
                                   rank=-1,
                                   workers=opt.workers * 2,
                                   pad=0.0,
                                   prefix=colorstr('test: '))[0]

    model.eval()
    inference(opt.data, test_loader, model, batch_size=opt.batch_size, imgsz=opt.imgsz,
              conf_thresh=opt.conf_thresh, iou_thresh=opt.iou_thresh, max_det=300,
              root_conf=opt.root_conf, oks_sigma=opt.oks_sigma, single_cls=opt.single_cls, names=names)



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=2, help='number of class')
    parser.add_argument('--model', type=str, default='./saved_model/best.pt')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./DataInfo.yaml', help='dataset.yaml path')
    parser.add_argument('--batch_size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single_cls', default=True, help='train multi-class data as single-class')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--results', default='./results')
    parser.add_argument('--conf_thresh', type=float, default=0.001)
    parser.add_argument('--iou_thresh', type=float, default=0.6)
    parser.add_argument('--root_conf', type=float, default=0.1)
    parser.add_argument('--oks_sigma', type=float, default=0.2)

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

