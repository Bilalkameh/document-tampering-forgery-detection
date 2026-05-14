"""Box-level validation for a trained Faster R-CNN model.

Computes mAP@0.5 and precision/recall/F1 at a confidence threshold over
any dataset split, then saves results to outputs/validation/<name>/.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reuse the dataset class from training
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train_faster_rcnn import YOLODetectionDataset, collate_fn


def build_model(num_classes: int = 2) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate(model, loader, device, conf_threshold: float, iou_threshold: float = 0.5):
    """
    Returns a dict with:
      - map50: area under precision-recall curve at IoU 0.5
      - precision, recall, f1: at the given conf_threshold
      - tp, fp, fn: box-level counts at the given conf_threshold
      - n_images, n_gt_boxes
    """
    model.eval()

    all_scores, all_tp_flags = [], []
    n_gt_total = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", unit="batch"):
            images = [img.to(device) for img in images]
            preds = model(images)

            for pred, tgt in zip(preds, targets):
                gt_boxes = tgt["boxes"].to(device)
                n_gt = len(gt_boxes)
                n_gt_total += n_gt

                scores = pred["scores"]
                boxes  = pred["boxes"]

                # --- mAP accumulation (all confident enough predictions) ---
                keep_all = scores >= 0.0  # keep everything for PR curve
                s_all = scores[keep_all]
                b_all = boxes[keep_all]

                if len(b_all) > 0 and n_gt > 0:
                    iou = torchvision.ops.box_iou(b_all, gt_boxes)
                    matched_gt: set = set()
                    for i in s_all.argsort(descending=True):
                        best_iou, best_j = iou[i].max(0)
                        best_j = best_j.item()
                        if best_iou.item() >= iou_threshold and best_j not in matched_gt:
                            all_tp_flags.append(1)
                            matched_gt.add(best_j)
                        else:
                            all_tp_flags.append(0)
                        all_scores.append(s_all[i].item())
                elif len(b_all) > 0:
                    all_scores.extend(s_all.cpu().tolist())
                    all_tp_flags.extend([0] * len(b_all))

                # --- threshold-based TP/FP/FN ---
                keep_thresh = scores >= conf_threshold
                b_thresh = boxes[keep_thresh]

                if n_gt == 0:
                    fp += len(b_thresh)
                elif len(b_thresh) == 0:
                    fn += n_gt
                else:
                    iou = torchvision.ops.box_iou(b_thresh, gt_boxes)
                    matched_gt = set()
                    matched_pred = set()
                    for i in scores[keep_thresh].argsort(descending=True):
                        best_iou, best_j = iou[i].max(0)
                        best_j = best_j.item()
                        if best_iou.item() >= iou_threshold and best_j not in matched_gt:
                            tp += 1
                            matched_gt.add(best_j)
                            matched_pred.add(i.item())
                        else:
                            fp += 1
                    fn += n_gt - len(matched_gt)

    # mAP@0.5
    map50 = 0.0
    if all_scores and n_gt_total > 0:
        order = np.argsort(all_scores)[::-1]
        tp_cum = np.cumsum(np.array(all_tp_flags)[order])
        fp_cum = np.cumsum(1 - np.array(all_tp_flags)[order])
        precision_curve = tp_cum / (tp_cum + fp_cum + 1e-9)
        recall_curve    = tp_cum / n_gt_total
        map50 = float(np.trapezoid(precision_curve, recall_curve)) if len(recall_curve) > 1 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "map50":      map50,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "tp":         tp,
        "fp":         fp,
        "fn":         fn,
        "n_gt_boxes": n_gt_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a trained Faster R-CNN model (box-level metrics).")
    parser.add_argument("--weights", type=str, required=True,         help="Path to best.pt weights")
    parser.add_argument("--data",    type=str, default="dataset",     help="YOLO dataset root folder")
    parser.add_argument("--split",   type=str, default="val",         choices=["train", "val", "test"])
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=4)
    parser.add_argument("--device",  type=str, default="cpu")
    parser.add_argument("--conf",    type=float, default=0.5,         help="Confidence threshold for P/R/F1")
    parser.add_argument("--iou",     type=float, default=0.5,         help="IoU threshold for match")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name",    type=str, default="fasterrcnn_val", help="Output subfolder name")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    dataset_dir = Path(args.data)
    images_dir  = dataset_dir / "images" / args.split
    labels_dir  = dataset_dir / "labels" / args.split
    if not images_dir.exists():
        raise FileNotFoundError(f"Images not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels not found: {labels_dir}")

    device = torch.device(
        args.device if args.device != "mps"
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    print("\nFaster R-CNN Validation")
    print("=" * 50)
    print(f"Weights: {weights_path}")
    print(f"Split:   {args.split}")
    print(f"Device:  {device}")
    print(f"Conf:    {args.conf}   IoU: {args.iou}")
    print("=" * 50)

    ds     = YOLODetectionDataset(images_dir, labels_dir, args.imgsz)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn)

    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    results = evaluate(model, loader, device, args.conf, args.iou)
    results.update({"split": args.split, "conf_threshold": args.conf, "iou_threshold": args.iou,
                     "weights": str(weights_path), "n_images": len(ds)})

    out_dir = Path("outputs/validation") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults ({args.split})")
    print("-" * 40)
    print(f"Images:      {results['n_images']}")
    print(f"GT boxes:    {results['n_gt_boxes']}")
    print(f"TP / FP / FN: {results['tp']} / {results['fp']} / {results['fn']}")
    print("-" * 40)
    print(f"mAP@0.5:     {results['map50']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"F1:          {results['f1']:.4f}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
