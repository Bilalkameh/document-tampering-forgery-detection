from pathlib import Path
import argparse
import csv
import random
import shutil
from datetime import datetime

import cv2
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(num_classes: int = 2) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def run_inference(model, image_path: Path, img_size: int, conf: float, device: torch.device):
    """Returns (pred_boxes_xyxy, pred_scores) in original image pixel coordinates."""
    orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig.size

    resized = orig.resize((img_size, img_size))
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)[0]

    boxes  = preds["boxes"].cpu()
    scores = preds["scores"].cpu()
    keep   = scores >= conf
    boxes  = boxes[keep].tolist()
    scores = scores[keep].tolist()

    # Scale back from img_size to original dimensions
    scaled = []
    for x1, y1, x2, y2 in boxes:
        scaled.append([
            x1 / img_size * orig_w,
            y1 / img_size * orig_h,
            x2 / img_size * orig_w,
            y2 / img_size * orig_h,
        ])

    return scaled, scores


# --------------------------------------------------------------------------- #
# Label helpers (identical to YOLO version)
# --------------------------------------------------------------------------- #

def read_yolo_labels(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


def yolo_box_to_xyxy(box, img_w, img_h):
    _, cx, cy, w, h = box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2


# --------------------------------------------------------------------------- #
# Drawing helpers (identical to YOLO version)
# --------------------------------------------------------------------------- #

def draw_gt(image, gt_boxes):
    h, w = image.shape[:2]
    if not gt_boxes:
        cv2.putText(image, "GT: clean", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 150, 0), 2)
        return image
    for box in gt_boxes:
        x1, y1, x2, y2 = yolo_box_to_xyxy(box, w, h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(image, "GT tampered", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return image


def draw_predictions(image, pred_boxes, pred_scores):
    if not pred_boxes:
        cv2.putText(image, "PRED: no tampering", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 150, 0), 2)
        return image
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(image, f"PRED {score:.2f}", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image


def add_header(image, title, color):
    cv2.rectangle(image, (0, 0), (image.shape[1], 55), (255, 255, 255), -1)
    cv2.putText(image, title, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return image


def resize_same_height(left, right):
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    target_h = max(h1, h2)
    if h1 != target_h:
        left  = cv2.resize(left,  (int(w1 * target_h / h1), target_h))
    if h2 != target_h:
        right = cv2.resize(right, (int(w2 * target_h / h2), target_h))
    return left, right


# --------------------------------------------------------------------------- #
# Sampling / classification helpers (identical to YOLO version)
# --------------------------------------------------------------------------- #

def classify(gt_tampered, pred_tampered):
    if gt_tampered and pred_tampered:
        return "TP"
    if not gt_tampered and not pred_tampered:
        return "TN"
    if not gt_tampered and pred_tampered:
        return "FP"
    return "FN"


def safe_div(a, b):
    return a / b if b else 0.0


def get_test_images(dataset_dir: Path, split: str):
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    clean, tampered = [], []
    for img in images:
        if read_yolo_labels(label_dir / f"{img.stem}.txt"):
            tampered.append(img)
        else:
            clean.append(img)
    return images, clean, tampered


def select_images(images, clean, tampered, n, seed, balanced):
    rng = random.Random(seed)
    if not balanced:
        return rng.sample(images, min(n, len(images)))
    n_clean     = n // 2
    n_tampered  = n - n_clean
    selected    = rng.sample(clean,    min(n_clean,    len(clean)))
    selected   += rng.sample(tampered, min(n_tampered, len(tampered)))
    remaining   = [img for img in images if img not in selected]
    if len(selected) < n:
        selected += rng.sample(remaining, min(n - len(selected), len(remaining)))
    rng.shuffle(selected)
    return selected


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def write_summary(run_dir, args, counts):
    tp, fp, tn, fn = counts["TP"], counts["FP"], counts["TN"], counts["FN"]
    accuracy  = safe_div(tp + tn, tp + fp + tn + fn)
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * precision * recall, precision + recall)

    lines = [
        "Demo Run Summary (Faster R-CNN)",
        "================================",
        "",
        f"Model weights:        {args.weights}",
        f"Dataset:              {args.dataset}",
        f"Split:                {args.split}",
        f"Number of images:     {args.num_images}",
        f"Confidence threshold: {args.conf}",
        f"Image size:           {args.imgsz}",
        "",
        "Results",
        "-------",
        f"TP: {tp}",
        f"FP: {fp}",
        f"TN: {tn}",
        f"FN: {fn}",
        f"Accuracy:  {accuracy:.3f}",
        f"Precision: {precision:.3f}",
        f"Recall:    {recall:.3f}",
        f"F1-score:  {f1:.3f}",
        "",
        "Meaning",
        "-------",
        "TP = tampered document correctly detected",
        "TN = clean document correctly ignored",
        "FP = clean document incorrectly flagged",
        "FN = tampered document missed",
        "",
        "Folders",
        "-------",
        "inputs/       original sampled test images",
        "predictions/  model predictions drawn on image",
        "side_by_side/ ground truth vs prediction panels",
        "results.csv   per-image results table",
    ]

    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return accuracy, precision, recall, f1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN demo run on random test images.")
    parser.add_argument("--dataset",    default="dataset")
    parser.add_argument("--weights",    required=True,      help="Path to Faster R-CNN best.pt")
    parser.add_argument("--split",      default="test")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--conf",       type=float, default=0.5)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--random",     action="store_true", help="Fully random sampling instead of balanced.")
    parser.add_argument("--out-root",   default="demo")
    args = parser.parse_args()

    dataset_dir  = Path(args.dataset)
    weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = torch.device(
        args.device if args.device != "mps"
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.out_root)
        / f"demo_fasterrcnn_{timestamp}_n{args.num_images}_conf{str(args.conf).replace('.', '')}"
    )
    input_dir      = run_dir / "inputs"
    prediction_dir = run_dir / "predictions"
    side_dir       = run_dir / "side_by_side"
    for d in (input_dir, prediction_dir, side_dir):
        d.mkdir(parents=True)

    print(f"Loading model from {weights_path} …")
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    images, clean, tampered = get_test_images(dataset_dir, args.split)
    selected = select_images(images, clean, tampered, args.num_images, args.seed, balanced=not args.random)

    print(f"Selected {len(selected)} images  ({sum(1 for p in selected if read_yolo_labels(dataset_dir / 'labels' / args.split / f'{p.stem}.txt'))} tampered, rest clean)")
    print(f"Demo folder: {run_dir}\n")

    label_dir = dataset_dir / "labels" / args.split
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    rows = []

    for i, image_path in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] {image_path.name}")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Skipping unreadable image.")
            continue

        gt_boxes    = read_yolo_labels(label_dir / f"{image_path.stem}.txt")
        gt_tampered = len(gt_boxes) > 0

        pred_boxes, pred_scores = run_inference(model, image_path, args.imgsz, args.conf, device)
        pred_tampered = len(pred_boxes) > 0

        outcome = classify(gt_tampered, pred_tampered)
        counts[outcome] += 1

        shutil.copy2(image_path, input_dir / image_path.name)

        pred_img = image.copy()
        draw_predictions(pred_img, pred_boxes, pred_scores)
        pred_path = prediction_dir / f"{i:02d}_{outcome}_{image_path.name}"
        cv2.imwrite(str(pred_path), pred_img)

        gt_panel   = image.copy()
        pred_panel = image.copy()
        draw_gt(gt_panel, gt_boxes)
        draw_predictions(pred_panel, pred_boxes, pred_scores)
        add_header(gt_panel,   "Ground Truth",     (0, 0, 255))
        add_header(pred_panel, "Model Prediction", (255, 0, 0))
        gt_panel, pred_panel = resize_same_height(gt_panel, pred_panel)
        combined  = cv2.hconcat([gt_panel, pred_panel])
        side_path = side_dir / f"{i:02d}_{outcome}_{image_path.name}"
        cv2.imwrite(str(side_path), combined)

        rows.append({
            "index":            i,
            "image_name":       image_path.name,
            "ground_truth":     "tampered" if gt_tampered else "clean",
            "prediction":       "tampered" if pred_tampered else "clean",
            "result":           outcome,
            "num_gt_boxes":     len(gt_boxes),
            "num_pred_boxes":   len(pred_boxes),
            "max_confidence":   max(pred_scores) if pred_scores else "",
            "prediction_image": str(pred_path),
            "side_by_side_image": str(side_path),
        })

    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accuracy, precision, recall, f1 = write_summary(run_dir, args, counts)

    print()
    print("Demo run complete")
    print("-" * 60)
    print(f"Run folder:    {run_dir}")
    print(f"Results CSV:   {csv_path}")
    print(f"Summary:       {run_dir / 'summary.txt'}")
    print("-" * 60)
    print(f"TP={counts['TP']}  FP={counts['FP']}  TN={counts['TN']}  FN={counts['FN']}")
    print(f"Accuracy={accuracy:.3f}  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
