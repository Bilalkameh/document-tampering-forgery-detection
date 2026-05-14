from pathlib import Path
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class YOLODetectionDataset(Dataset):
    """Reads YOLO-format labels and serves (image_tensor, target_dict) pairs."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, images_dir: Path, labels_dir: Path, img_size: int = 640):
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.samples = sorted(
            p for p in Path(images_dir).iterdir()
            if p.suffix.lower() in self.IMAGE_EXTS
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        img_tensor = torchvision.transforms.functional.to_tensor(img)

        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = (float(x) for x in parts[:5])
                x1 = max(0.0, (cx - w / 2) * self.img_size)
                y1 = max(0.0, (cy - h / 2) * self.img_size)
                x2 = min(float(self.img_size), (cx + w / 2) * self.img_size)
                y2 = min(float(self.img_size), (cy + h / 2) * self.img_size)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)  # 0 = background in torchvision; YOLO class 0 → class 1

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(num_classes: int = 2, pretrained: bool = True) -> torch.nn.Module:
    """Faster R-CNN ResNet-50 FPN with head replaced for num_classes."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# --------------------------------------------------------------------------- #
# Validation mAP@0.5
# --------------------------------------------------------------------------- #

def compute_val_map50(model, loader, device, conf_threshold=0.5, iou_threshold=0.5):
    """Single-class mAP@0.5 over a DataLoader. Returns float in [0, 1]."""
    model.eval()
    all_scores, all_tp, n_gt = [], [], 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="  val mAP", unit="batch", leave=False):
            images = [img.to(device) for img in images]
            preds = model(images)

            for pred, tgt in zip(preds, targets):
                gt_boxes = tgt["boxes"].to(device)
                n_gt += len(gt_boxes)

                scores = pred["scores"]
                boxes = pred["boxes"]
                keep = scores >= conf_threshold
                scores, boxes = scores[keep], boxes[keep]

                if len(boxes) == 0:
                    continue

                if len(gt_boxes) == 0:
                    all_scores.extend(scores.cpu().tolist())
                    all_tp.extend([0] * len(scores))
                    continue

                iou = torchvision.ops.box_iou(boxes, gt_boxes)  # M x N
                matched_gt = set()
                order = scores.argsort(descending=True)
                for i in order:
                    best_iou, best_j = iou[i].max(0)
                    best_j = best_j.item()
                    if best_iou.item() >= iou_threshold and best_j not in matched_gt:
                        all_tp.append(1)
                        matched_gt.add(best_j)
                    else:
                        all_tp.append(0)
                    all_scores.append(scores[i].item())

    if not all_scores or n_gt == 0:
        return 0.0

    order = np.argsort(all_scores)[::-1]
    tp_cum = np.cumsum(np.array(all_tp)[order])
    fp_cum = np.cumsum(1 - np.array(all_tp)[order])
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall = tp_cum / n_gt
    return float(np.trapezoid(precision, recall)) if len(recall) > 1 else 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for document tampering localization.")

    parser.add_argument("--data", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--name", type=str, default="fasterrcnn_baseline")
    parser.add_argument("--project", type=str, default="runs/detect_rcnn")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch instead of COCO weights")
    parser.add_argument("--val-conf", type=float, default=0.5, help="Confidence threshold for val mAP computation")
    parser.add_argument("--exist-ok", action="store_true")

    args = parser.parse_args()

    dataset_dir = Path(args.data)
    out_dir = Path(args.project) / args.name
    if out_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Run directory already exists: {out_dir}. Use --exist-ok to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "mps" else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print("\nFaster R-CNN Document Tampering Training")
    print("=" * 70)
    print(f"Data:       {dataset_dir}")
    print(f"Epochs:     {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch:      {args.batch}")
    print(f"Device:     {device}")
    print(f"Run dir:    {out_dir}")
    print(f"Torch:      {torch.__version__}")
    print("=" * 70)

    config_dir = Path("outputs/training_configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to: {config_path}\n")

    train_ds = YOLODetectionDataset(dataset_dir / "images/train", dataset_dir / "labels/train", args.imgsz)
    val_ds   = YOLODetectionDataset(dataset_dir / "images/val",   dataset_dir / "labels/val",   args.imgsz)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    model = build_model(num_classes=2, pretrained=not args.no_pretrained)
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    best_map50 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=True)
        for images, targets in pbar:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        map50 = compute_val_map50(model, val_loader, device, conf_threshold=args.val_conf)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:>3}/{args.epochs}  loss={avg_loss:.4f}  val_mAP@0.5={map50:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_map50": map50})

        weights_path = out_dir / "weights" / "last.pt"
        weights_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), weights_path)

        if map50 > best_map50:
            best_map50 = map50
            torch.save(model.state_dict(), out_dir / "weights" / "best.pt")
            print(f"  → new best mAP@0.5: {best_map50:.4f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"best_val_map50": best_map50, "history": history}, f, indent=2)

    print(f"\nTraining finished. Best val mAP@0.5: {best_map50:.4f}")
    print(f"Weights: {out_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
