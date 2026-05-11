from pathlib import Path
import argparse
import csv


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_gt_box(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return label_path.read_text(encoding="utf-8").strip() != ""


def has_prediction(pred_label_path: Path, threshold: float) -> bool:
    """
    Ultralytics prediction labels with save_conf=True are usually:
    class_id center_x center_y width height confidence

    If confidence is missing, any non-empty prediction line counts as detected.
    """
    if not pred_label_path.exists():
        return False

    text = pred_label_path.read_text(encoding="utf-8").strip()
    if text == "":
        return False

    for line in text.splitlines():
        parts = line.strip().split()

        if len(parts) >= 6:
            try:
                conf = float(parts[5])
                if conf >= threshold:
                    return True
            except ValueError:
                continue
        elif len(parts) == 5:
            return True

    return False


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate document-level forged/clean decision from YOLO predictions."
    )

    parser.add_argument("--dataset", type=str, default="dataset", help="Path to YOLO dataset folder")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument(
        "--pred-labels",
        type=str,
        required=True,
        help="Path to YOLO prediction labels folder, usually outputs/predictions/<run>/labels",
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out", type=str, default="outputs/document_level_eval.csv", help="Output CSV path")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images" / args.split
    gt_labels_dir = dataset_dir / "labels" / args.split
    pred_labels_dir = Path(args.pred_labels)

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    if not gt_labels_dir.exists():
        raise FileNotFoundError(f"Missing ground-truth labels directory: {gt_labels_dir}")

    if not pred_labels_dir.exists():
        raise FileNotFoundError(f"Missing prediction labels directory: {pred_labels_dir}")

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    rows = []

    tp = fp = tn = fn = 0

    for image_path in image_paths:
        gt_label_path = gt_labels_dir / f"{image_path.stem}.txt"
        pred_label_path = pred_labels_dir / f"{image_path.stem}.txt"

        gt_tampered = has_gt_box(gt_label_path)
        pred_tampered = has_prediction(pred_label_path, args.threshold)

        if gt_tampered and pred_tampered:
            tp += 1
            result = "TP"
        elif not gt_tampered and pred_tampered:
            fp += 1
            result = "FP"
        elif not gt_tampered and not pred_tampered:
            tn += 1
            result = "TN"
        else:
            fn += 1
            result = "FN"

        rows.append(
            {
                "image": str(image_path),
                "gt_tampered": int(gt_tampered),
                "pred_tampered": int(pred_tampered),
                "threshold": args.threshold,
                "result": result,
            }
        )

    total = tp + fp + tn + fn

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "gt_tampered", "pred_tampered", "threshold", "result"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDocument-Level Evaluation")
    print("-" * 50)
    print(f"Split:      {args.split}")
    print(f"Threshold:  {args.threshold}")
    print(f"Images:     {total}")
    print(f"TP:         {tp}")
    print(f"FP:         {fp}")
    print(f"TN:         {tn}")
    print(f"FN:         {fn}")
    print("-" * 50)
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"\nSaved per-image results to: {out_path}")


if __name__ == "__main__":
    main()