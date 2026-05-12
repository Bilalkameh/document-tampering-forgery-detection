from pathlib import Path
import csv
import argparse
import cv2


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_yolo_boxes(label_path: Path):
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

        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else None
        boxes.append((cls, cx, cy, w, h, conf))

    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    _, cx, cy, w, h, conf = box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2, conf


def draw_boxes(img, boxes, color, label_prefix):
    h, w = img.shape[:2]

    for box in boxes:
        x1, y1, x2, y2, conf = yolo_to_xyxy(box, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        if conf is not None:
            label = f"{label_prefix} {conf:.2f}"
        else:
            label = label_prefix

        cv2.putText(
            img,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return img


def find_prediction_image(pred_dir: Path, image_stem: str):
    for ext in IMAGE_EXTS:
        candidate = pred_dir / f"{image_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def make_side_by_side(image_path, gt_label_path, pred_label_path, out_path, title):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    gt_img = img.copy()
    pred_img = img.copy()

    gt_boxes = read_yolo_boxes(gt_label_path)
    pred_boxes = read_yolo_boxes(pred_label_path)

    gt_img = draw_boxes(gt_img, gt_boxes, (0, 0, 255), "GT")
    pred_img = draw_boxes(pred_img, pred_boxes, (255, 0, 0), "PRED")

    h, w = img.shape[:2]

    cv2.putText(gt_img, "Ground Truth", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(pred_img, "Prediction", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    combined = cv2.hconcat([gt_img, pred_img])

    cv2.putText(
        combined,
        title,
        (20, combined.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", default="outputs/document_eval_yolov8s_test_025.csv")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--pred-labels", default="runs/detect/outputs/predictions/sidtd_yolov8s_test_predictions_025/labels")
    parser.add_argument("--out-dir", default="Presentation Pictures/side_by_side")
    parser.add_argument("--per-class", type=int, default=2)

    args = parser.parse_args()

    eval_csv = Path(args.eval_csv)
    dataset = Path(args.dataset)
    pred_labels = Path(args.pred_labels)
    out_dir = Path(args.out_dir)

    rows_by_result = {"TP": [], "TN": [], "FP": [], "FN": []}

    with open(eval_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = row["result"]
            if result in rows_by_result:
                rows_by_result[result].append(row)

    for result, rows in rows_by_result.items():
        selected = rows[: args.per_class]

        for i, row in enumerate(selected, start=1):
            image_path = Path(row["image"])
            image_stem = image_path.stem

            split = image_path.parent.name
            gt_label_path = dataset / "labels" / split / f"{image_stem}.txt"
            pred_label_path = pred_labels / f"{image_stem}.txt"

            out_path = out_dir / result / f"{result}_{i}_{image_stem}.jpg"

            title = {
                "TP": "True Positive: tampered image correctly detected",
                "TN": "True Negative: clean image correctly ignored",
                "FP": "False Positive: clean image incorrectly flagged",
                "FN": "False Negative: tampered image missed",
            }[result]

            make_side_by_side(image_path, gt_label_path, pred_label_path, out_path, title)

    print(f"Presentation examples saved to: {out_dir}")


if __name__ == "__main__":
    main()