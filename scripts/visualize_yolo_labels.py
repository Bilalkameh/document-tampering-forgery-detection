from pathlib import Path
from PIL import Image, ImageDraw
import random


DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs/visual_checks/yolo_export")

CLASS_NAMES = {
    0: "tampered_region"
}


def yolo_to_bbox(cx, cy, bw, bh, image_width, image_height):
    """
    Converts YOLO normalized box to pixel coordinates.
    """
    box_width = bw * image_width
    box_height = bh * image_height

    x1 = (cx * image_width) - box_width / 2
    y1 = (cy * image_height) - box_height / 2
    x2 = x1 + box_width
    y2 = y1 + box_height

    return int(x1), int(y1), int(x2), int(y2)


def read_yolo_labels(label_path: Path):
    """
    Reads a YOLO label file.
    Empty files return an empty list.
    """
    if not label_path.exists():
        return []

    text = label_path.read_text().strip()

    if not text:
        return []

    boxes = []

    for line in text.splitlines():
        parts = line.strip().split()

        if len(parts) != 5:
            print(f"Skipping invalid label line in {label_path}: {line}")
            continue

        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        boxes.append((class_id, cx, cy, bw, bh))

    return boxes


def draw_labels(image_path: Path, label_path: Path, output_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size
    boxes = read_yolo_labels(label_path)

    for class_id, cx, cy, bw, bh in boxes:
        x1, y1, x2, y2 = yolo_to_bbox(cx, cy, bw, bh, width, height)

        class_name = CLASS_NAMES.get(class_id, str(class_id))
        text = f"{class_name}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, max(0, y1 - 15)), text, fill="red")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def collect_image_label_pairs(split: str):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    pairs = []

    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        pairs.append((image_path, label_path))

    return pairs


def main():
    random.seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        pairs = collect_image_label_pairs(split)

        clean_pairs = []
        tampered_pairs = []

        for image_path, label_path in pairs:
            labels = read_yolo_labels(label_path)

            if labels:
                tampered_pairs.append((image_path, label_path))
            else:
                clean_pairs.append((image_path, label_path))

        print(f"{split}: {len(clean_pairs)} clean, {len(tampered_pairs)} tampered")

        # Save a few tampered examples with boxes
        sample_tampered = random.sample(
            tampered_pairs,
            min(10, len(tampered_pairs))
        )

        for image_path, label_path in sample_tampered:
            output_path = OUTPUT_DIR / split / f"{image_path.stem}_boxed.jpg"
            draw_labels(image_path, label_path, output_path)

        # Save a few clean examples too; they should have no boxes
        sample_clean = random.sample(
            clean_pairs,
            min(5, len(clean_pairs))
        )

        for image_path, label_path in sample_clean:
            output_path = OUTPUT_DIR / split / f"{image_path.stem}_clean_check.jpg"
            draw_labels(image_path, label_path, output_path)

    print(f"Visual checks saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
