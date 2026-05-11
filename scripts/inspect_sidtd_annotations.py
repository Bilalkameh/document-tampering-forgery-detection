from pathlib import Path
from PIL import Image, ImageDraw
import json
import argparse


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def find_json_files(root: Path):
    return list(root.rglob("*.json"))


def is_quad(value):
    """
    Checks if a value looks like:
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    if not isinstance(value, list) or len(value) != 4:
        return False

    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            return False
        if not all(isinstance(v, (int, float)) for v in point):
            return False

    return True


def extract_fields_with_quads(obj, prefix=""):
    """
    Recursively searches a JSON object for fields that contain a 'quad'.
    Returns:
    [
      {
        "field_name": "...",
        "quad": [[x,y], ...],
        "value": "..."
      }
    ]
    """
    results = []

    if isinstance(obj, dict):
        if "quad" in obj and is_quad(obj["quad"]):
            field_name = prefix.split(".")[-1] if prefix else "unknown"
            results.append({
                "field_name": field_name,
                "quad": obj["quad"],
                "value": obj.get("value", "")
            })

        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            results.extend(extract_fields_with_quads(value, new_prefix))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            results.extend(extract_fields_with_quads(item, new_prefix))

    return results


def quad_to_bbox(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]

    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(xs))
    y2 = int(max(ys))

    return x1, y1, x2, y2


def bbox_to_yolo(x1, y1, x2, y2, image_width, image_height):
    box_width = x2 - x1
    box_height = y2 - y1

    center_x = x1 + box_width / 2
    center_y = y1 + box_height / 2

    return (
        center_x / image_width,
        center_y / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def collect_images(root: Path):
    images = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def find_matching_image(json_path: Path, all_images):
    """
    Tries to find an image that belongs to the same document annotation.

    This is heuristic because SIDTD/MIDV folder structures can vary.
    We first try same stem, then nearby folder.
    """
    json_stem = json_path.stem.lower()

    # SIDTD test samples use names like:
    #   alb_0_id.json       -> alb00.jpg
    #   alb_1_id.json       -> alb01.jpg
    #   aze_0_passport.json -> aze00.jpg
    parts = json_stem.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        sidtd_sample_stem = f"{parts[0]}{int(parts[1]):02d}"
        for img_path in all_images:
            if img_path.stem.lower() == sidtd_sample_stem:
                return img_path

    # Try exact same stem
    for img_path in all_images:
        if img_path.stem.lower() == json_stem:
            return img_path

    # Try image in same folder or nearby folder
    json_parent_parts = set(part.lower() for part in json_path.parts)

    candidates = []
    for img_path in all_images:
        shared_parts = len(json_parent_parts.intersection(set(part.lower() for part in img_path.parts)))
        candidates.append((shared_parts, img_path))

    candidates.sort(reverse=True, key=lambda x: x[0])

    if candidates:
        return candidates[0][1]

    return None


def draw_boxes(image_path, fields, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for field in fields:
        x1, y1, x2, y2 = quad_to_bbox(field["quad"])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), field["field_name"], fill="red")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/raw",
        help="Root folder where SIDTD templates/images/json files are stored"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of JSON files to inspect"
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path("outputs/visual_checks/sidtd_annotations")

    json_files = find_json_files(root)
    images = collect_images(root)

    print(f"Root: {root}")
    print(f"Found JSON files: {len(json_files)}")
    print(f"Found image files: {len(images)}")

    if not json_files:
        raise SystemExit("No JSON annotation files found. Check the --root path.")

    inspected = 0

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skipping unreadable JSON: {json_path} | {exc}")
            continue

        fields = extract_fields_with_quads(data)

        if not fields:
            continue

        print("\n" + "=" * 80)
        print(f"JSON: {json_path}")
        print(f"Fields with quads: {len(fields)}")

        for field in fields[:10]:
            x1, y1, x2, y2 = quad_to_bbox(field["quad"])
            print(
                f"- field={field['field_name']} "
                f"value={field['value']} "
                f"bbox=({x1},{y1},{x2},{y2})"
            )

        matching_image = find_matching_image(json_path, images)

        if matching_image:
            print(f"Possible matching image: {matching_image}")

            try:
                img = Image.open(matching_image).convert("RGB")
                width, height = img.size

                first_box = quad_to_bbox(fields[0]["quad"])
                yolo = bbox_to_yolo(*first_box, width, height)

                print(f"Image size: {width}x{height}")
                print(
                    "First field YOLO box: "
                    f"0 {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}"
                )

                output_path = output_dir / f"{json_path.stem}_boxed.jpg"
                draw_boxes(matching_image, fields, output_path)
                print(f"Saved visual check: {output_path}")

            except Exception as exc:
                print(f"Could not visualize matching image: {exc}")
        else:
            print("No matching image found.")

        inspected += 1

        if inspected >= args.max_files:
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
