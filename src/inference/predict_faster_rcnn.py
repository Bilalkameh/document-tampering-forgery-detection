from pathlib import Path
import argparse

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_model(num_classes: int = 2) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def predict(weights: str, source: str, out_name: str, conf: float, img_size: int, device_str: str):
    device = torch.device(device_str if device_str != "mps" else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model = build_model(num_classes=2)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    source_dir = Path(source)
    image_paths = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    labels_dir = Path("outputs/predictions") / out_name / "labels"
    images_dir = Path("outputs/predictions") / out_name / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Faster R-CNN inference on {len(image_paths)} images (conf >= {conf})")

    with torch.no_grad():
        for img_path in image_paths:
            orig = Image.open(img_path).convert("RGB")
            orig_w, orig_h = orig.size
            resized = orig.resize((img_size, img_size))
            img_tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

            preds = model(img_tensor)[0]

            boxes  = preds["boxes"].cpu()
            scores = preds["scores"].cpu()
            keep   = scores >= conf
            boxes  = boxes[keep]
            scores = scores[keep]

            # Save YOLO-format label (class cx cy w h conf), coords normalized to original image size
            label_lines = []
            for box, score in zip(boxes.tolist(), scores.tolist()):
                x1, y1, x2, y2 = box
                # Scale back from img_size to original dimensions
                x1 = x1 / img_size * orig_w
                y1 = y1 / img_size * orig_h
                x2 = x2 / img_size * orig_w
                y2 = y2 / img_size * orig_h
                cx = (x1 + x2) / 2 / orig_w
                cy = (y1 + y2) / 2 / orig_h
                w  = (x2 - x1) / orig_w
                h  = (y2 - y1) / orig_h
                label_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.6f}")

            (labels_dir / (img_path.stem + ".txt")).write_text("\n".join(label_lines))

            # Save annotated image
            draw = ImageDraw.Draw(orig)
            for box, score in zip(boxes.tolist(), scores.tolist()):
                x1 = box[0] / img_size * orig_w
                y1 = box[1] / img_size * orig_h
                x2 = box[2] / img_size * orig_w
                y2 = box[3] / img_size * orig_h
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, max(0, y1 - 12)), f"{score:.2f}", fill="red")
            orig.save(images_dir / img_path.name)

    print(f"Saved labels → {labels_dir}")
    print(f"Saved images → {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Faster R-CNN inference on document images.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source",  type=str, default="dataset/images/test")
    parser.add_argument("--conf",    type=float, default=0.5)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--name",    type=str,   default="fasterrcnn_predictions")
    parser.add_argument("--device",  type=str,   default="cpu")
    args = parser.parse_args()

    predict(args.weights, args.source, args.name, args.conf, args.imgsz, args.device)


if __name__ == "__main__":
    main()
