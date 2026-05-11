from pathlib import Path
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Validate a trained YOLO model.")

    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLO weights")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, 0, 1, etc.")
    parser.add_argument("--name", type=str, default="validation", help="Validation run name")

    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    print("Running YOLO validation")
    print(f"Weights: {args.weights}")
    print(f"Data:    {args.data}")
    print(f"Split:   {args.split}")
    print(f"Image:   {args.imgsz}")
    print(f"Batch:   {args.batch}")
    print(f"Device:  {args.device}")

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="outputs/validation",
        name=args.name,
        plots=True,
    )

    print("Validation finished.")
    print(metrics)


if __name__ == "__main__":
    main()