from pathlib import Path
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on document images.")

    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLO weights")
    parser.add_argument("--source", type=str, default="dataset/images/test", help="Image folder or image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--name", type=str, default="tamper_predictions", help="Prediction run name")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, 0, 1, etc.")

    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    print("Running YOLO inference")
    print(f"Weights: {args.weights}")
    print(f"Source:  {args.source}")
    print(f"Conf:    {args.conf}")
    print(f"Device:  {args.device}")

    model = YOLO(args.weights)

    model.predict(
        source=args.source,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project="outputs/predictions",
        name=args.name,
    )

    print("Prediction finished.")
    print(f"Saved results to outputs/predictions/{args.name}")


if __name__ == "__main__":
    main()