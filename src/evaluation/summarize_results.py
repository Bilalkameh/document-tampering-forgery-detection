from pathlib import Path
import argparse
import csv
from datetime import datetime


def fmt(x):
    return f"{x:.3f}"


def improvement(new, old):
    diff = new - old
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate clean report-ready evaluation summary for YOLO experiments."
    )

    parser.add_argument("--out-dir", type=str, default="outputs/evaluation_summary")
    parser.add_argument("--title", type=str, default="Document Tampering Detection Evaluation Summary")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manually recorded from validation/test outputs.
    # Update/add rows here when new experiments are completed.
    results = [
        {
            "model": "YOLOv8n Baseline",
            "split": "Validation",
            "precision": 0.892,
            "recall": 0.844,
            "map50": 0.887,
            "map5095": 0.713,
            "notes": "30 epochs, imgsz=640, batch=8",
        },
        {
            "model": "YOLOv8n Baseline",
            "split": "Test",
            "precision": 0.826,
            "recall": 0.865,
            "map50": 0.888,
            "map5095": 0.708,
            "notes": "Baseline test evaluation",
        },
        {
            "model": "YOLOv8s Optimized",
            "split": "Validation",
            "precision": 0.970,
            "recall": 0.859,
            "map50": 0.941,
            "map5095": 0.749,
            "notes": "YOLOv8s, imgsz=768, batch=4, optimized augmentation",
        },
        {
            "model": "YOLOv8s Optimized",
            "split": "Test",
            "precision": 0.936,
            "recall": 0.852,
            "map50": 0.930,
            "map5095": 0.727,
            "notes": "Final candidate test evaluation",
        },
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_path = out_dir / "evaluation_summary.csv"
    md_path = out_dir / "evaluation_summary.md"
    txt_path = out_dir / "evaluation_summary.txt"

    # Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "split", "precision", "recall", "map50", "map5095", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Extract test rows for comparison
    baseline_test = next(r for r in results if r["model"] == "YOLOv8n Baseline" and r["split"] == "Test")
    optimized_test = next(r for r in results if r["model"] == "YOLOv8s Optimized" and r["split"] == "Test")

    comparison_rows = [
        ("Precision", baseline_test["precision"], optimized_test["precision"]),
        ("Recall", baseline_test["recall"], optimized_test["recall"]),
        ("mAP@0.5", baseline_test["map50"], optimized_test["map50"]),
        ("mAP@0.5:0.95", baseline_test["map5095"], optimized_test["map5095"]),
    ]

    selected_model = "YOLOv8s Optimized"
    final_model_path = "runs/detect/sidtd_yolov8s_opt768_b4_w2/weights/best.pt"

    md = []
    md.append(f"# {args.title}\n")
    md.append(f"Generated: {timestamp}\n")

    md.append("## 1. Experiment Summary\n")
    md.append("| Model | Split | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Notes |")
    md.append("|---|---:|---:|---:|---:|---:|---|")

    for r in results:
        md.append(
            f"| {r['model']} | {r['split']} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
            f"{fmt(r['map50'])} | {fmt(r['map5095'])} | {r['notes']} |"
        )

    md.append("\n## 2. Test Set Comparison\n")
    md.append("| Metric | YOLOv8n Baseline | YOLOv8s Optimized | Difference |")
    md.append("|---|---:|---:|---:|")

    for metric, base, opt in comparison_rows:
        md.append(f"| {metric} | {fmt(base)} | {fmt(opt)} | {improvement(opt, base)} |")

    md.append("\n## 3. Final Model Selection\n")
    md.append(
        "The optimized YOLOv8s model was selected as the final model because it improved "
        "the main detection metrics on the test set, especially precision, mAP@0.5, and "
        "mAP@0.5:0.95. Recall decreased only slightly compared with the baseline, while "
        "the localization quality improved."
    )

    md.append("\n\nFinal selected model:\n")
    md.append(f"```text\n{final_model_path}\n```")

    md.append("\n## 4. Short Report Paragraph\n")
    md.append(
        "We first trained a YOLOv8n baseline detector for tampered-region localization. "
        "The baseline achieved 0.888 mAP@0.5 and 0.708 mAP@0.5:0.95 on the test set. "
        "We then trained an optimized YOLOv8s model using a higher input resolution and "
        "document-specific augmentation choices. The optimized model achieved 0.930 "
        "mAP@0.5 and 0.727 mAP@0.5:0.95 on the test set. Precision improved from "
        "0.826 to 0.936, while recall decreased only slightly from 0.865 to 0.852. "
        "Therefore, the optimized YOLOv8s model was selected as the final detector."
    )

    md_content = "\n".join(md)
    md_path.write_text(md_content, encoding="utf-8")

    # Plain text version for quick sharing
    txt = f"""
{args.title}
Generated: {timestamp}

FINAL TEST COMPARISON

YOLOv8n Baseline:
  Precision:    {fmt(baseline_test['precision'])}
  Recall:       {fmt(baseline_test['recall'])}
  mAP@0.5:      {fmt(baseline_test['map50'])}
  mAP@0.5:0.95: {fmt(baseline_test['map5095'])}

YOLOv8s Optimized:
  Precision:    {fmt(optimized_test['precision'])}
  Recall:       {fmt(optimized_test['recall'])}
  mAP@0.5:      {fmt(optimized_test['map50'])}
  mAP@0.5:0.95: {fmt(optimized_test['map5095'])}

IMPROVEMENT:
  Precision:    {improvement(optimized_test['precision'], baseline_test['precision'])}
  Recall:       {improvement(optimized_test['recall'], baseline_test['recall'])}
  mAP@0.5:      {improvement(optimized_test['map50'], baseline_test['map50'])}
  mAP@0.5:0.95: {improvement(optimized_test['map5095'], baseline_test['map5095'])}

Selected final model:
  {final_model_path}

Conclusion:
  The optimized YOLOv8s model is selected as the final detector because it improves
  precision and localization quality while only slightly reducing recall.
""".strip()

    txt_path.write_text(txt, encoding="utf-8")

    print("Evaluation summary generated.")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Text: {txt_path}")


if __name__ == "__main__":
    main()