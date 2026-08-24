import os
import json
import cv2
import pandas as pd
from typing import List, Dict, Any
from src.detector import ObjectDetectorEngine


def run_batch_inference_pipeline(
    image_paths: List[str],
    output_img_dir: str = "output_annotated",
    output_report_dir: str = "output_reports",
    conf_threshold: float = 0.25,
) -> pd.DataFrame:
    """Processes a batch of images, saves annotated outputs, and logs metrics to CSV/JSON."""
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_report_dir, exist_ok=True)

    engine = ObjectDetectorEngine(model_name="yolov8n.pt")
    all_detections: List[Dict[str, Any]] = []

    print(f"\n🚀 Running Inference Pipeline on {len(image_paths)} images...\n")

    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        print(f"🔍 Processing: {file_name}")

        detections, annotated_img = engine.predict_image(img_path, conf_threshold=conf_threshold)
        all_detections.extend(detections)

        # Save annotated image output
        output_image_path = os.path.join(output_img_dir, f"detected_{file_name}")
        cv2.imwrite(output_image_path, annotated_img)
        print(f"   └─ 🖼️ Saved annotated image -> '{output_image_path}' ({len(detections)} objects)")

    # Build Pandas DataFrame report
    df_report = pd.DataFrame(all_detections)

    csv_path = os.path.join(output_report_dir, "detection_metrics.csv")
    json_path = os.path.join(output_report_dir, "detection_metrics.json")

    df_report.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2)

    print(f"\n✅ Batch Inference Complete!")
    print(f"📊 Exported CSV Report: '{csv_path}'")
    print(f"📊 Exported JSON Report: '{json_path}'")

    return df_report