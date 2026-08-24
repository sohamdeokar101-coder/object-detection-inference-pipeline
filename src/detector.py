import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any, List, Tuple


class ObjectDetectorEngine:
    """Core YOLOv8 inference and multi-class visualization engine."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        print(f"⚙️ Loading YOLO model weights: '{model_name}'...")
        self.model = YOLO(model_name)
        # Pre-defined bright color palette in BGR format
        self.class_colors = {
            "bus": (255, 0, 0),        # Blue
            "person": (0, 255, 0),     # Green
            "car": (0, 255, 255),      # Yellow
            "frisbee": (255, 0, 255),  # Magenta
            "kite": (0, 165, 255),     # Orange
        }

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Returns BGR color tuple for a class name."""
        return self.class_colors.get(class_name, (0, 255, 0))

    def predict_image(
        self, image_path: str, conf_threshold: float = 0.25
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Runs object detection across multiple objects and returns structured metrics with color tags."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        img_bgr = cv2.imread(image_path)
        results = self.model(img_bgr, conf=conf_threshold)[0]

        detections = []
        annotated_img = img_bgr.copy()

        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = self.model.names[cls_id]

            x1, y1, x2, y2 = map(int, coords)
            box_color = self.get_color(class_name)

            detections.append({
                "image_name": os.path.basename(image_path),
                "class_name": class_name,
                "confidence": round(conf, 4),
                "color_bgr": f"BGR{box_color}",  # Store color string representation
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
            })

            # Draw visual bounding box onto image
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                annotated_img,
                label,
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
            )

        return detections, annotated_img