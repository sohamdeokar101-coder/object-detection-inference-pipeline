import os
import pytest
from src.detector import ObjectDetectorEngine
from src.dataset import load_images_from_directory


def test_detector_engine_initialization():
    """Validates that detector engine correctly loads model weights."""
    engine = ObjectDetectorEngine(model_name="yolov8n.pt")
    assert engine.model is not None


def test_inference_on_sample_image():
    """Validates inference output format on local real sample image."""
    sample_file = "sample_images/real_bus.jpg"
    assert os.path.exists(sample_file), "Real sample image missing."

    engine = ObjectDetectorEngine(model_name="yolov8n.pt")
    detections, annotated_img = engine.predict_image(sample_file, conf_threshold=0.25)

    assert isinstance(detections, list)
    assert len(detections) > 0
    assert annotated_img.shape[2] == 3  # Ensure 3-channel BGR image
    assert "class_name" in detections[0]
    assert any(d["class_name"] in ["bus", "person"] for d in detections)
