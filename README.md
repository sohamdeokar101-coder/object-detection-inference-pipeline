# 👁️ Batch Object Detection & Vision Inference Pipeline

> A standard style for README files

A pure Machine Learning batch inference pipeline built with **YOLOv8**, **OpenCV**, **NumPy**, **Pandas**, and **Rich**. This engine ingests multi-source image data, executes object detection, annotates bounding box visualizations directly on image arrays with class-specific colors, and exports structured CSV/JSON performance metric reports.

[![Standard Readme Compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PyTest](https://img.shields.io/badge/PyTest-Passed-brightgreen.svg?style=flat-square)](https://docs.pytest.org/)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Generator](#generator)
- [Badge](#badge)
- [Example READMEs](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [License](#license)

---

## Background

Deploying computer vision models in real-world machine learning engineering requires efficient batch processing without web server framework overhead. Traditional web API wrappers add network latency when processing bulk image data.

This project implements an operationalized **Batch Object Detection Engine**:
- **Multi-Source Ingestion:** Ingests local files or downloads real-world datasets from online storage endpoint URLs.
- **YOLOv8 Core ML Engine:** Executes GPU/CPU computer vision inference, extracting bounding box coordinates ($x_{\min}, y_{\min}, x_{\max}, y_{\max}$), confidence scores, and class labels.
- **Class-Specific Color Annotations:** Dynamically maps distinct BGR colors to object classes (e.g., Blue for bus, Green for person, Magenta for frisbee) and draws bounding overlays directly onto OpenCV image arrays.
- **Structured Metrics Reporting:** Exports structured prediction metrics to `output_reports/detection_metrics.csv` and `output_reports/detection_metrics.json`.
- **Rich Terminal UI:** Renders color-coded, formatted terminal tables detailing detection confidence and bounding box properties.

---


## Install

### Prerequisites
- Python 3.11 or higher
- Git

### Setup

Clone the repository and set up your Python virtual environment:

```bash

object-detection-inference-pipeline/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Multi-source dataset ingestion & directory scanner
│   ├── detector.py         # YOLOv8 engine, inference, & OpenCV dynamic color mapping
│   └── pipeline.py         # Batch orchestrator & CSV/JSON report generators
├── sample_images/
│   ├── download_sample.py  # Utility to pull real-world images from online endpoints
│   └── create_sample.py    # Synthetic image generator
├── output_annotated/       # Rendered images with visual bounding box overlays (git-ignored)
├── output_reports/         # Generated detection_metrics.csv & detection_metrics.json
├── tests/
│   ├── __init__.py
│   └── test_detector.py    # Automated PyTest suite for inference validation
├── main.py                 # Master CLI batch inference execution entrypoint
├── requirements.txt        # Production dependencies
├── .gitignore              # Version control exclusions
└── README.md               # Technical system documentation


# Clone the repository
git clone [https://github.com/sohamdeokar101-coder/object-detection-inference-pipeline.git](https://github.com/sohamdeokar101-coder/object-detection-inference-pipeline.git)
cd object-detection-inference-pipeline

# Initialize and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

💻 Sample Pipeline Output & Analytics
Terminal Console Summary Output
Plaintext
               --- Summary Detection Report ---               
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Image Name           ┃ Class Name ┃ Confidence ┃ Bounding Box Color ┃
┣━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ real_bus.jpg         ┃ bus        ┃     0.8729 ┃ Blue               ┃
┃ real_bus.jpg         ┃ person     ┃     0.8653 ┃ Green              ┃
┃ real_bus.jpg         ┃ person     ┃     0.8540 ┃ Green              ┃
┃ real_bus.jpg         ┃ person     ┃     0.8252 ┃ Green              ┃
┃ real_bus.jpg         ┃ person     ┃     0.2673 ┃ Green              ┃
┃ synthetic_sample.jpg ┃ frisbee    ┃     0.5014 ┃ Magenta            ┃
┃ synthetic_sample.jpg ┃ kite       ┃     0.3179 ┃ Orange             ┃
┗━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┻━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛

