# Samsung Image Defect Detection 

A two-stage Deep Learning pipeline designed to automatically detect and classify visual artifacts and defects in images.

---

##  Pipeline Architecture

This project uses a modular approach, split into three main components:

### 1. Phase 1: Binary Detection (`binary_model.py`)

- Scans the image using a sliding window (50x50px)
- Uses a Binary CNN to detect regions containing *any* distortion
- Applies Non-Maximum Suppression (NMS) to merge overlapping bounding boxes

### 2. Phase 2: Multi-Class Classification (`classification_model.py`)

- Crops suspected defect patches from Phase 1
- Uses a ResNet-based CNN (`SamsungDefectDetector`)
- Classifies the specific *type* of defect

### 3. The Orchestrator (`pipeline.py`)

- Main execution script
- Connects detection and classification stages
- Handles data flow and output generation

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/Snafuzila/Samsung-Artifact-Detection.git
cd Samsung-Artifact-Detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate environment:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision pandas pillow tqdm
```


---

## How to Use

### 1. Command Line Interface (CLI)

**Basic run:**

```bash
python pipeline.py path/to/your_image
```

This generates a timestamped JSON file automatically, for example:

```
defect_report_20260430_153000.json
```

**Advanced run:**

```bash
python pipeline.py path/to/your_image.bmp --threshold 0.95 --iou 0.1 --output final_results.json
```

### 2. Programmatically (Python)

```python
from pipeline import run_pipeline

run_pipeline(
    image_path="test_images/TE42@gt.bmp",
    threshold=0.90,
    iou=0.05,
    output_filename="my_custom_results.json"
)
```

---

## Parameters Guide

| Parameter     | Default | Description                            | Tuning Advice                                                                 |
|---------------|---------|----------------------------------------|-------------------------------------------------------------------------------|
| `--threshold` | `0.90`  | Confidence for binary detection | Lower (0.70) → higher recall, more detections. Higher (0.98) → fewer false positives |
| `--iou`       | `0.05`  | IoU threshold for NMS                  | Lower → merges nearby boxes. Higher → keeps close detections separate         |
| `--output`    | Auto    | Output JSON filename                   | default `defect_report_date_time.json`                                            |


Additional In-Code Parameters (Internal)
These parameters govern the underlying mechanics of the image processing. They are hardcoded within the Python files for optimal baseline performance, but can be modified in the code for specific use cases:

#### WINDOW_SIZE (int, default: 50)
Effect: Defines the dimensions (50x50 pixels) of the square patch extracted during the sliding window phase and the expected input size for both CNN models.
<span style="color:red">Note: Changing this requires retraining the models, as they expect a fixed input dimension.</span>

#### STRIDE (int, default: 10)
Effect: The step size (in pixels) the sliding window takes as it moves across the image.
Tuning: A smaller stride (e.g., 5) provides higher resolution detection but significantly increases inference time. A larger stride (e.g., 50) speeds up processing but might miss very small or edge-case defects.

#### MARGIN (int, default: 20)
Effect: Inflates the detected bounding box by this amount (in pixels) before applying NMS.
Tuning: This is used to aggressively cluster nearby positive detections into a single event. Increase the margin if the pipeline outputs multiple distinct points for a single, large artifact. Decrease it if distinct, nearby defects are mistakenly merged into one.

---
## Output Format

The pipeline outputs a JSON file containing all detected defects.

Each entry includes:

- Bounding box coordinates
- Binary detection score
- Predicted class name
- Predicted class index
- Classification confidence

**Example (`defect_report.json`):**

```json
[
    {
        "box": [1815.0, 705.0, 1865.0, 755.0],
        "binary_score": 0.9995,
        "predicted_class_name": "Edge False Color",
        "predicted_class_idx": 1,
        "classification_confidence": 0.9842
    },
    {
        "box": [6025.0, 735.0, 6075.0, 785.0],
        "binary_score": 0.9876,
        "predicted_class_name": "Broken Line",
        "predicted_class_idx": 0,
        "classification_confidence": 0.9102
    }
]
```
