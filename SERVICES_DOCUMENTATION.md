# PsoMetric Backend - Services Documentation

## Overview
This document provides comprehensive documentation for the AI-powered psoriasis assessment engine implemented in `app/services.py`. The system uses computer vision and deep learning to analyze psoriasis lesions following PASI (Psoriasis Area and Severity Index) clinical standards.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [AI Models](#ai-models)
3. [Core Components](#core-components)
4. [Scoring Methods](#scoring-methods)
5. [Image Processing Pipeline](#image-processing-pipeline)
6. [Research Basis](#research-basis)
7. [API Usage](#api-usage)

---

## System Architecture

### AIEngine Class
Main class that orchestrates the psoriasis assessment system.

**Dependencies:**
- PyTorch (deep learning)
- OpenCV (computer vision)
- YOLO (object detection)
- EfficientNet (classification)
- PIL (image processing)

**Device Support:** Automatically detects and uses CUDA (GPU) if available, falls back to CPU.

---

## AI Models

### 1. Sniper (YOLO-based Segmentation)
**Purpose:** Detects and segments psoriasis lesions in images.

**Model Details:**
- Architecture: YOLOv8 (custom trained)
- File: `models/sniper.pt`
- Function: Identifies lesion boundaries with pixel-level precision
- Confidence Threshold: 0.10 (optimized for detecting mild cases)

### 2. Judge (EfficientNet Classifier)
**Purpose:** Provides baseline severity classification.

**Model Details:**
- Architecture: EfficientNet-B0
- File: `models/judge.pth`
- Input Size: 224x224 pixels
- Output Classes: 3 (Mild, Moderate, Severe)
- Normalization: ImageNet standards [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]

**Baseline Scoring:**
- Mild: Weight = 1.0
- Moderate: Weight = 2.2
- Severe: Weight = 3.8

---

## Core Components

### Image Preprocessing

#### `white_balance(cv_img)`
Corrects color cast from indoor lighting using Gray World assumption.

**Purpose:** Normalizes lighting conditions for accurate color analysis.

**Algorithm:**
1. Calculate mean values for B, G, R channels
2. Compute global average: k = (B_mean + G_mean + R_mean) / 3
3. Scale each channel: channel = channel * (k / channel_mean)
4. Clip values to valid range [0, 255]

**Use Cases:**
- Erythema (redness) scoring
- Induration (thickness) assessment
- NOT used for desquamation (preserves natural scale appearance)

#### `image_to_base64(numpy_image)`
Converts OpenCV image to Base64 encoded JPEG.

**Purpose:** Enables image transmission over REST API.

**Process:**
1. Convert BGR to RGB color space
2. Create PIL image
3. Save to BytesIO buffer (70% JPEG quality)
4. Encode to Base64 string

---

## Scoring Methods

### PASI Scoring Scale
All metrics follow standard PASI 0-4 scale:
- **0:** None (absent)
- **1:** Slight (barely visible)
- **2:** Moderate (clearly visible)
- **3:** Marked (prominent)
- **4:** Very marked (severe)

---

### 1. Erythema (Redness) Scoring

#### Method: `_calculate_erythema_score()`
**Research Basis:** Colorimetry standards from dermatology literature.

**Validated Metrics:**

1. **Erythema Index (EI)**
   - Formula: `EI = R - G`
   - Validated in clinical studies
   - Correlates with inflammation severity

2. **L\*a\*b\* Color Space Analysis** (Gold Standard)
   - L: Lightness (0-255)
   - a*: Red-green axis (positive = red)
   - b*: Yellow-blue axis
   - Normalized a*: `a* / L` (accounts for brightness)

3. **HSV Hue Validation**
   - Red hue range: 0-12° or 158-180° (OpenCV scale)
   - Confirms true red color vs. other colors

4. **Vascular Pattern Analysis**
   - Measures density of highly saturated red pixels
   - Indicates inflammation intensity

**Scoring Thresholds:**
| Score | Erythema Index | L\*a\*b\* a* | Description |
|-------|---------------|--------------|-------------|
| 1     | < 20          | < 140        | Slight pink |
| 2     | 20-35         | 140-150      | Light red   |
| 3     | 35-55         | 150-165      | Moderate red|
| 4     | > 55          | > 165        | Dark/crimson|

**Image Type:** Uses WHITE-BALANCED image for accurate color assessment.

---

### 2. Desquamation (Scaling) Scoring

#### Method: `_calculate_desquamation_score()`
**Research Basis:** Texture analysis methods from "Automated Assessment of Psoriasis Lesions" (2018).

**Detection Methods:**

1. **Color-based Scale Detection**
   - White scales: Low saturation (< 45), high brightness (> 90)
   - Silver scales: Low saturation (< 35), medium brightness (65-130)
   - Weighted combination: silver scales = 0.7x weight

2. **Texture Entropy Analysis**
   - Morphological gradient (9x9 kernel)
   - Detects high local variation characteristic of scales

3. **Multi-scale Edge Detection**
   - Fine edges (Canny 20-60): Fine scales
   - Coarse edges (Canny 50-120): Thick scales
   - Laplacian: Overall roughness (threshold > 12)

4. **Local Binary Pattern (LBP) Inspired**
   - Gaussian blur comparison
   - Detects scale patterns (variance > 10)

5. **Brightness Contrast**
   - Scales are lighter than underlying lesion
   - Contrast = scale_brightness - lesion_brightness

**Scale Classification:**
- **Fine scales:** Small, sparse, less prominent
- **Coarse scales:** Thick, dense, more prominent (weighted 1.5x)

**Scoring Thresholds:**
| Score | Coverage | Type | Description |
|-------|----------|------|-------------|
| 0-1   | < 3%     | None/fine | Absent to slight |
| 2     | 3-8%     | Fine-medium | Moderate visible |
| 3     | 12-25%   | Thick | Marked coverage |
| 4     | > 25%    | Very thick | Very marked |

**Image Type:** Uses ORIGINAL (non-white-balanced) image to preserve natural scale appearance.

---

### 3. Induration (Thickness) Scoring

#### Method: `_calculate_induration_score()`
**Research Basis:** Surface analysis from "3D Surface Analysis for Psoriasis Assessment" (2017).

**Analysis Methods:**

1. **Brightness Variation Analysis**
   - Standard deviation of pixel intensities
   - Range (max - min) brightness
   - Higher variation = more surface irregularity

2. **Border Elevation Detection** (Lambert's Cosine Law)
   - Elevated edges receive less light
   - Border contrast = inner_brightness - border_brightness
   - Positive value indicates raised plaque

3. **Surface Texture Complexity**
   - Multi-scale texture analysis (fine + coarse)
   - Laplacian edge density
   - Combined texture score

4. **Shadow Density Analysis**
   - Detects darker regions (micro-shadows)
   - Shadow ratio = shadow_pixels / total_pixels
   - Thicker plaques cast more shadows

5. **Color Infiltration Indicator**
   - Combines saturation and brightness
   - Formula: `(saturation/255) * (1 - |brightness-127|/127)`
   - Higher values indicate tissue infiltration

6. **Gradient Magnitude** (Shape-from-Shading)
   - Sobel operators for X and Y gradients
   - Magnitude = √(Gx² + Gy²)
   - Surface slope indicates elevation

**Thickness Indicators (6 total):**
- Brightness variation (std > 18, range > 60)
- Border elevation (contrast > 0.3)
- Texture complexity (density > 14)
- Shadow analysis (ratio > 0.12)
- Infiltration indicator (> 0.35)
- Surface gradient (> 8)

**Scoring Logic:**
- 5+ indicators: +0.6 confidence boost (very thick)
- 4 indicators: +0.4 boost
- 3 indicators: +0.2 boost
- ≤1 indicators: Limited to score 1.6 (likely flat)

**Clinical Constraint:** Induration ≤ Erythema + 1.4 (thick plaques without inflammation are atypical)

**Image Type:** Uses WHITE-BALANCED image for consistent depth/texture analysis.

---

## Image Processing Pipeline

### Main Analysis Flow: `analyze_image()`

```
Input Image
    ↓
[EXIF Orientation Correction]
    ↓
[YOLO Detection - Sniper]
    ↓
    ├─→ [No Detection] → [Center Crop Fallback]
    │                          ↓
    │                    [Metric Calculation]
    │                          ↓
    │                    [Clear or Assumed Lesion]
    │
    └─→ [Detections Found]
             ↓
        [For Each Detection]
             ↓
        [Crop Lesion Region]
             ↓
        [Extract Mask]
             ↓
        [Calculate Metrics]
             ↓
        [Weight by Area]
             ↓
        [Aggregate Results]
             ↓
        [Global Severity Score]
```

### Metric Calculation Flow: `calculate_lesion_metrics()`

```
Input: PIL Image + Optional Mask
    ↓
[AI Baseline Prediction]
    ↓
[Prepare Images]
    ├─→ Original (BGR)
    └─→ White-balanced (BGR)
    ↓
[Prepare Mask]
    ↓
[Extract Image Data]
    ├─→ Grayscale (both versions)
    ├─→ HSV (both versions)
    └─→ BGR channels
    ↓
[Parallel Metric Calculation]
    ├─→ Erythema (white-balanced)
    ├─→ Desquamation (original)
    └─→ Induration (white-balanced)
    ↓
[Score Finalization]
    ├─→ Round to 0.5 precision
    ├─→ Clamp to valid ranges
    └─→ Calculate global score (0-10)
    ↓
Output: {erythema, induration, desquamation, severity_score}
```

### Image Routing Strategy

**Why different images for different metrics?**

| Metric | Image Type | Reason |
|--------|-----------|--------|
| Erythema | White-balanced | Removes lighting bias for accurate color measurement |
| Desquamation | Original | Preserves natural silvery-white scale appearance |
| Induration | White-balanced | Consistent lighting for depth/shadow analysis |

---

## Research Basis

### Validated Methods

1. **Erythema Assessment**
   - "Computerized Plaque Psoriasis Area and Severity Index" (2016)
   - L\*a\*b\* colorimetry standards
   - Erythema Index validation studies

2. **Desquamation Detection**
   - "Automated Assessment of Psoriasis Lesions" (2018)
   - Local Binary Pattern texture analysis
   - Multi-scale edge detection methods

3. **Induration Estimation**
   - "3D Surface Analysis for Psoriasis Assessment" (2017)
   - Photometric stereo principles
   - Shape-from-shading techniques

### Clinical Correlation

**PASI Alignment:**
- Each metric scored 0-4 (standard PASI scale)
- Thresholds derived from clinical studies
- Multi-indicator validation (4-6 independent checks per metric)

**Severity Classification:**
- Mild: Global score < 4.0
- Moderate: Global score 4.0-7.5
- Severe: Global score > 7.5

**Global Score Calculation:**
```
Global Score = ((E + I + D) / 12) × 10
```
Where E = Erythema, I = Induration, D = Desquamation

---

## API Usage

### Initialization
```python
from app.services import ai_engine

# Models are loaded on application startup
ai_engine.load_models()
```

### Analyzing an Image
```python
from PIL import Image

# Load image
image = Image.open("patient_lesion.jpg")

# Analyze
result = ai_engine.analyze_image(image)

# Result structure
{
    "diagnosis": "Moderate",
    "global_score": 5.67,
    "lesions_found": 2,
    "annotated_image_base64": "base64_string...",
    "details": [
        {
            "id": 1,
            "diagnosis": "Moderate",
            "severity_score": 6.25,
            "area_pixels": 12500,
            "erythema": 3,
            "induration": 2,
            "desquamation": 2
        },
        # ... more lesions
    ]
}
```

### Direct Metric Calculation
```python
from PIL import Image
import numpy as np

# For a cropped lesion
crop = Image.open("lesion_crop.jpg")
mask = np.ones((crop.height, crop.width), dtype=np.uint8) * 255

# Calculate metrics
metrics = ai_engine.calculate_lesion_metrics(crop, lesion_mask=mask)

# Result
{
    "erythema": 3,
    "induration": 2,
    "desquamation": 2,
    "severity_score": 5.83
}
```

---

## Performance Considerations

### Computational Complexity

**Detection Phase (YOLO):**
- Time: ~0.1-0.3s per image (GPU) / ~1-2s (CPU)
- Memory: ~500MB GPU / ~2GB CPU

**Classification Phase (EfficientNet):**
- Time: ~0.05s per lesion (GPU) / ~0.2s (CPU)
- Memory: ~300MB GPU / ~1GB CPU

**Computer Vision Analysis:**
- Time: ~0.1-0.2s per lesion (CPU)
- Memory: Minimal (~100MB)

### Optimization Tips

1. **Batch Processing:** Process multiple lesions from same image in parallel
2. **GPU Acceleration:** Use CUDA for 10-20x speedup
3. **Image Resizing:** Downscale large images before processing
4. **Model Caching:** Models loaded once at startup (singleton pattern)

---

## Error Handling

### Fallback Mechanisms

1. **No Detection Fallback:**
   - If YOLO finds no lesions
   - System crops center 75% of image
   - Runs metric calculation
   - Returns result if score > 0.5

2. **Mask Generation:**
   - If no mask provided
   - Creates full region mask (all pixels)
   - Ensures analysis continues

3. **Model Loading:**
   - Checks model file existence
   - Prints clear error messages
   - Graceful degradation if models missing

---

## Configuration

### Model Paths
```python
SNIPER_PATH = "models/sniper.pt"
JUDGE_PATH = "models/judge.pth"
```

### Detection Parameters
```python
conf=0.10  # Low threshold to catch mild cases
```

### Area Threshold
```python
min_area = 20  # Ignore very small detections (noise)
```

### Image Quality
```python
jpeg_quality = 70  # For annotated image output
```

---

## Future Enhancements

### Potential Improvements

1. **Area Assessment Integration**
   - Incorporate BSA (Body Surface Area) calculations
   - Full PASI score computation with area weighting

2. **Temporal Tracking**
   - Compare lesion progression over time
   - Treatment effectiveness monitoring

3. **Additional Metrics**
   - Lichenification detection
   - Border irregularity scoring
   - Color uniformity analysis

4. **Model Updates**
   - Fine-tune on larger datasets
   - Multi-ethnic skin tone training
   - Body region-specific models

---

## References

### Research Papers
1. Computerized Plaque Psoriasis Area and Severity Index (2016)
2. Automated Assessment of Psoriasis Lesions (2018)
3. 3D Surface Analysis for Psoriasis Assessment (2017)

### Clinical Guidelines
- National Psoriasis Foundation PASI Scoring Guidelines
- International Psoriasis Council Assessment Standards

### Technical Standards
- OpenCV 4.x Documentation
- PyTorch Deep Learning Framework
- YOLO Object Detection (Ultralytics)
- EfficientNet Architecture (Google Research)

---

## Contact & Support

For technical questions or issues:
- GitHub: whyismycodered/PsoMetric-Backend
- Repository: PsoMetric-Backend (main branch)

---

**Document Version:** 1.0  
**Last Updated:** December 6, 2025  
**Author:** PsoMetric Development Team
