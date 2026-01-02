# RF-DETR Detection System

Real-time object detection system for retail shelf monitoring using RF-DETR-Seg model. Detects bottles and beverage containers with instance segmentation.

## Setup

```bash
pip install -r requirements.txt
```

## Detection (main.py)

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image` | string | None | Path to image file for single image inference |
| `--output` | string | None | Output path for annotated image result |
| `--mode` | choice | `sahi` | Inference mode: `full` (single pass) or `sahi` (2x2 tiled) |
| `--checkpoint` | string | `runs/rfdetr_seg_training/checkpoint_best_ema.pth` | Path to model checkpoint |
| `--conf` | float | `0.5` | Confidence threshold (0.0-1.0) |
| `--openvino` | flag | False | Enable OpenVINO optimization for Intel CPU |
| `--camera` | int | `1` | Camera device index |
| `--save-labels` | flag | False | Save detections to text files |
| `--labels-dir` | string | `labels` | Directory for label output files |

### Examples

#### Camera Mode (Real-time Detection)
```bash
# Basic - uses default camera and SAHI mode
python main.py

# Use full frame mode (faster, less accurate for small objects)
python main.py --mode full

# Different camera
python main.py --camera 0

# Lower confidence threshold
python main.py --conf 0.3

# With OpenVINO CPU optimization
python main.py --openvino

# Save detection labels
python main.py --save-labels

# Save labels to custom directory
python main.py --save-labels --labels-dir ./output/labels
```

#### Image Mode (Single Image)
```bash
# Process single image
python main.py --image photo.jpg

# Save annotated result
python main.py --image photo.jpg --output result.jpg

# With custom confidence
python main.py --image photo.jpg --conf 0.4 --output result.jpg

# Save detection labels
python main.py --image photo.jpg --save-labels

# Full example with all options
python main.py --image photo.jpg --output result.jpg --mode sahi --conf 0.5 --save-labels --labels-dir ./labels
```

### Runtime Controls (Camera Mode)
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |
| `M` | Toggle mask visibility |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

### Label Output Format

When using `--save-labels`, detection files are saved as:
```
labels/
  frame_20231215_143022_000000.txt
  frame_20231215_143022_000001.txt
images/
  frame_20231215_143022_000000.jpg
  frame_20231215_143022_000001.jpg
```

Each line in the `.txt` file:
```
class_id poly_x1 poly_y1 poly_x2 poly_y2 ... | bbox_x1 bbox_y1 bbox_x2 bbox_y2
```
- All coordinates are normalized (0.0-1.0) relative to full image (1280x960)
- `|` separates polygon points from bounding box
- Bounding box format: x1 y1 x2 y2 (top-left and bottom-right corners)

---

## ROI Calibration (calibrate.py)

Define a region of interest to filter detections.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--camera` | int | `1` | Camera device index |
| `--image` | string | None | Image file for calibration |
| `--config` | string | `roi_config.json` | ROI configuration file path |
| `--show` | flag | False | Display current ROI and exit |

### Examples

```bash
# Calibrate using camera
python calibrate.py

# Calibrate using specific camera
python calibrate.py --camera 0

# Calibrate from image file
python calibrate.py --image reference.jpg

# Show current ROI configuration
python calibrate.py --show

# Use custom config file
python calibrate.py --config my_roi.json
```

### Calibration Controls
| Key/Action | Description |
|------------|-------------|
| Click | Add corner point (4 points required) |
| `R` | Reset all points |
| `S` | Save ROI configuration |
| `Q` | Cancel without saving |

---

## Detection Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | objects | Generic objects |
| 1 | DS_b_330 | Delster bottle 330ml |
| 2 | DS_b_330_4pa | Delster bottle 330ml 4-pack |
| 3 | DS_c_500 | Delster can 500ml |
| 4 | HN_b_330 | Heineken bottle 330ml |
| 5 | HN_b_330_6pa | Heineken bottle 330ml 6-pack |
| 6 | HN_c_330 | Heineken can 330ml |
| 7 | HN_c_500 | Heineken can 500ml |
| 8 | other | Other objects |
| 9 | other-drink | Other drinks |

---

## Inference Modes

### SAHI Mode (default)
- Splits image into 2x2 tiles with 20% overlap
- Better accuracy for small objects
- Slower due to multiple inference passes

### Full Mode
- Single inference pass on entire image
- Faster processing
- May miss small objects
