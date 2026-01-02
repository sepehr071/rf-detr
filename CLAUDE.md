# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RF-DETR is a real-time object detection system for retail shelf monitoring, specializing in bottle and beverage container detection using the RF-DETR-Seg model (RFDETRSegPreview). Python 3.13+, Windows-based.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Detection
```bash
# Real-time camera detection (default: SAHI mode, camera 1)
python main.py

# Full frame mode (faster, less accurate for small objects)
python main.py --mode full

# Hybrid mode (full + SAHI combined, best accuracy, ~2x slower)
python main.py --mode hybrid

# Process single image
python main.py --image photo.jpg --output result.jpg

# With OpenVINO CPU optimization
python main.py --openvino

# Save detection labels
python main.py --save-labels --labels-dir ./labels

# Custom confidence threshold
python main.py --conf 0.45

# Save SAHI tile images for debugging
python main.py --verbose

# Timing breakdown per frame
python main.py --log

# Enable object tracking with persistent IDs
python main.py --track

# Apply additional NMS to remove duplicate detections
python main.py --nms

# Send positioning data via HTTP POST (auto-enables positioning)
python main.py --web

# Publish positioning data via MQTT (auto-enables positioning)
python main.py --mqtt
```

### ROI Calibration
```bash
python calibrate.py                  # Interactive camera calibration
python calibrate.py --image img.jpg  # Calibrate from image
python calibrate.py --show           # Display current ROI
```

### No automated tests exist - test manually via CLI

## Architecture

### Modular Design (Thin Orchestrator Pattern)

The codebase follows a **thin orchestrator pattern** where `main.py` coordinates between specialized modules without containing business logic.

```
main.py (Orchestrator, ~500 lines)
│
├── CLI parsing & setup
├── create_pipeline() → InferencePipeline
│
└── pipeline.run(frame) → InferenceResult
        │
        ├── utils/preprocessing.py   → preprocess_frame()
        ├── model.py                 → predict*()
        ├── tracker.py               → update() [optional]
        ├── model.py                 → offset_detections()
        ├── roi.py                   → filter_detections()
        └── utils/id_helper.py       → add_bbox_ids()
```

### Core Modules

**Orchestration Layer:**
- **main.py** - Thin orchestrator: CLI parsing, module initialization, loop control (no business logic)
- **pipeline.py** - `InferencePipeline` class: unified inference flow for camera and image modes

**Inference Layer:**
- **model.py** - `DetectionModel` class: RF-DETR-Seg inference (full/SAHI/hybrid modes)
- **tracker.py** - `ObjectTracker` class: BoT-SORT for persistent object IDs

**Preprocessing Layer:**
- **roi.py** - `ROIManager`: polygon-based ROI with crop/mask operations
- **utils/preprocessing.py** - `preprocess_frame()`: resize → crop → mask → RGB pipeline

**I/O Layer:**
- **camera.py** - `ImageCapture` class: camera/image input
- **utils/label_io.py** - `save_detections_to_txt()`: label file serialization
- **utils/network_publish.py** - `publish_mqtt()`, `publish_http()`: network publishing

**Configuration & Visualization:**
- **config.py** - All configuration constants (camera, model, network, thresholds, classes)
- **visualization.py** - `Visualizer` class: drawing detections and summary panel

**Utilities:**
- **nms.py** - NMS algorithms (class-aware, non-max merging)
- **calibrate.py** - Interactive ROI definition tool
- **utils/id_helper.py** - Bounding box ID generation

### Processing Pipeline
```
Capture → pipeline.run(frame) → InferenceResult
              │
              ├── Preprocess (resize → crop ROI → mask → RGB)
              ├── Inference (full/sahi/hybrid)
              ├── NMS [optional]
              ├── Tracker [optional]
              ├── Offset coordinates
              ├── Filter by ROI
              └── Add bbox IDs
```

**Optional steps:**
- **NMS**: Applied when `--nms` flag is used (threshold from config)
- **Tracker**: Applied when `--track` flag is used (BoT-SORT)
- **Positioning**: Automatically enabled when `--web` or `--mqtt` is used

### Inference Modes
- **full**: Single-pass inference on cropped ROI, fastest
- **sahi**: 3×2 tiled inference with 30% overlap on cropped ROI, better for small objects (default)
- **hybrid**: Runs BOTH full + SAHI inference, merges with NMS. Best accuracy (+6-14% AP), ~2x slower

### ROI Preview
The preview window shows the **cropped ROI frame** (what the model actually sees), not the full camera frame. This helps verify that the ROI is correctly configured and objects are visible to the model.

### SAHI Configuration (config.py)
```python
SAHI_TILES_X = 3         # 3 columns
SAHI_TILES_Y = 2         # 2 rows
SAHI_OVERLAP = 0.3       # 30% overlap for better stitching
NMS_IOU_THRESHOLD = 0.3  # Post-SAHI duplicate removal (LOWER = more aggressive)
```

### Merge Settings (config.py) - For Hybrid Mode
```python
MERGE_IOU_THRESHOLD = 0.3      # IoU threshold for merging full+SAHI (LOWER = fewer duplicates)
MERGE_CLASS_AGNOSTIC = True    # Ignore class labels when merging (recommended)
MERGE_STRATEGY = "nms"         # "nms" or "nmm" (non-max merging)
FILTER_OTHER_CLASS = False     # Whether to hide "Other" class detections
```

**Important**: NMS IoU threshold is counter-intuitive:
- **LOWER threshold (0.2-0.3)** = More aggressive suppression = Fewer duplicates
- **HIGHER threshold (0.5-0.7)** = Less aggressive = More duplicates kept

### Network Configuration (config.py)
```python
# MQTT
MQTT_BROKER_HOST = "89.36.137.77"
MQTT_BROKER_PORT = 1883
MQTT_DEFAULT_TOPIC = "test/topic"

# HTTP
HTTP_ENDPOINT = "http://91.107.184.69:3000/data/process"
HTTP_TIMEOUT = 5

# Payload defaults
PAYLOAD_STORE_ID = 1
PAYLOAD_DEVICE_ID = 1
PAYLOAD_LEVEL = 4

# Post-processing
POST_NMS_THRESHOLD = 0.45
POST_NMS_CLASS_AGNOSTIC = True
```

### Label Output Format
When using `--save-labels`, each line in the `.txt` file:
```
class_id poly_x1 poly_y1 poly_x2 poly_y2 ... | bbox_x1 bbox_y1 bbox_x2 bbox_y2
```
- All coordinates normalized (0-1) relative to full image
- `|` separates polygon from bounding box

### Detection Classes (Remapped at Inference)
Model outputs 10 classes, remapped to 7 display classes:

| New ID | Display Name | Model Output IDs |
|--------|--------------|------------------|
| 0 | DS_b_330 | 1 |
| 1 | DS_b_330_4pa | 2 |
| 2 | DS_c_500 | 3 |
| 3 | HN_b_330 | 4 |
| 4 | HN_b_330_6pa | 5 |
| 5 | (skipped) | - |
| 6 | HN_c_500 | 6, 7 (HN_c_330 merged) |
| -1 | Other | 0, 8, 9 (objects, other, other-drink) |

Class remapping is configured in `config.py` via `CLASS_REMAP` dict and applied automatically after inference by `remap_class_ids()`.

### Key Files
- Model checkpoint: `runs/rfdetr_seg_training/checkpoint_best_ema.pth`
- ROI config: `roi_config.json` (generated by calibrate.py)

### Key Functions

**Pipeline (pipeline.py):**
- `create_pipeline(checkpoint, openvino, config)` - Factory to create configured InferencePipeline
- `pipeline.run(frame)` - Run full inference pipeline, returns `InferenceResult`
- `pipeline.update_confidence(conf)` - Update confidence threshold at runtime

**Preprocessing (utils/preprocessing.py):**
- `preprocess_frame(frame, roi, max_size)` - Full preprocessing pipeline, returns `PreprocessedFrame`

**Label I/O (utils/label_io.py):**
- `save_detections_to_txt(detections, path, h, w)` - Save detections to normalized text format
- `load_detections_from_txt(path, h, w)` - Load detections from text file

**Network (utils/network_publish.py):**
- `configure_mqtt(host, port, topic)` - Configure MQTT connection
- `publish_mqtt(positions, save_log, log_dir)` - Publish positions via MQTT
- `publish_http(positions, endpoint, timeout)` - Publish positions via HTTP POST
- `format_payload(positions, store_id, device_id, level)` - Format positions to JSON payload

**ROI (roi.py):**
- `roi.crop_to_roi(image)` - Crops to ROI bounding box, returns (cropped, x_offset, y_offset)
- `roi.apply_mask_to_cropped()` - Applies polygon mask to cropped image
- `roi.filter_detections(detections)` - Filter detections to ROI polygon

**Model (model.py):**
- `model.predict(frame, conf)` - Full-frame inference
- `model.predict_sahi(frame, conf, verbose, tiles_dir)` - Tiled inference with NMS merging
- `model.predict_hybrid(frame, conf, verbose, tiles_dir)` - Full + SAHI combined inference
- `offset_detections(detections, x, y, h, w)` - Maps detection coords back to original space
- `remap_class_ids(detections)` - Remaps model class IDs to display class IDs

**Positioning:**
- `Positioning.calculate_positions_from_detections(detections, ...)` - Calculates shelf positions from Detections
- `Positioning.calculate_positions(detection_file, ...)` - Calculates shelf positions from label file (legacy)

### Key Data Classes

**PipelineConfig** (pipeline.py):
```python
@dataclass
class PipelineConfig:
    mode: str = "sahi"           # full, sahi, hybrid
    confidence: float = 0.5
    nms_enabled: bool = False
    track_enabled: bool = False
    verbose: bool = False
    tiles_dir: Optional[str] = None
```

**InferenceResult** (pipeline.py):
```python
@dataclass
class InferenceResult:
    detections: Detections       # Final detections (original coords)
    cropped_detections: Detections  # For visualization (cropped coords)
    frame_display: np.ndarray    # BGR frame for visualization
    frame_resized: np.ndarray    # Full resized frame (for saving)
    original_height: int
    original_width: int
    inference_time: float
    timings: Dict[str, float]
```

**PreprocessedFrame** (utils/preprocessing.py):
```python
@dataclass
class PreprocessedFrame:
    frame_rgb: np.ndarray        # RGB frame for model
    frame_masked: np.ndarray     # BGR masked frame (for tracker)
    frame_resized: np.ndarray    # Full resized frame
    original_height: int
    original_width: int
    x_offset: int                # ROI crop offset
    y_offset: int
    scale: float
    scaled_roi: Optional[ROIManager]
    timings: Dict[str, float]
```

### Verbose Mode (--verbose)
Saves SAHI tile images to `verbose_tiles/` directory for debugging:
```
verbose_tiles/
├── 20241230_143052_tile_0_0.jpg
├── 20241230_143052_tile_0_1.jpg
├── 20241230_143052_tile_0_2.jpg
├── 20241230_143052_tile_1_0.jpg
├── 20241230_143052_tile_1_1.jpg
└── 20241230_143052_tile_1_2.jpg
```
Naming format: `{timestamp}_tile_{row}_{col}.jpg`

### Dependencies
- `rfdetr` - RF-DETR-Seg model (RFDETRSegPreview class)
- `supervision` - Detection result handling
- `opencv-python` - Image/camera operations
- `openvino` - Optional Intel CPU optimization
- `boxmot` - BoT-SORT object tracking (optional, for --track)

### Model Info
- Model: RFDETRSegPreview (instance segmentation)
- Default resolution: 432×432 (internal)
- Backbone: DINOv2 (modified patch size 12)
- Only `threshold` parameter exposed in predict() - no iou_threshold

### Hybrid Mode Theory
Hybrid inference combines full-image and sliced (SAHI) inference to get the best of both:
- **Full-image inference**: Better for large objects, preserves global context
- **SAHI sliced inference**: Better for small objects, higher resolution per object
- **Why combine**: Slicing can miss large objects spanning multiple tiles and lose global context, while full-image can miss small objects

Research shows +6-14% AP improvement when combining both approaches.

### Object Tracking (--track)
When enabled, BoT-SORT tracker assigns persistent IDs to detected objects across frames, stabilizing detections even when objects briefly disappear.

**Basic Usage:**
```bash
python main.py --track                    # Enable tracking
python main.py --track --mode hybrid      # With hybrid mode
python main.py --track --nms              # With additional NMS
```

**How BoT-SORT Works:**

The tracker uses **Bag of Tricks SORT** algorithm which combines:
- **IoU Matching**: Compares detection positions using Intersection over Union
- **Motion Prediction**: Predicts where objects will move based on velocity
- **Track Management**: Maintains object identities across frames

**Track Lifecycle:**
1. **Initialization**: Detection with confidence ≥ `TRACK_HIGH_THRESH` (0.5) creates new track
2. **Active**: Track matched to detection in current frame
3. **Lost**: Not matched, but kept alive for `TRACKER_BUFFER` frames (5 frames = ~10 sec at 0.5 FPS)
4. **Deleted**: Lost for more than buffer frames

**Configuration Settings (config.py):**
```python
TRACKER_FRAME_RATE = 0.5      # Expected FPS - MUST match your camera!
TRACKER_BUFFER = 5            # Keep lost tracks for 5 frames (~10 sec at 0.5 FPS)
TRACKER_HIGH_THRESH = 0.5     # Min confidence to start new tracks
TRACKER_MATCH_THRESH = 0.8    # IoU similarity for matching (0-1)
```

**Setting Details:**
- **FRAME_RATE**: Used for motion prediction. If too high, predicts slow motion (bad). If too low, predicts fast motion (worse). **Adjust to match actual camera FPS!**
- **BUFFER**: How long to remember lost objects. Higher = more stable IDs, but slower to forget occlusions
- **HIGH_THRESH**: Only confident detections create new tracks. Lower = more IDs (noisier). Higher = fewer IDs (miss objects)
- **MATCH_THRESH**: How similar detection must be to existing track. Lower = stricter (fewer ID switches). Higher = looser (more ID switches)

**When to Use Tracking:**

✅ **Use when:**
- Processing video/camera feeds (objects move between frames)
- Need stable object IDs across frames
- Filtering noisy single-frame false positives
- Counting unique objects
- Analyzing object trajectories

❌ **Don't use when:**
- Processing single images (no temporal context)
- Object IDs don't matter, only total counts
- Need maximum performance (tracking adds ~50ms overhead per frame)

**Pipeline Position:**
```
Inference → [Optional NMS] → TRACKER → Offset Coords → Filter → Draw
```

**Important:** Tracker operates in **cropped ROI coordinate space** (before offsetting). This means:
- Tracker receives `frame_masked` (BGR, cropped to ROI)
- Detections are in cropped coordinates
- After tracking, coordinates are offset back to full image
- **Tracker IDs persist through coordinate transformation**

**Display Format:**
- **Tracked objects**: `#42 DS_b_330` (ID + class name, no confidence)
- **Untracked objects**: `DS_b_330: 0.85` (class name + confidence)
- Untracked = new detection or confidence < TRACK_HIGH_THRESH

**Key Functions:**
- `tracker.update(detections, frame)` - Updates tracker, returns detections with `tracker_id` field
  - `tracker_id >= 0`: Persistent ID
  - `tracker_id == -1`: Untracked

**Troubleshooting:**

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| Too many ID switches | MATCH_THRESH too high or FRAME_RATE mismatch | Lower MATCH_THRESH to 0.7 or adjust FRAME_RATE |
| Missing IDs for valid objects | TRACK_HIGH_THRESH too high | Lower to 0.4 or use with --conf 0.4 |
| IDs persist after object leaves | TRACKER_BUFFER too high | Reduce to 2-3 frames |
| Duplicate IDs for same object | FRAME_RATE wrong | Measure actual FPS and update config |

### Additional NMS (--nms)
Applies post-inference NMS using supervision's `with_nms()` method to remove duplicate detections.

**Settings:**
```python
threshold = 0.45        # IoU threshold for duplicate removal
class_agnostic = True   # Compare all boxes regardless of class
```

**When to use:**
- Seeing duplicate detections in any mode
- Want consistent NMS across all inference modes
- Need additional filtering on top of SAHI/Hybrid internal NMS

**Note:** This is applied AFTER mode-specific NMS (SAHI/Hybrid), acting as an additional filter.

### Web & MQTT Integration
Both `--web` and `--mqtt` flags automatically enable positioning calculation.

**--web behavior:**
- Auto-enables `--positioning`
- Calculates shelf positions directly from detections (no file I/O)
- Sends HTTP POST to endpoint configured in `config.py` (`HTTP_ENDPOINT`)
- Payload format: JSON with store_id, device_id, products, other_products

**--mqtt behavior:**
- Auto-enables `--positioning`
- Publishes to MQTT broker configured in `config.py` (`MQTT_BROKER_HOST`, `MQTT_BROKER_PORT`)
- Saves payload logs to `mqtt_logs/` directory
- Same payload format as web

**Network publishing functions (utils/network_publish.py):**
```python
from utils.network_publish import publish_mqtt, publish_http, configure_mqtt

# Configure and publish
configure_mqtt()  # Uses config.py defaults
result = publish_mqtt(positions)  # Returns PublishResult
result = publish_http(positions)  # Returns PublishResult
```

**--save-labels:**
- Now optional (for debugging/logging only)
- Not required for `--web` or `--mqtt` to work
- Saves detection files to `labels/` and images to `images/`

### References & Sources
- [SAHI GitHub Repository](https://github.com/obss/sahi) - Slicing Aided Hyper Inference library
- [SAHI Paper (arXiv:2202.06934)](https://arxiv.org/abs/2202.06934) - Original SAHI research paper
- [Ultralytics SAHI Guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/) - SAHI with YOLO integration
- [SAHI Explained (Encord)](https://encord.com/blog/slicing-aided-hyper-inference-explained/) - Detailed SAHI explanation
- [LearnOpenCV SAHI Tutorial](https://learnopencv.com/slicing-aided-hyper-inference/) - Practical implementation guide
- [BoxMOT GitHub](https://github.com/mikel-brostrom/boxmot) - Pluggable SOTA multi-object tracking (BoT-SORT, ByteTrack)
