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

### Core Modules
- **main.py** - Entry point with CLI, runs detection loop or single-image inference
- **model.py** - DetectionModel class wrapping RF-DETR-Seg, supports full-frame and SAHI inference; `offset_detections()` for coordinate mapping
- **config.py** - Centralized configuration (camera, model paths, thresholds, SAHI settings, class names/colors)
- **camera.py** - ImageCapture class for camera/image input (single-frame capture mode)
- **roi.py** - ROIManager for polygon-based ROI with `crop_to_roi()` and `apply_mask_to_cropped()`
- **visualization.py** - Visualizer class for drawing detections and summary panel
- **nms.py** - Non-Maximum Suppression post-processing (class-aware for SAHI)
- **calibrate.py** - Interactive ROI definition tool
- **tracker.py** - ObjectTracker class wrapping BoT-SORT for persistent object ID tracking

### Processing Pipeline
```
Capture → Resize (max 1280px) → Crop to ROI bbox → Polygon Mask → RGB → Inference → [Optional NMS] → [Tracker] → Offset Coords → ROI Filter → [Positioning] → Draw/Save
```

The system crops to ROI bounding box before inference (more efficient than full-image masking), then maps coordinates back to original image space.

**Optional steps:**
- **NMS**: Applied when `--nms` flag is used (threshold=0.45, class-agnostic)
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
- `roi.crop_to_roi(image)` - Crops to ROI bounding box, returns (cropped, x_offset, y_offset)
- `roi.apply_mask_to_cropped()` - Applies polygon mask to cropped image
- `model.offset_detections()` - Maps detection coords back to original image space
- `model.predict_sahi()` - Tiled inference with NMS merging, supports `verbose` and `tiles_dir` params
- `model.predict_hybrid()` - Full + SAHI combined inference with merged results
- `model.merge_detections(d1, d2, strategy, iou_threshold, class_agnostic)` - Merges detection sets with configurable NMS/NMM
- `model.remap_class_ids(detections)` - Remaps model class IDs to display class IDs (applied automatically)
- `nms.class_aware_nms(boxes, scores, class_ids, iou_threshold, class_agnostic)` - NMS with optional class-agnostic mode
- `nms.non_max_merge(boxes, scores, class_ids, iou_threshold)` - NMM: combines overlapping boxes instead of discarding
- `camera.flush_buffer()` - Discards old camera frames to ensure fresh capture
- `Positioning.calculate_positions_from_detections(detections, ...)` - Calculates shelf positions directly from Detections object (used by --web/--mqtt)
- `Positioning.calculate_positions(detection_file, ...)` - Calculates shelf positions from label file (legacy)

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

**How it works:**
- Detections are passed to BoT-SORT after inference
- Tracker assigns unique IDs based on motion and position
- Labels display as `#42 DS_b_330` instead of `DS_b_330: 0.85`

**Tracker Settings (config.py):**
```python
TRACKER_FRAME_RATE = 0.5      # Webcam FPS (adjust to match your camera)
TRACKER_BUFFER = 5            # Frames to keep lost tracks
TRACKER_HIGH_THRESH = 0.5     # Detection threshold for new tracks
TRACKER_MATCH_THRESH = 0.8    # Matching threshold for track association
```

**Key Functions:**
- `tracker.update(detections, frame)` - Updates tracker with new detections, returns detections with `tracker_id` field

**Pipeline with tracking:**
```
Capture → Resize → Crop ROI → RGB → Inference → [Optional NMS] → [TRACKER] → Offset → Filter → Draw
```

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
- Sends HTTP POST to `http://91.107.184.69:3000/data/process`
- Payload format: JSON with store_id, device_id, products, other_products

**--mqtt behavior:**
- Auto-enables `--positioning`
- Publishes to MQTT broker at `89.36.137.77:1883`
- Same payload format as web

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
