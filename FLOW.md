# RF-DETR Detection System - Code Flow

## Quick Overview

```
python main.py
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  main.py (Entry Point - 56 lines)                               │
│  ├── Parse CLI args        → cli.py                             │
│  └── DetectionRunner.run() → runner.py                          │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  runner.py (DetectionRunner Class)                              │
│  ├── setup()               → Initialize all components          │
│  ├── _run_camera_loop()    → Real-time detection                │
│  └── _run_image()          → Single image inference             │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
rf-detr/
├── main.py                 # Entry point (56 lines)
├── cli.py                  # CLI argument parsing
├── runner.py               # DetectionRunner orchestrator
├── pipeline.py             # Inference pipeline
├── model.py                # RF-DETR model
├── camera.py               # Image/camera capture
├── visualization.py        # Drawing & display
├── roi.py                  # Region of interest
├── tracker.py              # Object tracking
├── config.py               # Configuration
├── nms.py                  # NMS algorithms
│
├── utils/
│   ├── io_setup.py         # Directory setup
│   ├── banner.py           # Startup banner
│   ├── preprocessing.py    # Frame preprocessing
│   ├── label_io.py         # Label file I/O
│   ├── network_publish.py  # MQTT/HTTP publishing
│   ├── keypoint_processor.py # Rotation detection
│   └── id_helper.py        # Bbox ID generation
│
└── Positioning/
    └── positioning_module.py # Shelf position calculation
```

---

## Detailed Flow

### 1. Entry Point: `main.py`

```python
# main.py
def main():
    args = create_argument_parser().parse_args()  # → cli.py
    runner = DetectionRunner(args)                 # → runner.py
    runner.run()
```

**Calls:**
- `cli.py` → Parse command-line arguments
- `runner.py` → Create and run DetectionRunner

---

### 2. Setup Phase: `runner.setup()`

```
DetectionRunner.setup()
│
├── Step 1: Setup directories
│   └── utils/io_setup.py → setup_directories()
│       Creates: labels/, images/, positions/, mqtt_logs/, verbose_tiles/
│
├── Step 2: Create inference pipeline
│   └── pipeline.py → create_pipeline()
│       ├── model.py → DetectionModel (loads checkpoint)
│       ├── tracker.py → ObjectTracker (if --track)
│       └── roi.py → ROIManager (loads roi_config.json)
│
├── Step 3: Initialize keypoint processor (if positioning enabled)
│   └── utils/keypoint_processor.py → KeypointProcessor
│
├── Step 4: Check ROI configuration
│   └── roi.py → pipeline.roi.is_complete()
│
├── Step 5: Print startup banner
│   └── utils/banner.py → print_banner()
│
├── Step 6: Open camera (camera mode)
│   └── camera.py → ImageCapture()
│
└── Step 7: Create visualizer (GUI mode)
    └── visualization.py → Visualizer()
```

---

### 3. Camera Loop: `runner._run_camera_loop()`

```
while True:
    │
    ├── Step 1: Capture frame
    │   └── camera.py → camera.capture_single()
    │
    ├── Step 2: Run inference pipeline
    │   └── pipeline.py → pipeline.run(frame)
    │       ├── utils/preprocessing.py → preprocess_frame()
    │       ├── model.py → predict() / predict_sahi() / predict_hybrid()
    │       ├── tracker.py → tracker.update() [optional]
    │       └── Returns: InferenceResult
    │
    ├── Step 3: Save labels [optional]
    │   └── utils/label_io.py → save_detections_to_txt()
    │
    ├── Step 4: Process keypoints
    │   └── utils/keypoint_processor.py → process()
    │
    ├── Step 5: Calculate positions
    │   └── Positioning/ → calculate_positions_from_detections()
    │
    ├── Step 6: Publish results
    │   └── utils/network_publish.py → publish_mqtt() / publish_http()
    │
    ├── Step 7: Calculate FPS
    │
    ├── Step 8: Display results
    │   └── visualization.py → draw_detections(), draw_summary(), show()
    │
    └── Step 9: Handle keyboard input
        └── visualization.py → handle_input()
```

---

### 4. Image Mode: `runner._run_image()`

```
_run_image()
│
├── Step 1: Load image
│   └── camera.py → load_image()
│
├── Step 2: Run inference pipeline
│   └── pipeline.py → pipeline.run(frame)
│
├── Step 3: Save labels [optional]
│   └── utils/label_io.py → save_detections_to_txt()
│
├── Step 4: Process keypoints
│   └── utils/keypoint_processor.py → process()
│
├── Step 5: Calculate positions
│   └── Positioning/ → calculate_positions_from_detections()
│
├── Step 6: Publish results
│   └── utils/network_publish.py → publish_mqtt() / publish_http()
│
├── Step 7: Visualize & save
│   └── visualization.py → draw_detections(), draw_summary()
│   └── cv2.imwrite() → Save result image
│
└── Step 8: Display result
    └── cv2.imshow()
```

---

### 5. Cleanup: `runner.cleanup()`

```
cleanup()
├── camera.py → camera.release()
├── visualization.py → visualizer.destroy()
└── utils/keypoint_processor.py → keypoint_processor.cleanup()
```

---

## Pipeline Internals: `pipeline.run()`

```
pipeline.run(frame)
│
├── 1. Preprocess
│   └── utils/preprocessing.py → preprocess_frame()
│       ├── Resize to max dimension
│       ├── Crop to ROI (if configured)
│       ├── Apply polygon mask
│       └── Convert BGR → RGB
│
├── 2. Inference (based on mode)
│   └── model.py
│       ├── full:   predict()        - Single pass
│       ├── sahi:   predict_sahi()   - 3×2 tiled inference
│       └── hybrid: predict_hybrid() - Full + SAHI combined
│
├── 3. Post-process
│   ├── nms.py → Apply NMS (if --nms)
│   ├── tracker.py → Update tracker (if --track)
│   ├── model.py → offset_detections() - Map coords to original
│   ├── roi.py → filter_detections() - Filter to ROI
│   └── utils/id_helper.py → add_bbox_ids()
│
└── Return: InferenceResult
    ├── detections (final)
    ├── cropped_detections (for display)
    ├── frame_display
    ├── frame_resized
    ├── inference_time
    └── timings
```

---

## Component Responsibilities

| File | Purpose |
|------|---------|
| `main.py` | Entry point, validate checkpoint, create runner |
| `cli.py` | Define and parse CLI arguments |
| `runner.py` | Orchestrate setup, detection loop, cleanup |
| `pipeline.py` | Unified inference pipeline |
| `model.py` | RF-DETR model inference |
| `camera.py` | Camera capture, image loading |
| `visualization.py` | Draw detections, summary panel |
| `roi.py` | ROI management, crop/mask |
| `tracker.py` | BoT-SORT object tracking |
| `config.py` | All configuration constants |
| `utils/io_setup.py` | Create output directories |
| `utils/banner.py` | Print startup banner |
| `utils/preprocessing.py` | Frame resize, crop, mask |
| `utils/label_io.py` | Save/load detection labels |
| `utils/network_publish.py` | MQTT/HTTP publishing |
| `utils/keypoint_processor.py` | Rotation angle detection |
| `Positioning/` | Calculate shelf positions |

---

## Call Graph Summary

```
main.py
└── runner.py (DetectionRunner)
    ├── utils/io_setup.py
    ├── utils/banner.py
    ├── pipeline.py
    │   ├── utils/preprocessing.py
    │   ├── model.py
    │   ├── tracker.py
    │   ├── roi.py
    │   └── nms.py
    ├── camera.py
    ├── visualization.py
    ├── utils/label_io.py
    ├── utils/keypoint_processor.py
    ├── utils/network_publish.py
    └── Positioning/
```

---

## Usage Examples

```bash
# Camera mode (default)
python main.py

# Image mode
python main.py --image photo.jpg

# With tracking
python main.py --track

# Headless (no GUI)
python main.py --cli

# Publish via MQTT
python main.py --mqtt

# Full + SAHI combined
python main.py --mode hybrid
```
