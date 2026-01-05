"""
Configuration Module
====================
All configuration constants for the RF-DETR detection system.
"""

# ============================================================================
# CAMERA SETTINGS
# ============================================================================

CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960

# ============================================================================
# MODEL SETTINGS
# ============================================================================

DEFAULT_CHECKPOINT = "models/model_2jan_best_ema.pth"
DEFAULT_CONFIDENCE = 0.4

# ============================================================================
# SAHI SETTINGS (Tiling for small object detection)
# ============================================================================

SAHI_OVERLAP = 0.3       # 30% overlap for better stitching
SAHI_TILES_X = 2         # 3x2 grid (width > height after ROI crop)
SAHI_TILES_Y = 2
MAX_IMAGE_SIZE = 1280    # Max dimension for inference resize
VERBOSE_TILES_DIR = "verbose_tiles"  # Directory for saving tile images

# ============================================================================
# NMS SETTINGS
# ============================================================================

NMS_IOU_THRESHOLD = 0.2  # IoU threshold for SAHI tile merging (LOWER = more aggressive)

# ============================================================================
# MERGE SETTINGS (for hybrid mode: full + SAHI)
# ============================================================================

MERGE_IOU_THRESHOLD = 0.45         # IoU threshold for merging (LOWER = fewer duplicates)
MERGE_CLASS_AGNOSTIC = True       # Ignore class labels when merging (recommended)
MERGE_STRATEGY = "nms"            # "nms" or "nmm" (non-max merging)

# ============================================================================
# ROI SETTINGS
# ============================================================================

ROI_CONFIG_FILE = "roi_config.json"

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

WINDOW_NAME = "RF-DETR Detection"
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Summary panel width
PANEL_WIDTH = 250

# ============================================================================
# CLASS CONFIGURATION (Original model outputs)
# ============================================================================

CLASS_NAMES_ORIGINAL = [
    'objects',        # 0 - generic objects
    'DS_b_330',       # 1 - Delster bottle 330ml
    'DS_b_330_4pa',   # 2 - Delster bottle 330ml 4-pack
    'DS_c_500',       # 3 - Delster can 500ml
    'HN_b_330',       # 4 - Heineken bottle 330ml
    'HN_b_330_6pa',   # 5 - Heineken bottle 330ml 6-pack
    'HN_c_330',       # 6 - Heineken can 330ml
    'HN_c_500',       # 7 - Heineken can 500ml
    'other',          # 8 - other objects
    'other-drink'     # 9 - other drinks
]

# ============================================================================
# CLASS REMAPPING (Model output -> New IDs)
# ============================================================================

# Map model output class IDs to new class IDs
CLASS_REMAP = {
    0: -1,   # objects -> Other
    1: 0,    # DS_b_330 -> 0
    2: 1,    # DS_b_330_4pa -> 1
    3: 2,    # DS_c_500 -> 2
    4: 3,    # HN_b_330 -> 3
    5: 4,    # HN_b_330_6pa -> 4
    6: 6,    # HN_c_330 -> HN_c_500 (merged into 6)
    7: 6,    # HN_c_500 -> 6
    8: -1,   # other -> Other
    9: -1,   # other-drink -> Other
}

# New class names (new_id -> name)
CLASS_NAMES = {
    0: 'DS_b_330',
    1: 'DS_b_330_4pa',
    2: 'DS_c_500',
    3: 'HN_b_330',
    4: 'HN_b_330_6pa',
    # 5 is intentionally skipped
    6: 'HN_c_500',
    -1: 'Other',
}

# New class colors (new_id -> BGR color)
CLASS_COLORS = {
    0: (0, 255, 0),      # DS_b_330 - green
    1: (0, 255, 255),    # DS_b_330_4pa - yellow
    2: (0, 165, 255),    # DS_c_500 - orange
    3: (255, 0, 0),      # HN_b_330 - blue
    4: (255, 0, 255),    # HN_b_330_6pa - magenta
    6: (0, 0, 255),      # HN_c_500 - red
    -1: (128, 128, 128), # Other - gray
}

# Whether to filter out "Other" class (-1) from display
FILTER_OTHER_CLASS = False

# ============================================================================
# TRACKER SETTINGS (BoT-SORT object tracking)
# ============================================================================

TRACKER_FRAME_RATE = 0.5      # Webcam FPS (adjust to match your camera)
TRACKER_BUFFER = 5            # Frames to keep lost tracks (~10 sec at 0.5 FPS)
TRACKER_HIGH_THRESH = 0.5     # Detection threshold for new tracks
TRACKER_MATCH_THRESH = 0.8    # Matching threshold for track association

# ============================================================================
# ROI VISUALIZATION COLORS
# ============================================================================

ROI_COLOR_POINT = (0, 255, 0)      # Green
ROI_COLOR_LINE = (0, 255, 255)     # Yellow
ROI_COLOR_FILL = (0, 255, 0)       # Green fill
ROI_COLOR_TEXT = (255, 255, 255)   # White

# ============================================================================
# NETWORK SETTINGS - MQTT
# ============================================================================

MQTT_BROKER_HOST = "89.36.137.77"
MQTT_BROKER_PORT = 1883
MQTT_DEFAULT_TOPIC = "test/topic"

# ============================================================================
# NETWORK SETTINGS - HTTP
# ============================================================================

HTTP_ENDPOINT = "http://91.107.184.69:3000/data/process"
HTTP_TIMEOUT = 5

# ============================================================================
# PAYLOAD DEFAULTS
# ============================================================================

PAYLOAD_STORE_ID = 1
PAYLOAD_DEVICE_ID = 1
PAYLOAD_LEVEL = 4

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

IMAGES_DIR = "images"
LABELS_DIR = "labels"
MQTT_LOGS_DIR = "mqtt_logs"
POSITIONS_DIR = "positions"

# ============================================================================
# POST-PROCESSING SETTINGS
# ============================================================================

POST_NMS_THRESHOLD = 0.45
POST_NMS_CLASS_AGNOSTIC = True

# ============================================================================
# UI SETTINGS
# ============================================================================

CONFIDENCE_INCREMENT = 0.05
CONFIDENCE_MIN = 0.05
CONFIDENCE_MAX = 0.95
DISPLAY_KEY_WAIT_MS = 1

# ============================================================================
# KEYPOINT DETECTION SETTINGS (for big object angle calculation)
# ============================================================================

KEYPOINT_MODEL_PATH = "checkpoints/best.pt"
KEYPOINT_OPENVINO_PATH = "checkpoints/best_openvino_model"  # Auto-generated OpenVINO model
KEYPOINT_CROP_MARGIN = 20              # Pixels of margin around bbox when cropping
KEYPOINT_TEMP_DIR = "temp_crops"       # Directory for temporary crop files
KEYPOINT_BIG_OBJECT_CLASSES = {}   # DS_b_330_4pa, HN_b_330_6pa
KEYPOINT_MIN_CONFIDENCE = 0.5          # Minimum keypoint confidence

# ============================================================================
# OPENVINO SETTINGS
# ============================================================================

# OpenVINO is enabled by default for better performance
# If OpenVINO model doesn't exist, it will be auto-exported from PyTorch model
OPENVINO_ENABLED = True                # Enable OpenVINO optimization by default
OPENVINO_DEVICE = "CPU"                # Device for OpenVINO inference (CPU, GPU, AUTO)
