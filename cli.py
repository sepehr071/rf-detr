"""
CLI argument parsing for RF-DETR detection system.
"""

import argparse

from config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE,
    CAMERA_INDEX,
    LABELS_DIR,
    POSITIONS_DIR,
)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="RF-DETR Detection System")

    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to image file for single image inference"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for image inference result"
    )
    parser.add_argument(
        "--mode", choices=["full", "sahi", "hybrid"], default="sahi",
        help="Inference mode: 'full' for full frame, 'sahi' for tiled, 'hybrid' for full+sahi combined"
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONFIDENCE,
        help="Confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--no-openvino", action="store_true",
        help="Disable OpenVINO optimization (OpenVINO is enabled by default)"
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--save-labels", action="store_true",
        help="Save detections to text file (class_id + polygon per line)"
    )
    parser.add_argument(
        "--labels-dir", type=str, default=LABELS_DIR,
        help="Directory to save label files"
    )
    parser.add_argument(
        "--log", action="store_true",
        help="Print timing breakdown for each pipeline stage per frame"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Save SAHI tile images to verbose_tiles/ directory for debugging"
    )
    parser.add_argument(
        "--positioning", action="store_true", default=True,
        help="Calculate shelf positions after detection (enabled by default)"
    )
    parser.add_argument(
        "--positioning-dir", type=str, default=POSITIONS_DIR,
        help="Directory to save positioning JSON files"
    )
    parser.add_argument(
        "--mqtt", action="store_true",
        help="Publish positioning data via MQTT (auto-enables positioning)"
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Send positioning data via HTTP POST (auto-enables positioning)"
    )
    parser.add_argument(
        "--track", action="store_true",
        help="Enable object tracking with persistent IDs (BoT-SORT)"
    )
    parser.add_argument(
        "--nms", action="store_true",
        help="Apply additional NMS after inference to remove duplicate detections"
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="CLI mode: no preview window, terminal output only (faster, for headless operation)"
    )

    return parser
