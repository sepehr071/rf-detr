"""
Directory setup utilities for RF-DETR detection system.
"""

import os

from config import (
    VERBOSE_TILES_DIR,
    LABELS_DIR,
    IMAGES_DIR,
    MQTT_LOGS_DIR,
    POSITIONS_DIR,
)


def setup_directories(args) -> dict:
    """Create required output directories and return paths."""
    dirs = {}

    if args.verbose:
        dirs['tiles'] = VERBOSE_TILES_DIR
        os.makedirs(VERBOSE_TILES_DIR, exist_ok=True)

    if args.save_labels:
        dirs['labels'] = args.labels_dir
        dirs['images'] = os.path.join(os.path.dirname(args.labels_dir) or ".", IMAGES_DIR)
        os.makedirs(dirs['labels'], exist_ok=True)
        os.makedirs(dirs['images'], exist_ok=True)

    if args.positioning or args.web or args.mqtt:
        dirs['positions'] = args.positioning_dir
        os.makedirs(dirs['positions'], exist_ok=True)

    if args.mqtt:
        dirs['mqtt_logs'] = MQTT_LOGS_DIR
        os.makedirs(MQTT_LOGS_DIR, exist_ok=True)

    return dirs
