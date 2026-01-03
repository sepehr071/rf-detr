"""
Startup banner display for RF-DETR detection system.
"""

from config import CAMERA_WIDTH, CAMERA_HEIGHT, VERBOSE_TILES_DIR


def print_banner(args, has_roi: bool, use_openvino: bool):
    """Print startup configuration banner."""
    print("=" * 60)
    print("RF-DETR DETECTION SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Confidence: {args.conf}")
    print(f"OpenVINO: {'Yes' if use_openvino else 'No (disabled)'}")
    if not args.image:
        print(f"Camera: {args.camera} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    else:
        print(f"Image: {args.image}")
    print(f"ROI: {'Loaded' if has_roi else 'Not configured'}")
    print(f"Save Labels: {'Yes -> ' + args.labels_dir if args.save_labels else 'No'}")
    print(f"Positioning: {'Yes' if args.positioning or args.web or args.mqtt else 'No'}")
    print(f"MQTT: {'Yes' if args.mqtt else 'No'}")
    print(f"Web POST: {'Yes' if args.web else 'No'}")
    if not args.image:
        print(f"Tracking: {'Yes (BoT-SORT)' if args.track else 'No'}")
    print(f"NMS: {'Yes' if args.nms else 'No'}")
    print(f"Verbose Tiles: {'Yes -> ' + VERBOSE_TILES_DIR if args.verbose else 'No'}")
    print(f"CLI Mode: {'Yes (no preview)' if args.cli else 'No (GUI)'}")
    print("=" * 60)
