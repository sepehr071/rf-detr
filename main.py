"""
RF-DETR Detection Main Application
==================================
Entry point and orchestrator for the bottle detection system.

This is the thin orchestrator that coordinates between specialized modules.
All business logic has been extracted into dedicated modules:
- pipeline.py: Core inference pipeline
- utils/preprocessing.py: Frame preprocessing
- utils/label_io.py: Label file serialization
- utils/network_publish.py: MQTT and HTTP publishing

Usage:
    # SAHI tiling mode (default, better for small objects)
    python main.py

    # Full frame mode
    python main.py --mode full

    # Hybrid mode (full + SAHI combined)
    python main.py --mode hybrid

    # Process single image file
    python main.py --image path/to/image.jpg

    # With object tracking
    python main.py --track

    # Send data via MQTT/HTTP
    python main.py --mqtt
    python main.py --web
"""

import os
import sys
import time
import argparse
import cv2

from config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    VERBOSE_TILES_DIR,
    LABELS_DIR,
    IMAGES_DIR,
    MQTT_LOGS_DIR,
    POSITIONS_DIR,
    CONFIDENCE_INCREMENT,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
    DISPLAY_KEY_WAIT_MS
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


def print_banner(args, has_roi: bool):
    """Print startup configuration banner."""
    print("=" * 60)
    print("RF-DETR DETECTION SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Confidence: {args.conf}")
    print(f"OpenVINO: {'Yes' if args.openvino else 'No'}")
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
    print("=" * 60)


def run_detection_loop(args, pipeline, camera, visualizer, dirs, keypoint_processor=None):
    """
    Main camera detection loop - orchestration only.

    Args:
        args: Parsed CLI arguments
        pipeline: Configured InferencePipeline
        camera: ImageCapture instance
        visualizer: Visualizer instance
        dirs: Dict of output directory paths
        keypoint_processor: Optional KeypointProcessor for big object angle detection
    """
    from utils.label_io import save_detections_to_txt
    from utils.network_publish import configure_mqtt, publish_mqtt, publish_http

    # Configure MQTT if enabled
    if args.mqtt:
        configure_mqtt()

    print("\n🎯 Starting detection loop...")
    print("   Press 'Q' to quit")
    print("   Press 'S' to save screenshot")
    print("   Press 'M' to toggle masks")
    print("   Press '+'/'-' to adjust confidence")
    print()

    # State variables
    show_masks = True
    fps = 0.0
    frame_times = []
    frame_counter = 0
    confidence = args.conf

    try:
        while True:
            loop_start = time.time()

            # === CAPTURE IMAGE ===
            frame = camera.capture_single()
            if frame is None:
                print("⚠️ Failed to capture frame, retrying...")
                continue

            # === RUN PIPELINE ===
            result = pipeline.run(frame)
            detections = result.detections
            cropped_detections = result.cropped_detections

            # === SAVE LABELS (for debugging) ===
            if args.save_labels and len(detections) > 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_name = f"frame_{timestamp}_{frame_counter:06d}"

                label_path = os.path.join(dirs['labels'], f"{base_name}.txt")
                save_detections_to_txt(
                    detections, label_path,
                    result.original_height, result.original_width
                )

                # Save raw image for reference (use frame from result)
                image_path = os.path.join(dirs['images'], f"{base_name}.jpg")
                cv2.imwrite(image_path, result.frame_resized)

                frame_counter += 1

            # === CALCULATE KEYPOINT ANGLES FOR BIG OBJECTS ===
            angle_map = {}
            if keypoint_processor and len(detections) > 0:
                angle_map = keypoint_processor.process(detections, result.frame_resized)

            # === CALCULATE AND PUBLISH POSITIONS ===
            if (args.positioning or args.web or args.mqtt) and len(detections) > 0:
                from Positioning import calculate_positions_from_detections

                positions = calculate_positions_from_detections(
                    detections,
                    image_width=result.original_width,
                    image_height=result.original_height,
                    enable_collision_resolution=True,
                    angle_map=angle_map,
                )

                if args.mqtt and positions:
                    pub_result = publish_mqtt(positions, log_dir=dirs.get('mqtt_logs', MQTT_LOGS_DIR))
                    print(f"[MQTT] {pub_result.message}")

                if args.web and positions:
                    pub_result = publish_http(positions)
                    print(f"[WEB] {pub_result.message}")

            # === DRAW RESULTS ===
            if pipeline.roi and pipeline.roi.is_complete():
                # Use cropped/masked frame for display (already in result)
                annotated = visualizer.draw_detections(
                    result.frame_display, cropped_detections, show_masks=show_masks
                )
            else:
                # No ROI - use full resized frame
                annotated = visualizer.draw_detections(
                    result.frame_resized, detections, show_masks=show_masks
                )

            # === CALCULATE FPS ===
            frame_times.append(time.time() - loop_start)
            if len(frame_times) > 10:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))

            # === DRAW SUMMARY PANEL ===
            display = visualizer.draw_summary(
                annotated, detections, fps, args.mode, result.inference_time
            )

            # === SHOW RESULT ===
            visualizer.show(display)

            # === LOG TIMING ===
            if args.log:
                result.timings['total'] = time.time() - loop_start
                timing_str = " | ".join(f"{k}: {int(v*1000)}ms" for k, v in result.timings.items())
                print(f"[LOG] {timing_str}")

            # === HANDLE INPUT ===
            key = visualizer.handle_input(DISPLAY_KEY_WAIT_MS)

            if key.lower() == 'q':
                print("\n👋 Quitting...")
                break
            elif key.lower() == 's':
                visualizer.save_screenshot(display)
            elif key.lower() == 'm':
                show_masks = not show_masks
                print(f"🎭 Masks: {'ON' if show_masks else 'OFF'}")
            elif key == '+' or key == '=':
                confidence = min(CONFIDENCE_MAX, confidence + CONFIDENCE_INCREMENT)
                pipeline.update_confidence(confidence)
                print(f"📈 Confidence: {confidence:.2f}")
            elif key == '-' or key == '_':
                confidence = max(CONFIDENCE_MIN, confidence - CONFIDENCE_INCREMENT)
                pipeline.update_confidence(confidence)
                print(f"📉 Confidence: {confidence:.2f}")

    finally:
        camera.release()
        visualizer.destroy()
        if keypoint_processor:
            keypoint_processor.cleanup()
        print("✅ Done!")


def run_image_inference(args, pipeline, dirs, keypoint_processor=None):
    """
    Single image inference - orchestration only.

    Args:
        args: Parsed CLI arguments
        pipeline: Configured InferencePipeline
        dirs: Dict of output directory paths
        keypoint_processor: Optional KeypointProcessor for big object angle detection
    """
    from camera import load_image
    from utils.label_io import save_detections_to_txt
    from utils.network_publish import configure_mqtt, publish_mqtt, publish_http
    from visualization import Visualizer

    # Load image
    frame = load_image(args.image)
    if frame is None:
        return

    # Configure MQTT if enabled
    if args.mqtt:
        configure_mqtt()

    print(f"\n🔍 Running inference ({args.mode} mode)...")

    # === RUN PIPELINE ===
    result = pipeline.run(frame)
    detections = result.detections
    cropped_detections = result.cropped_detections

    print(f"✅ Inference completed in {result.inference_time:.2f}s")
    print(f"\n📊 Detected {len(detections)} objects")

    # === LOG TIMING ===
    if args.log:
        timing_str = " | ".join(f"{k}: {int(v*1000)}ms" for k, v in result.timings.items())
        print(f"\n[LOG] {timing_str}")

    # === SAVE LABELS ===
    if args.save_labels and len(detections) > 0:
        base_name = os.path.splitext(os.path.basename(args.image))[0]

        label_path = os.path.join(dirs['labels'], f"{base_name}.txt")
        save_detections_to_txt(
            detections, label_path,
            result.original_height, result.original_width
        )
        print(f"📝 Labels saved to: {label_path}")

        # Save raw image (use frame from result)
        raw_image_path = os.path.join(dirs['images'], f"{base_name}.jpg")
        cv2.imwrite(raw_image_path, result.frame_resized)
        print(f"🖼️ Raw image saved to: {raw_image_path}")

    # === CALCULATE KEYPOINT ANGLES FOR BIG OBJECTS ===
    angle_map = {}
    if keypoint_processor and len(detections) > 0:
        angle_map = keypoint_processor.process(detections, result.frame_resized)

    # === CALCULATE AND PUBLISH POSITIONS ===
    if (args.positioning or args.web or args.mqtt) and len(detections) > 0:
        from Positioning import calculate_positions_from_detections

        positions = calculate_positions_from_detections(
            detections,
            image_width=result.original_width,
            image_height=result.original_height,
            enable_collision_resolution=True,
            angle_map=angle_map,
        )

        if positions:
            print(f"📍 Calculated {len(positions)} positions")

        if args.mqtt and positions:
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            pub_result = publish_mqtt(
                positions,
                log_dir=dirs.get('mqtt_logs', MQTT_LOGS_DIR),
                log_name=f"{base_name}.txt"
            )
            print(f"[MQTT] {pub_result.message}")

        if args.web and positions:
            pub_result = publish_http(positions)
            print(f"[WEB] {pub_result.message}")

    # === VISUALIZE ===
    visualizer = Visualizer()

    if pipeline.roi and pipeline.roi.is_complete():
        # Use cropped/masked frame for display (already in result)
        annotated = visualizer.draw_detections(
            result.frame_display, cropped_detections, show_masks=True
        )
    else:
        # No ROI - use full resized frame
        annotated = visualizer.draw_detections(
            result.frame_resized, detections, show_masks=True
        )

    display = visualizer.draw_summary(annotated, detections, 0, args.mode, result.inference_time)

    # Save result
    output_path = args.output
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = f"{base_name}_detection.jpg"

    cv2.imwrite(output_path, display)
    print(f"✅ Result saved to: {output_path}")

    # Show result
    print("\n📺 Displaying result... Press any key to close")
    h, w = display.shape[:2]
    cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
    screen_scale = min(1.0, 1400 / w, 900 / h)
    cv2.resizeWindow("Detection Result", int(w * screen_scale), int(h * screen_scale))
    cv2.imshow("Detection Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
        "--openvino", action="store_true",
        help="Enable OpenVINO optimization for Intel CPU"
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

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        checkpoint_dir = os.path.dirname(args.checkpoint) or "runs/rfdetr_seg_training"
        if os.path.exists(checkpoint_dir):
            print("\nAvailable checkpoints:")
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pth'):
                    print(f"  - {os.path.join(checkpoint_dir, f)}")
        sys.exit(1)

    # Setup directories
    dirs = setup_directories(args)

    # Create pipeline configuration
    from pipeline import PipelineConfig, create_pipeline

    config = PipelineConfig(
        mode=args.mode,
        confidence=args.conf,
        nms_enabled=args.nms,
        track_enabled=args.track if not args.image else False,  # No tracking for single images
        verbose=args.verbose,
        tiles_dir=dirs.get('tiles')
    )

    # Create pipeline
    print("\n📦 Loading model...")
    pipeline = create_pipeline(
        checkpoint_path=args.checkpoint,
        use_openvino=args.openvino,
        config=config
    )

    # Initialize keypoint processor if positioning enabled (for big object rotation)
    keypoint_processor = None
    if args.positioning or args.web or args.mqtt:
        from utils.keypoint_processor import KeypointProcessor
        keypoint_processor = KeypointProcessor(save_debug_crops=args.verbose)
        print("🔑 Keypoint processor initialized for rotation detection")
        if args.verbose:
            print("   Debug crops will be saved to debug_crops/ directory")

    # Check ROI
    has_roi = pipeline.roi is not None
    if not has_roi and not args.image:
        print("⚠️ No ROI configured. Run 'python calibrate.py' first.")
        response = input("Continue without ROI? (y/n): ")
        if response.lower() != 'y':
            return

    # Print banner
    print_banner(args, has_roi)

    # Run appropriate mode
    if args.image:
        run_image_inference(args, pipeline, dirs, keypoint_processor)
    else:
        from camera import ImageCapture
        from visualization import Visualizer

        print("\n📷 Opening camera...")
        camera = ImageCapture(args.camera, CAMERA_WIDTH, CAMERA_HEIGHT)
        if not camera.open():
            print("❌ Failed to open camera. Exiting.")
            return

        visualizer = Visualizer()
        visualizer.create_window(CAMERA_WIDTH, CAMERA_HEIGHT)

        if args.track:
            print("\n🔗 Tracker initialized (BoT-SORT)")

        run_detection_loop(args, pipeline, camera, visualizer, dirs, keypoint_processor)


if __name__ == "__main__":
    main()
