"""
RF-DETR Detection Main Application
==================================
Main entry point for the bottle detection system.

Captures single images from camera, processes with RF-DETR model,
and displays results continuously.

Usage:
    # Full frame mode (default)
    python main.py
    
    # SAHI tiling mode (better for small objects)
    python main.py --mode sahi
    
    # Custom confidence threshold
    python main.py --conf 0.4
    
    # With OpenVINO optimization
    python main.py --openvino
    
    # Process single image file
    python main.py --image path/to/image.jpg
    
    # Save detections to text file (class_id + polygon per line)
    python main.py --save-labels
    python main.py --image test.jpg --save-labels --labels-dir labels/
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

from config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT
)


def save_detections_to_txt(
    detections,
    output_path: str,
    image_height: int,
    image_width: int
):
    """
    Save detections to a text file with class_id, polygon, and bounding box per line.

    Format: class_id poly_x1 poly_y1 ... | bbox_x1 bbox_y1 bbox_x2 bbox_y2 (normalized coordinates)
    The pipe '|' separates polygon points from bounding box.
    If mask is available, uses mask polygon. Otherwise uses bounding box corners as polygon.
    
    Args:
        detections: supervision Detections object
        output_path: Path to output .txt file
        image_height: Image height for normalization
        image_width: Image width for normalization
    """
    lines = []
    
    for i in range(len(detections)):
        class_id = detections.class_id[i]
        
        # Try to get polygon from mask
        if detections.mask is not None and detections.mask[i] is not None:
            mask = detections.mask[i].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify polygon if it has too many points
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Flatten and normalize coordinates
                points = simplified.reshape(-1, 2)
                normalized_points = []
                for x, y in points:
                    normalized_points.extend([x / image_width, y / image_height])
                
                # Format: class_id poly_x1 poly_y1 ... | bbox_x1 bbox_y1 bbox_x2 bbox_y2
                coords_str = ' '.join(f"{p:.6f}" for p in normalized_points)
                box = detections.xyxy[i]
                bbox_str = f"{box[0]/image_width:.6f} {box[1]/image_height:.6f} {box[2]/image_width:.6f} {box[3]/image_height:.6f}"
                lines.append(f"{class_id} {coords_str} | {bbox_str}")
        else:
            # Fall back to bounding box as polygon
            box = detections.xyxy[i]
            x1, y1, x2, y2 = box
            
            # Normalize coordinates
            x1_n, x2_n = x1 / image_width, x2 / image_width
            y1_n, y2_n = y1 / image_height, y2 / image_height
            
            # Box as polygon: 4 corners + bbox at end
            lines.append(f"{class_id} {x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f} {x1_n:.6f} {y2_n:.6f} | {x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f}")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return len(lines)


def format_mqtt_payload(
    positions: list,
    store_id: int = 1,
    device_id: int = 1,
    level: int = 4
) -> dict:
    """
    Transform positioning data to MQTT payload format.

    Args:
        positions: List of position dictionaries from calculate_positions()
        store_id: Store identifier (hardcoded)
        device_id: Device identifier (hardcoded)
        level: Shelf level (hardcoded)

    Returns:
        Dict with store_id, device_id, products, and other_products
    """
    products = []
    other_products = []

    for pos in positions:
        if 'shelf_position' not in pos:
            continue  # Skip error entries

        item = {
            "class_id": pos["class_id"],
            "level": level,
            "facing": pos.get("rotation", 0),
            "position": {
                "x": pos["shelf_position"][0],
                "y": pos["shelf_position"][1]
            }
        }

        if pos["class_id"] == -1:
            other_products.append(item)
        else:
            products.append(item)

    return {
        "store_id": store_id,
        "device_id": device_id,
        "products": products,
        "other_products": other_products
    }


def run_detection_loop(
    checkpoint_path: str,
    mode: str = "full",
    confidence: float = DEFAULT_CONFIDENCE,
    use_openvino: bool = False,
    camera_index: int = CAMERA_INDEX,
    save_labels: bool = False,
    labels_dir: str = "labels",
    log_timing: bool = False,
    verbose: bool = False,
    positioning: bool = False,
    positioning_dir: str = "positions",
    mqtt_enabled: bool = False,
    web_enabled: bool = False,
    track_enabled: bool = False,
    nms_enabled: bool = False
):
    """
    Main detection loop - captures images and processes continuously.

    Args:
        checkpoint_path: Path to model checkpoint
        mode: Inference mode ('full', 'sahi', or 'hybrid')
        confidence: Confidence threshold (0-1)
        use_openvino: Whether to use OpenVINO optimization
        camera_index: Camera device index
        save_labels: Whether to save detection labels to text files (debug/logging)
        labels_dir: Directory to save label files
        log_timing: Whether to print timing breakdown for each frame
        verbose: Whether to save SAHI tile images for debugging
        positioning: Whether to calculate shelf positions (auto-enabled with --web or --mqtt)
        positioning_dir: Directory to save positioning JSON files
        mqtt_enabled: Whether to publish positioning data via MQTT
        web_enabled: Whether to send positioning data via HTTP POST
        track_enabled: Whether to enable object tracking with persistent IDs
        nms_enabled: Whether to apply additional NMS after inference
    """
    from camera import ImageCapture
    from model import DetectionModel, resize_for_inference, offset_detections
    from roi import ROIManager
    from visualization import Visualizer
    from config import VERBOSE_TILES_DIR

    # Auto-enable positioning when web or mqtt is enabled
    if web_enabled or mqtt_enabled:
        positioning = True

    print("=" * 60)
    print("RF-DETR DETECTION SYSTEM")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Confidence: {confidence}")
    print(f"OpenVINO: {'Yes' if use_openvino else 'No'}")
    print(f"Camera: {camera_index} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    print(f"Save Labels: {'Yes -> ' + labels_dir if save_labels else 'No'}")
    print(f"Positioning: {'Yes' if positioning else 'No'}")
    print(f"MQTT: {'Yes' if mqtt_enabled else 'No'}")
    print(f"Web POST: {'Yes' if web_enabled else 'No'}")
    print(f"Tracking: {'Yes (BoT-SORT)' if track_enabled else 'No'}")
    print(f"NMS: {'Yes (threshold=0.45)' if nms_enabled else 'No'}")
    print(f"Log Timing: {'Yes' if log_timing else 'No'}")
    print(f"Verbose Tiles: {'Yes -> ' + VERBOSE_TILES_DIR if verbose else 'No'}")
    print("=" * 60)

    # Create verbose tiles directory if needed
    tiles_dir = None
    if verbose:
        tiles_dir = VERBOSE_TILES_DIR
        os.makedirs(tiles_dir, exist_ok=True)

    # Create labels and images directories if saving
    if save_labels:
        os.makedirs(labels_dir, exist_ok=True)
        images_dir = os.path.join(os.path.dirname(labels_dir) or ".", "images")
        os.makedirs(images_dir, exist_ok=True)
        frame_counter = 0

    # Create positioning directory if needed
    if positioning:
        os.makedirs(positioning_dir, exist_ok=True)

    # Configure MQTT if enabled
    if mqtt_enabled:
        import mqtt_handler
        mqtt_handler.configure(
            broker_host="89.36.137.77",
            broker_port=1883,
            default_topic="test/topic"
        )
        mqtt_logs_dir = "mqtt_logs"
        os.makedirs(mqtt_logs_dir, exist_ok=True)

    # Initialize ROI
    roi = ROIManager()
    has_roi = roi.load()
    if not has_roi:
        print("⚠️ No ROI configured. Run 'python calibrate.py' first.")
        response = input("Continue without ROI? (y/n): ")
        if response.lower() != 'y':
            return
        roi = None
    
    # Initialize model
    print("\n📦 Loading model...")
    model = DetectionModel(checkpoint_path, use_openvino)
    
    # Initialize camera
    print("\n📷 Opening camera...")
    camera = ImageCapture(camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)
    if not camera.open():
        print("❌ Failed to open camera. Exiting.")
        return
    
    # Initialize visualizer
    visualizer = Visualizer()
    visualizer.create_window(CAMERA_WIDTH, CAMERA_HEIGHT)

    # Initialize tracker if enabled
    tracker = None
    if track_enabled:
        from tracker import ObjectTracker
        tracker = ObjectTracker()
        print("\n🔗 Tracker initialized (BoT-SORT)")

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
    
    try:
        while True:
            loop_start = time.time()
            timings = {}

            # === CAPTURE IMAGE ===
            t0 = time.time()
            frame = camera.capture_single()
            timings['capture'] = time.time() - t0
            if frame is None:
                print("⚠️ Failed to capture frame, retrying...")
                continue

            # === RESIZE IF NEEDED ===
            t0 = time.time()
            frame_resized, scale = resize_for_inference(frame)
            original_h, original_w = frame_resized.shape[:2]
            timings['resize'] = time.time() - t0

            # === CROP TO ROI (more efficient than masking) ===
            t0 = time.time()
            x_offset, y_offset = 0, 0
            if roi and roi.is_complete():
                # Scale ROI if image was resized
                if scale != 1.0:
                    scaled_roi = roi.get_scaled(scale)
                else:
                    scaled_roi = roi

                # Crop to ROI bounding box
                frame_cropped, x_offset, y_offset = scaled_roi.crop_to_roi(frame_resized)
                timings['crop'] = time.time() - t0

                # Apply polygon mask within cropped area (for non-rectangular ROI edges)
                t0 = time.time()
                frame_masked = scaled_roi.apply_mask_to_cropped(frame_cropped, x_offset, y_offset)
                timings['mask'] = time.time() - t0
            else:
                frame_masked = frame_resized
                scaled_roi = None
                timings['crop'] = time.time() - t0
                timings['mask'] = 0

            # === CONVERT TO RGB FOR MODEL ===
            t0 = time.time()
            frame_rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
            timings['convert'] = time.time() - t0

            # === RUN INFERENCE ===
            t0 = time.time()
            if mode == "hybrid":
                detections = model.predict_hybrid(frame_rgb, confidence, verbose=verbose, tiles_dir=tiles_dir)
            elif mode == "sahi":
                detections = model.predict_sahi(frame_rgb, confidence, verbose=verbose, tiles_dir=tiles_dir)
            else:
                detections = model.predict(frame_rgb, confidence)
            inference_time = time.time() - t0
            timings['inference'] = inference_time

            # === APPLY NMS (if enabled) ===
            if nms_enabled:
                detections = detections.with_nms(threshold=0.45, class_agnostic=True)

            # === UPDATE TRACKER (if enabled) ===
            if tracker is not None:
                t0 = time.time()
                # Pass BGR frame for tracking (frame_masked is before RGB conversion)
                detections = tracker.update(detections, frame_masked)
                timings['tracking'] = time.time() - t0

            # === KEEP CROPPED DETECTIONS FOR VISUALIZATION ===
            # Detections are currently in cropped coordinate space
            cropped_detections = detections

            # === OFFSET DETECTIONS BACK TO ORIGINAL COORDINATES ===
            t0 = time.time()
            if x_offset != 0 or y_offset != 0:
                detections = offset_detections(
                    detections, x_offset, y_offset, original_h, original_w
                )
            timings['offset'] = time.time() - t0

            # === FILTER DETECTIONS BY ROI POLYGON ===
            t0 = time.time()
            if scaled_roi and scaled_roi.is_complete():
                detections = scaled_roi.filter_detections(detections)
            timings['filter'] = time.time() - t0

            # === SAVE LABELS AND RAW IMAGE (for debugging) ===
            if save_labels and len(detections) > 0:
                # Use original (resized) image dimensions for normalization
                h, w = original_h, original_w
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_name = f"frame_{timestamp}_{frame_counter:06d}"

                # Save label file (coordinates are in original image space)
                label_path = os.path.join(labels_dir, f"{base_name}.txt")
                save_detections_to_txt(detections, label_path, h, w)

                # Save raw image (full resized frame for reference)
                image_path = os.path.join(images_dir, f"{base_name}.jpg")
                cv2.imwrite(image_path, frame_resized)

                frame_counter += 1

            # === CALCULATE POSITIONS AND SEND (independent of save_labels) ===
            if positioning and len(detections) > 0:
                from Positioning import calculate_positions_from_detections
                import json

                # Calculate positions directly from detections
                positions = calculate_positions_from_detections(
                    detections,
                    image_width=original_w,
                    image_height=original_h,
                    enable_collision_resolution=False
                )

                # Publish via MQTT if enabled
                if mqtt_enabled and positions:
                    import mqtt_handler
                    payload = format_mqtt_payload(positions)
                    payload_json = json.dumps(payload, indent=2)

                    # Save to log file
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    mqtt_log_path = os.path.join(mqtt_logs_dir, f"mqtt_{timestamp}.txt")
                    with open(mqtt_log_path, 'w') as f:
                        f.write(payload_json)

                    # Send via MQTT
                    rc = mqtt_handler.send(payload_json)
                    if rc == 0:
                        print(f"[MQTT] Sent {len(positions)} positions")
                    else:
                        print(f"[MQTT] Failed to send (rc={rc})")

                # Send via HTTP POST if enabled
                if web_enabled and positions:
                    import requests
                    payload = format_mqtt_payload(positions)
                    try:
                        response = requests.post(
                            "http://91.107.184.69:3000/data/process",
                            json=payload,
                            timeout=5
                        )
                        if response.status_code == 200:
                            print(f"[WEB] Sent {len(positions)} positions (status={response.status_code})")
                        else:
                            print(f"[WEB] Server returned status={response.status_code}")
                    except requests.exceptions.RequestException as e:
                        print(f"[WEB] Failed to send: {e}")

            # === DRAW RESULTS ===
            t0 = time.time()
            # Draw on cropped frame (what model sees) with cropped coordinates
            if roi and roi.is_complete():
                # Convert cropped frame back to BGR for display
                display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                annotated = visualizer.draw_detections(
                    display_frame, cropped_detections, show_masks=show_masks
                )
            else:
                # No ROI, use full frame
                annotated = visualizer.draw_detections(
                    frame_resized, detections, show_masks=show_masks
                )

            # === CALCULATE FPS ===
            frame_times.append(time.time() - loop_start)
            if len(frame_times) > 10:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))

            # === DRAW SUMMARY PANEL ===
            display = visualizer.draw_summary(
                annotated, detections, fps, mode, inference_time
            )
            timings['draw'] = time.time() - t0

            # === SHOW RESULT ===
            t0 = time.time()
            visualizer.show(display)
            timings['display'] = time.time() - t0

            # === LOG TIMING ===
            timings['total'] = time.time() - loop_start
            if log_timing:
                timing_str = " | ".join(f"{k}: {int(v*1000)}ms" for k, v in timings.items())
                print(f"[LOG] {timing_str}")

            # === HANDLE INPUT ===
            key = visualizer.handle_input(1)

            if key.lower() == 'q':
                print("\n👋 Quitting...")
                break
            elif key.lower() == 's':
                visualizer.save_screenshot(display)
            elif key.lower() == 'm':
                show_masks = not show_masks
                print(f"🎭 Masks: {'ON' if show_masks else 'OFF'}")
            elif key == '+' or key == '=':
                confidence = min(0.95, confidence + 0.05)
                print(f"📈 Confidence: {confidence:.2f}")
            elif key == '-' or key == '_':
                confidence = max(0.05, confidence - 0.05)
                print(f"📉 Confidence: {confidence:.2f}")
    
    finally:
        # Cleanup
        camera.release()
        visualizer.destroy()
        print("✅ Done!")


def run_image_inference(
    checkpoint_path: str,
    image_path: str,
    mode: str = "full",
    confidence: float = DEFAULT_CONFIDENCE,
    use_openvino: bool = False,
    output_path: str = None,
    save_labels: bool = False,
    labels_dir: str = "labels",
    log_timing: bool = False,
    verbose: bool = False,
    positioning: bool = False,
    positioning_dir: str = "positions",
    mqtt_enabled: bool = False,
    web_enabled: bool = False,
    nms_enabled: bool = False
):
    """
    Run inference on a single image file.

    Args:
        checkpoint_path: Path to model checkpoint
        image_path: Path to input image
        mode: Inference mode ('full', 'sahi', or 'hybrid')
        confidence: Confidence threshold (0-1)
        use_openvino: Whether to use OpenVINO optimization
        output_path: Path for output image (optional)
        save_labels: Whether to save detection labels to text file (debug/logging)
        labels_dir: Directory to save label files
        log_timing: Whether to print timing breakdown
        verbose: Whether to save SAHI tile images for debugging
        positioning: Whether to calculate shelf positions (auto-enabled with --web or --mqtt)
        positioning_dir: Directory to save positioning JSON files
        mqtt_enabled: Whether to publish positioning data via MQTT
        web_enabled: Whether to send positioning data via HTTP POST
        nms_enabled: Whether to apply additional NMS after inference
    """
    from camera import load_image
    from model import DetectionModel, resize_for_inference, offset_detections
    from roi import ROIManager
    from visualization import Visualizer
    from config import VERBOSE_TILES_DIR

    # Auto-enable positioning when web or mqtt is enabled
    if web_enabled or mqtt_enabled:
        positioning = True

    print("=" * 60)
    print("RF-DETR IMAGE INFERENCE")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Mode: {mode.upper()}")
    print(f"Confidence: {confidence}")
    print(f"Save Labels: {'Yes -> ' + labels_dir if save_labels else 'No'}")
    print(f"Positioning: {'Yes' if positioning else 'No'}")
    print(f"MQTT: {'Yes' if mqtt_enabled else 'No'}")
    print(f"Web POST: {'Yes' if web_enabled else 'No'}")
    print(f"NMS: {'Yes (threshold=0.45)' if nms_enabled else 'No'}")
    print(f"Log Timing: {'Yes' if log_timing else 'No'}")
    print(f"Verbose Tiles: {'Yes -> ' + VERBOSE_TILES_DIR if verbose else 'No'}")
    print("=" * 60)

    # Create verbose tiles directory if needed
    tiles_dir = None
    if verbose:
        tiles_dir = VERBOSE_TILES_DIR
        os.makedirs(tiles_dir, exist_ok=True)

    total_start = time.time()
    timings = {}

    # Load image
    t0 = time.time()
    frame = load_image(image_path)
    timings['load'] = time.time() - t0
    if frame is None:
        return

    # Initialize ROI
    roi = ROIManager()
    has_roi = roi.load()
    if not has_roi:
        print("⚠️ No ROI configured. Processing full image.")
        roi = None

    # Initialize model
    print("\n📦 Loading model...")
    t0 = time.time()
    model = DetectionModel(checkpoint_path, use_openvino)
    timings['model_load'] = time.time() - t0

    # Resize if needed
    t0 = time.time()
    frame_resized, scale = resize_for_inference(frame)
    original_h, original_w = frame_resized.shape[:2]
    timings['resize'] = time.time() - t0

    # Crop to ROI (more efficient than masking)
    t0 = time.time()
    x_offset, y_offset = 0, 0
    if roi and roi.is_complete():
        if scale != 1.0:
            scaled_roi = roi.get_scaled(scale)
        else:
            scaled_roi = roi

        # Crop to ROI bounding box
        frame_cropped, x_offset, y_offset = scaled_roi.crop_to_roi(frame_resized)
        timings['crop'] = time.time() - t0

        # Apply polygon mask within cropped area
        t0 = time.time()
        frame_masked = scaled_roi.apply_mask_to_cropped(frame_cropped, x_offset, y_offset)
        timings['mask'] = time.time() - t0
    else:
        frame_masked = frame_resized
        scaled_roi = None
        timings['crop'] = time.time() - t0
        timings['mask'] = 0

    # Convert to RGB
    t0 = time.time()
    frame_rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
    timings['convert'] = time.time() - t0

    # Run inference
    print(f"\n🔍 Running inference ({mode} mode)...")
    t0 = time.time()
    if mode == "hybrid":
        detections = model.predict_hybrid(frame_rgb, confidence, verbose=verbose, tiles_dir=tiles_dir)
    elif mode == "sahi":
        detections = model.predict_sahi(frame_rgb, confidence, verbose=verbose, tiles_dir=tiles_dir)
    else:
        detections = model.predict(frame_rgb, confidence)
    inference_time = time.time() - t0
    timings['inference'] = inference_time
    print(f"✅ Inference completed in {inference_time:.2f}s")

    # === APPLY NMS (if enabled) ===
    if nms_enabled:
        detections = detections.with_nms(threshold=0.45, class_agnostic=True)

    # Keep cropped detections for visualization
    cropped_detections = detections

    # Offset detections back to original coordinates
    t0 = time.time()
    if x_offset != 0 or y_offset != 0:
        detections = offset_detections(
            detections, x_offset, y_offset, original_h, original_w
        )
    timings['offset'] = time.time() - t0

    # Filter detections by ROI polygon
    t0 = time.time()
    if scaled_roi and scaled_roi.is_complete():
        detections = scaled_roi.filter_detections(detections)
    timings['filter'] = time.time() - t0

    # Print summary
    print(f"\n📊 Detected {len(detections)} objects")

    # Log timing
    timings['total'] = time.time() - total_start
    if log_timing:
        timing_str = " | ".join(f"{k}: {int(v*1000)}ms" for k, v in timings.items())
        print(f"\n[LOG] {timing_str}")

    # Save labels and raw image (for debugging)
    if save_labels and len(detections) > 0:
        os.makedirs(labels_dir, exist_ok=True)
        images_dir = os.path.join(os.path.dirname(labels_dir) or ".", "images")
        os.makedirs(images_dir, exist_ok=True)

        # Use original image dimensions for normalization
        h, w = original_h, original_w
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save label file (coordinates are in original image space)
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        save_detections_to_txt(detections, label_path, h, w)
        print(f"📝 Labels saved to: {label_path}")

        # Save raw image (full resized frame for reference)
        raw_image_path = os.path.join(images_dir, f"{base_name}.jpg")
        cv2.imwrite(raw_image_path, frame_resized)
        print(f"🖼️ Raw image saved to: {raw_image_path}")

    # Calculate positions and send (independent of save_labels)
    if positioning and len(detections) > 0:
        from Positioning import calculate_positions_from_detections
        import json

        # Calculate positions directly from detections
        positions = calculate_positions_from_detections(
            detections,
            image_width=original_w,
            image_height=original_h,
            enable_collision_resolution=False
        )

        if positions:
            print(f"📍 Calculated {len(positions)} positions")

        # Publish via MQTT if enabled
        if mqtt_enabled and positions:
            import mqtt_handler
            mqtt_handler.configure(
                broker_host="89.36.137.77",
                broker_port=1883,
                default_topic="test/topic"
            )
            payload = format_mqtt_payload(positions)
            payload_json = json.dumps(payload, indent=2)

            # Save to file
            mqtt_logs_dir = "mqtt_logs"
            os.makedirs(mqtt_logs_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mqtt_log_path = os.path.join(mqtt_logs_dir, f"{base_name}.txt")
            with open(mqtt_log_path, 'w') as f:
                f.write(payload_json)

            # Send via MQTT
            rc = mqtt_handler.send(payload_json)
            if rc == 0:
                print(f"[MQTT] Sent {len(positions)} positions")
            else:
                print(f"[MQTT] Failed to send (rc={rc})")

        # Send via HTTP POST if enabled
        if web_enabled and positions:
            import requests
            payload = format_mqtt_payload(positions)
            try:
                response = requests.post(
                    "http://91.107.184.69:3000/data/process",
                    json=payload,
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"[WEB] Sent {len(positions)} positions (status={response.status_code})")
                else:
                    print(f"[WEB] Server returned status={response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"[WEB] Failed to send: {e}")

    # Visualize - use cropped frame (what model sees) with cropped coordinates
    visualizer = Visualizer()
    if roi and roi.is_complete():
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        annotated = visualizer.draw_detections(display_frame, cropped_detections, show_masks=True)
    else:
        annotated = visualizer.draw_detections(frame_resized, detections, show_masks=True)
    display = visualizer.draw_summary(annotated, detections, 0, mode, inference_time)
    
    # Save result
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RF-DETR Detection System")
    
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file for single image inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for image inference result"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "sahi", "hybrid"],
        default="sahi",
        help="Inference mode: 'full' for full frame, 'sahi' for tiled, 'hybrid' for full+sahi combined"
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--openvino",
        action="store_true",
        help="Enable OpenVINO optimization for Intel CPU"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--save-labels",
        action="store_true",
        help="Save detections to text file (class_id + polygon per line)"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="labels",
        help="Directory to save label files (default: labels/)"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Print timing breakdown for each pipeline stage per frame"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Save SAHI tile images to verbose_tiles/ directory for debugging"
    )
    parser.add_argument(
        "--positioning",
        action="store_true",
        help="Calculate shelf positions after detection (auto-enabled with --web or --mqtt)"
    )
    parser.add_argument(
        "--positioning-dir",
        type=str,
        default="positions",
        help="Directory to save positioning JSON files (default: positions/)"
    )
    parser.add_argument(
        "--mqtt",
        action="store_true",
        help="Publish positioning data via MQTT (auto-enables positioning)"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Send positioning data via HTTP POST (auto-enables positioning)"
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable object tracking with persistent IDs (BoT-SORT)"
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Apply additional NMS after inference to remove duplicate detections"
    )

    args = parser.parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = os.path.dirname(args.checkpoint) or "runs/rfdetr_seg_training"
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pth'):
                    print(f"  - {os.path.join(checkpoint_dir, f)}")
        sys.exit(1)
    
    # Run image or camera mode
    if args.image:
        run_image_inference(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            mode=args.mode,
            confidence=args.conf,
            use_openvino=args.openvino,
            output_path=args.output,
            save_labels=args.save_labels,
            labels_dir=args.labels_dir,
            log_timing=args.log,
            verbose=args.verbose,
            positioning=args.positioning,
            positioning_dir=args.positioning_dir,
            mqtt_enabled=args.mqtt,
            web_enabled=args.web,
            nms_enabled=args.nms
        )
    else:
        run_detection_loop(
            checkpoint_path=args.checkpoint,
            mode=args.mode,
            confidence=args.conf,
            use_openvino=args.openvino,
            camera_index=args.camera,
            save_labels=args.save_labels,
            labels_dir=args.labels_dir,
            log_timing=args.log,
            verbose=args.verbose,
            positioning=args.positioning,
            positioning_dir=args.positioning_dir,
            mqtt_enabled=args.mqtt,
            web_enabled=args.web,
            track_enabled=args.track,
            nms_enabled=args.nms
        )


if __name__ == "__main__":
    main()
