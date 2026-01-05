"""
Detection Runner
================
Orchestrates the detection pipeline for camera and image modes.

Flow:
    DetectionRunner(args)
        │
        ├── setup()
        │   ├── setup_directories()      → utils/io_setup.py
        │   ├── create_pipeline()        → pipeline.py
        │   ├── KeypointProcessor()      → utils/keypoint_processor.py
        │   ├── print_banner()           → utils/banner.py
        │   └── ImageCapture()           → camera.py
        │
        └── run()
            ├── _run_camera_loop()  OR  _run_image()
            │   ├── capture/load         → camera.py
            │   ├── pipeline.run()       → pipeline.py
            │   ├── save_labels()        → utils/label_io.py
            │   ├── keypoint_processor() → utils/keypoint_processor.py
            │   ├── calculate_positions()→ Positioning/
            │   ├── publish_mqtt/http()  → utils/network_publish.py
            │   └── visualizer.draw()    → visualization.py
            │
            └── cleanup()
"""

import os
import time
import cv2

from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MQTT_LOGS_DIR,
    CONFIDENCE_INCREMENT,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
    DISPLAY_KEY_WAIT_MS,
)


class DetectionRunner:
    """
    Main detection orchestrator.

    Coordinates all components for real-time camera or single image detection.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, args):
        """Initialize runner with CLI arguments."""
        self.args = args

        # Components (initialized in setup)
        self.pipeline = None           # → pipeline.py
        self.camera = None             # → camera.py
        self.visualizer = None         # → visualization.py
        self.keypoint_processor = None # → utils/keypoint_processor.py
        self.dirs = {}                 # → utils/io_setup.py

        # Settings
        self.cli_mode = getattr(args, 'cli', False)
        self.use_openvino = not getattr(args, 'no_openvino', False)
        self.confidence = args.conf
        self.show_masks = True

    # =========================================================================
    # SETUP - Initialize all components
    # =========================================================================

    def setup(self) -> bool:
        """
        Initialize all components before running.

        Flow:
            1. Setup directories      → utils/io_setup.py
            2. Create pipeline        → pipeline.py
            3. Init keypoint processor→ utils/keypoint_processor.py (optional)
            4. Check ROI              → roi.py
            5. Print banner           → utils/banner.py
            6. Open camera            → camera.py (camera mode only)
            7. Create visualizer      → visualization.py (GUI mode only)

        Returns:
            bool: True if setup successful, False otherwise
        """
        # -----------------------------------------------------------------
        # Step 1: Setup directories
        # -----------------------------------------------------------------
        from utils.io_setup import setup_directories
        self.dirs = setup_directories(self.args)

        # -----------------------------------------------------------------
        # Step 2: Create inference pipeline
        # -----------------------------------------------------------------
        from pipeline import PipelineConfig, create_pipeline

        config = PipelineConfig(
            mode=self.args.mode,
            confidence=self.args.conf,
            nms_enabled=self.args.nms,
            track_enabled=self.args.track if not self.args.image else False,
            verbose=self.args.verbose,
            tiles_dir=self.dirs.get('tiles')
        )

        print("\n Loading model...")
        self.pipeline = create_pipeline(
            checkpoint_path=self.args.checkpoint,
            use_openvino=self.use_openvino,
            config=config
        )

        # -----------------------------------------------------------------
        # Step 3: Initialize keypoint processor (for rotation detection)
        # -----------------------------------------------------------------
        if self.args.positioning or self.args.web or self.args.mqtt:
            from utils.keypoint_processor import KeypointProcessor
            self.keypoint_processor = KeypointProcessor(
                use_openvino=self.use_openvino,
                save_debug_crops=self.args.verbose
            )
            print(" Keypoint processor initialized")

        # -----------------------------------------------------------------
        # Step 4: Check ROI configuration
        # -----------------------------------------------------------------
        has_roi = self.pipeline.roi is not None
        if not has_roi and not self.args.image:
            print(" No ROI configured. Run 'python calibrate.py' first.")
            response = input("Continue without ROI? (y/n): ")
            if response.lower() != 'y':
                return False

        # -----------------------------------------------------------------
        # Step 5: Print startup banner
        # -----------------------------------------------------------------
        from utils.banner import print_banner
        print_banner(self.args, has_roi, self.use_openvino)

        # -----------------------------------------------------------------
        # Step 6: Open camera (camera mode only)
        # -----------------------------------------------------------------
        if not self.args.image:
            from camera import ImageCapture, find_available_camera

            # Auto-detect camera if not specified
            camera_index = self.args.camera
            if camera_index is None:
                camera_index = find_available_camera()

            print(f"\n Opening camera {camera_index}...")
            self.camera = ImageCapture(camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)
            if not self.camera.open():
                print(" Failed to open camera.")
                return False

            # -------------------------------------------------------------
            # Step 7: Create visualizer (GUI mode only)
            # -------------------------------------------------------------
            if not self.cli_mode:
                from visualization import Visualizer
                self.visualizer = Visualizer()
                self.visualizer.create_window(CAMERA_WIDTH, CAMERA_HEIGHT)

            if self.args.track:
                print("\n Tracker initialized (BoT-SORT)")

        return True

    # =========================================================================
    # RUN - Main entry point
    # =========================================================================

    def run(self):
        """
        Run detection - automatically selects camera or image mode.

        Flow:
            setup() → [_run_camera_loop() OR _run_image()] → cleanup()
        """
        if not self.setup():
            return

        try:
            if self.args.image:
                self._run_image()
            else:
                self._run_camera_loop()
        finally:
            self.cleanup()

    # =========================================================================
    # CAMERA LOOP - Real-time detection
    # =========================================================================

    def _run_camera_loop(self):
        """
        Main camera detection loop.

        Flow (per frame):
            1. Capture frame           → camera.py
            2. Run pipeline            → pipeline.py
            3. Save labels (optional)  → utils/label_io.py
            4. Process keypoints       → utils/keypoint_processor.py
            5. Calculate positions     → Positioning/
            6. Publish MQTT/HTTP       → utils/network_publish.py
            7. Calculate FPS
            8. Display results         → visualization.py
            9. Handle keyboard input
        """
        from utils.label_io import save_detections_to_txt
        from utils.network_publish import configure_mqtt, publish_mqtt, publish_http

        # Configure MQTT
        if self.args.mqtt:
            configure_mqtt()

        self._print_controls()

        fps = 0.0
        frame_times = []
        frame_counter = 0

        try:
            while True:
                loop_start = time.time()

                # ---------------------------------------------------------
                # Step 1: Capture frame → camera.py
                # ---------------------------------------------------------
                frame = self.camera.capture_single()
                if frame is None:
                    print(" Failed to capture frame, retrying...")
                    continue

                # ---------------------------------------------------------
                # Step 2: Run inference pipeline → pipeline.py
                # ---------------------------------------------------------
                result = self.pipeline.run(frame)
                detections = result.detections
                cropped_detections = result.cropped_detections

                # ---------------------------------------------------------
                # Step 3: Save labels (optional) → utils/label_io.py
                # ---------------------------------------------------------
                if self.args.save_labels and len(detections) > 0:
                    frame_counter = self._save_frame_labels(
                        result, detections, frame_counter, save_detections_to_txt
                    )

                # ---------------------------------------------------------
                # Step 4: Process keypoints → utils/keypoint_processor.py
                # ---------------------------------------------------------
                angle_map = {}
                if self.keypoint_processor and len(detections) > 0:
                    angle_map = self.keypoint_processor.process(
                        detections, result.frame_resized
                    )

                # ---------------------------------------------------------
                # Step 5-6: Calculate positions & publish → Positioning/, network_publish.py
                # ---------------------------------------------------------
                if (self.args.positioning or self.args.web or self.args.mqtt) and len(detections) > 0:
                    self._process_and_publish_positions(
                        result, detections, angle_map, publish_mqtt, publish_http
                    )

                # ---------------------------------------------------------
                # Step 7: Calculate FPS
                # ---------------------------------------------------------
                fps = self._update_fps(frame_times, loop_start)

                # ---------------------------------------------------------
                # Step 8: Display results → visualization.py
                # ---------------------------------------------------------
                if self.cli_mode:
                    self._print_cli_status(detections, result, fps)
                else:
                    display = self._draw_gui_frame(result, detections, cropped_detections, fps)

                # ---------------------------------------------------------
                # Log timing (optional)
                # ---------------------------------------------------------
                if self.args.log:
                    self._log_timing(result, loop_start)

                # ---------------------------------------------------------
                # Step 9: Handle keyboard input
                # ---------------------------------------------------------
                if not self.cli_mode:
                    if self._handle_keyboard_input(display):
                        break

        except KeyboardInterrupt:
            print("\n Interrupted by user...")

    # =========================================================================
    # IMAGE MODE - Single image inference
    # =========================================================================

    def _run_image(self):
        """
        Single image inference.

        Flow:
            1. Load image              → camera.py
            2. Run pipeline            → pipeline.py
            3. Save labels (optional)  → utils/label_io.py
            4. Process keypoints       → utils/keypoint_processor.py
            5. Calculate positions     → Positioning/
            6. Publish MQTT/HTTP       → utils/network_publish.py
            7. Visualize & save        → visualization.py
            8. Display result (GUI)
        """
        from camera import load_image
        from utils.label_io import save_detections_to_txt
        from utils.network_publish import configure_mqtt, publish_mqtt, publish_http
        from visualization import Visualizer

        # -----------------------------------------------------------------
        # Step 1: Load image → camera.py
        # -----------------------------------------------------------------
        frame = load_image(self.args.image)
        if frame is None:
            return

        # Configure MQTT
        if self.args.mqtt:
            configure_mqtt()

        # -----------------------------------------------------------------
        # Step 2: Run inference pipeline → pipeline.py
        # -----------------------------------------------------------------
        print(f"\n Running inference ({self.args.mode} mode)...")
        result = self.pipeline.run(frame)
        detections = result.detections
        cropped_detections = result.cropped_detections

        print(f" Inference completed in {result.inference_time:.2f}s")
        print(f"\n Detected {len(detections)} objects")

        if self.args.log:
            self._log_timing(result)

        # -----------------------------------------------------------------
        # Step 3: Save labels (optional) → utils/label_io.py
        # -----------------------------------------------------------------
        if self.args.save_labels and len(detections) > 0:
            self._save_image_labels(result, detections, save_detections_to_txt)

        # -----------------------------------------------------------------
        # Step 4: Process keypoints → utils/keypoint_processor.py
        # -----------------------------------------------------------------
        angle_map = {}
        if self.keypoint_processor and len(detections) > 0:
            angle_map = self.keypoint_processor.process(
                detections, result.frame_resized
            )

        # -----------------------------------------------------------------
        # Step 5-6: Calculate positions & publish → Positioning/, network_publish.py
        # -----------------------------------------------------------------
        if (self.args.positioning or self.args.web or self.args.mqtt) and len(detections) > 0:
            self._process_and_publish_positions(
                result, detections, angle_map, publish_mqtt, publish_http,
                is_image_mode=True
            )

        # -----------------------------------------------------------------
        # Step 7: Visualize & save → visualization.py
        # -----------------------------------------------------------------
        visualizer = Visualizer()
        display = self._create_visualization(
            visualizer, result, detections, cropped_detections
        )

        output_path = self._save_result_image(display)
        print(f" Result saved to: {output_path}")

        # -----------------------------------------------------------------
        # Step 8: Display result (GUI mode only)
        # -----------------------------------------------------------------
        if not self.cli_mode:
            self._show_image_result(display)
        else:
            print(" CLI mode: preview skipped")

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cleanup(self):
        """
        Release all resources.

        Cleans up:
            - camera.py        → camera.release()
            - visualization.py → visualizer.destroy()
            - keypoint_processor.py → cleanup()
        """
        if self.camera:
            self.camera.release()
        if self.visualizer:
            self.visualizer.destroy()
        if self.keypoint_processor:
            self.keypoint_processor.cleanup()
        print(" Done!")

    # =========================================================================
    # HELPER METHODS - Camera Loop
    # =========================================================================

    def _print_controls(self):
        """Print keyboard controls."""
        print("\n Starting detection loop...")
        if self.cli_mode:
            print("   CLI mode: Press Ctrl+C to quit")
        else:
            print("   Press 'Q' to quit")
            print("   Press 'S' to save screenshot")
            print("   Press 'M' to toggle masks")
            print("   Press '+'/'-' to adjust confidence")
        print()

    def _save_frame_labels(self, result, detections, frame_counter, save_func):
        """Save detection labels for a camera frame."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"frame_{timestamp}_{frame_counter:06d}"

        label_path = os.path.join(self.dirs['labels'], f"{base_name}.txt")
        save_func(detections, label_path, result.original_height, result.original_width)

        image_path = os.path.join(self.dirs['images'], f"{base_name}.jpg")
        cv2.imwrite(image_path, result.frame_resized)

        return frame_counter + 1

    def _update_fps(self, frame_times, loop_start):
        """Update and return FPS."""
        frame_times.append(time.time() - loop_start)
        if len(frame_times) > 10:
            frame_times.pop(0)
        return 1.0 / (sum(frame_times) / len(frame_times))

    def _print_cli_status(self, detections, result, fps):
        """Print CLI status line."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Detected {len(detections)} objects | "
              f"Inference: {int(result.inference_time*1000)}ms | FPS: {fps:.1f}")

    def _draw_gui_frame(self, result, detections, cropped_detections, fps):
        """Draw GUI frame with detections."""
        if self.pipeline.roi and self.pipeline.roi.is_complete():
            annotated = self.visualizer.draw_detections(
                result.frame_display, cropped_detections, show_masks=self.show_masks
            )
        else:
            annotated = self.visualizer.draw_detections(
                result.frame_resized, detections, show_masks=self.show_masks
            )

        display = self.visualizer.draw_summary(
            annotated, detections, fps, self.args.mode, result.inference_time
        )
        self.visualizer.show(display)
        return display

    def _handle_keyboard_input(self, display) -> bool:
        """Handle keyboard input. Returns True if should quit."""
        key = self.visualizer.handle_input(DISPLAY_KEY_WAIT_MS)

        if key.lower() == 'q':
            print("\n Quitting...")
            return True
        elif key.lower() == 's':
            self.visualizer.save_screenshot(display)
        elif key.lower() == 'm':
            self.show_masks = not self.show_masks
            print(f" Masks: {'ON' if self.show_masks else 'OFF'}")
        elif key in ['+', '=']:
            self.confidence = min(CONFIDENCE_MAX, self.confidence + CONFIDENCE_INCREMENT)
            self.pipeline.update_confidence(self.confidence)
            print(f" Confidence: {self.confidence:.2f}")
        elif key in ['-', '_']:
            self.confidence = max(CONFIDENCE_MIN, self.confidence - CONFIDENCE_INCREMENT)
            self.pipeline.update_confidence(self.confidence)
            print(f" Confidence: {self.confidence:.2f}")

        return False

    # =========================================================================
    # HELPER METHODS - Image Mode
    # =========================================================================

    def _save_image_labels(self, result, detections, save_func):
        """Save detection labels for an image."""
        base_name = os.path.splitext(os.path.basename(self.args.image))[0]

        label_path = os.path.join(self.dirs['labels'], f"{base_name}.txt")
        save_func(detections, label_path, result.original_height, result.original_width)
        print(f" Labels saved to: {label_path}")

        raw_image_path = os.path.join(self.dirs['images'], f"{base_name}.jpg")
        cv2.imwrite(raw_image_path, result.frame_resized)
        print(f" Raw image saved to: {raw_image_path}")

    def _create_visualization(self, visualizer, result, detections, cropped_detections):
        """Create annotated visualization."""
        if self.pipeline.roi and self.pipeline.roi.is_complete():
            annotated = visualizer.draw_detections(
                result.frame_display, cropped_detections, show_masks=True
            )
        else:
            annotated = visualizer.draw_detections(
                result.frame_resized, detections, show_masks=True
            )
        return visualizer.draw_summary(
            annotated, detections, 0, self.args.mode, result.inference_time
        )

    def _save_result_image(self, display):
        """Save result image and return path."""
        output_path = self.args.output
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.args.image))[0]
            output_path = f"{base_name}_detection.jpg"
        cv2.imwrite(output_path, display)
        return output_path

    def _show_image_result(self, display):
        """Display image result in window."""
        print("\n Displaying result... Press any key to close")
        h, w = display.shape[:2]
        cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
        screen_scale = min(1.0, 1400 / w, 900 / h)
        cv2.resizeWindow("Detection Result", int(w * screen_scale), int(h * screen_scale))
        cv2.imshow("Detection Result", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # =========================================================================
    # HELPER METHODS - Shared
    # =========================================================================

    def _process_and_publish_positions(self, result, detections, angle_map,
                                        publish_mqtt, publish_http, is_image_mode=False):
        """Calculate positions and publish via MQTT/HTTP."""
        from Positioning import calculate_positions_from_detections

        positions = calculate_positions_from_detections(
            detections,
            image_width=result.original_width,
            image_height=result.original_height,
            enable_collision_resolution=True,
            angle_map=angle_map,
        )

        if is_image_mode and positions:
            print(f" Calculated {len(positions)} positions")

        if self.args.mqtt and positions:
            log_name = None
            if is_image_mode:
                base_name = os.path.splitext(os.path.basename(self.args.image))[0]
                log_name = f"{base_name}.txt"
            pub_result = publish_mqtt(
                positions,
                log_dir=self.dirs.get('mqtt_logs', MQTT_LOGS_DIR),
                log_name=log_name
            )
            print(f"[MQTT] {pub_result.message}")

        if self.args.web and positions:
            pub_result = publish_http(positions)
            print(f"[WEB] {pub_result.message}")

    def _log_timing(self, result, loop_start=None):
        """Log timing breakdown."""
        if loop_start:
            result.timings['total'] = time.time() - loop_start
        timing_str = " | ".join(f"{k}: {int(v*1000)}ms" for k, v in result.timings.items())
        print(f"[LOG] {timing_str}")
