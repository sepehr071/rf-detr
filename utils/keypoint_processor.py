"""
Keypoint Processor Module
=========================
Handles keypoint detection and angle calculation for "big objects" (multipacks).

Flow:
1. Filter detections for big objects (class_id 1, 4)
2. Crop each detection with margin
3. Run YOLO pose model to detect 4-corner keypoints
4. Calculate rotation angle from keypoints
5. Return angle_map: {bbox_id: angle}

Performance optimizations:
- Direct array inference (no file I/O)
- OpenVINO auto-export/load for faster inference
"""

import os
import math
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import (
    KEYPOINT_MODEL_PATH,
    KEYPOINT_OPENVINO_PATH,
    KEYPOINT_CROP_MARGIN,
    KEYPOINT_BIG_OBJECT_CLASSES,
    KEYPOINT_MIN_CONFIDENCE,
    OPENVINO_ENABLED,
)


@dataclass
class KeypointResult:
    """Result from keypoint detection for a single object."""
    bbox_id: str
    keypoints: List[List[float]]  # 4 corner points [[x,y], ...]
    keypoint_confidences: List[float]
    angle: float  # Calculated rotation angle (0-359)


class KeypointProcessor:
    """
    Processes detections to extract keypoints and calculate rotation angles
    for big objects (multipacks).
    
    OpenVINO optimization:
    - If OpenVINO model exists, loads it directly
    - If not, exports PyTorch model to OpenVINO format first
    - This happens automatically on first load
    """

    def __init__(
        self,
        model_path: str = KEYPOINT_MODEL_PATH,
        openvino_path: str = KEYPOINT_OPENVINO_PATH,
        margin: int = KEYPOINT_CROP_MARGIN,
        big_object_classes: set = None,
        min_confidence: float = KEYPOINT_MIN_CONFIDENCE,
        use_openvino: bool = OPENVINO_ENABLED,
        save_debug_crops: bool = False,
    ):
        """
        Initialize the keypoint processor.

        Args:
            model_path: Path to YOLO pose model checkpoint (.pt)
            openvino_path: Path for OpenVINO model (auto-generated)
            margin: Pixels of margin around bbox when cropping
            big_object_classes: Set of class IDs to process (default: {1, 4})
            min_confidence: Minimum keypoint confidence threshold
            use_openvino: Use OpenVINO for faster inference (default: True)
            save_debug_crops: Save cropped images to debug_crops/ for debugging
        """
        self.model_path = model_path
        self.openvino_path = openvino_path
        self.margin = margin
        self.big_object_classes = big_object_classes or KEYPOINT_BIG_OBJECT_CLASSES
        self.min_confidence = min_confidence
        self.use_openvino = use_openvino
        self.save_debug_crops = save_debug_crops
        self._model = None  # Lazy-loaded
        self._crop_counter = 0
        self._using_openvino = False

    def _get_model(self):
        """
        Lazy-load the YOLO pose model with OpenVINO optimization.
        
        Logic:
        1. If OpenVINO enabled and OpenVINO model exists -> load it
        2. If OpenVINO enabled but no OpenVINO model -> export first, then load
        3. If OpenVINO disabled -> load PyTorch model directly
        """
        if self._model is not None:
            return self._model

        if not os.path.exists(self.model_path):
            print(f"[KEYPOINT] Warning: Model not found at {self.model_path}")
            return None

        try:
            from ultralytics import YOLO

            # Check if we should use OpenVINO
            if self.use_openvino:
                openvino_model_dir = self.openvino_path
                
                # Check if OpenVINO model already exists
                if os.path.exists(openvino_model_dir) and os.path.isdir(openvino_model_dir):
                    # OpenVINO model exists, load it directly
                    print(f"[KEYPOINT] Loading OpenVINO model from {openvino_model_dir}")
                    self._model = YOLO(openvino_model_dir)
                    self._using_openvino = True
                    print(f"[KEYPOINT] ✅ OpenVINO model loaded successfully")
                else:
                    # OpenVINO model doesn't exist, export first
                    print(f"[KEYPOINT] OpenVINO model not found, exporting...")
                    self._model = self._export_and_load_openvino()
            else:
                # Load PyTorch model directly
                print(f"[KEYPOINT] Loading PyTorch model from {self.model_path}")
                self._model = YOLO(self.model_path)
                self._using_openvino = False
                print(f"[KEYPOINT] ✅ PyTorch model loaded")

        except Exception as e:
            print(f"[KEYPOINT] Error loading model: {e}")
            return None

        return self._model

    def _export_and_load_openvino(self):
        """
        Export PyTorch model to OpenVINO format and load it.
        
        Returns:
            YOLO model loaded from OpenVINO format
        """
        from ultralytics import YOLO

        try:
            # Load PyTorch model first
            print(f"[KEYPOINT] Loading PyTorch model for export: {self.model_path}")
            pt_model = YOLO(self.model_path)

            # Export to OpenVINO format
            print(f"[KEYPOINT] Exporting to OpenVINO format...")
            export_path = pt_model.export(format='openvino')
            
            print(f"[KEYPOINT] ✅ Exported to: {export_path}")
            
            # Load the exported OpenVINO model
            self._using_openvino = True
            return YOLO(export_path)

        except Exception as e:
            print(f"[KEYPOINT] ⚠️ OpenVINO export failed: {e}")
            print(f"[KEYPOINT] Falling back to PyTorch model")
            self._using_openvino = False
            return YOLO(self.model_path)

    def process(
        self,
        detections,
        frame: np.ndarray,
        calibrator=None,
        debug: bool = True,
    ) -> Dict[str, float]:
        """
        Process detections and return angle map for big objects.

        Args:
            detections: supervision.Detections with xyxy, class_id
            frame: BGR frame (in original/resized coordinates)
            calibrator: Optional FisheyeCalibrator for world coord angle calculation
            debug: Print debug info

        Returns:
            Dict mapping bbox_id -> angle (0-359 degrees)
        """
        angle_map = {}

        if len(detections) == 0:
            return angle_map

        # Check if model is available
        model = self._get_model()
        if model is None:
            return angle_map

        # Filter for big objects
        big_objects = self._filter_big_objects(detections)

        if debug and big_objects:
            openvino_status = "OpenVINO" if self._using_openvino else "PyTorch"
            print(f"[KEYPOINT] Found {len(big_objects)} big objects (class_id in {self.big_object_classes}) [{openvino_status}]")

        if not big_objects:
            return angle_map

        # Process each big object
        for idx, bbox, class_id in big_objects:
            # Generate bbox-based ID
            bbox_id = self._generate_bbox_id(bbox)

            # Crop with margin
            cropped, offset = self._crop_with_margin(frame, bbox)

            if cropped is None or cropped.size == 0:
                if debug:
                    print(f"[KEYPOINT] Crop failed for class={class_id} bbox={bbox_id}")
                continue

            # Save debug crops if enabled
            if self.save_debug_crops:
                debug_dir = "debug_crops"
                os.makedirs(debug_dir, exist_ok=True)
                self._crop_counter += 1
                debug_path = os.path.join(debug_dir, f"crop_{self._crop_counter:04d}_class{class_id}.jpg")
                cv2.imwrite(debug_path, cropped)
                print(f"[KEYPOINT] Saved debug crop: {debug_path} ({cropped.shape[1]}x{cropped.shape[0]})")

            # Run keypoint inference (direct array - no file I/O!)
            kp_result = self._run_keypoint_inference(cropped)

            if kp_result and len(kp_result.get('keypoints', [])) >= 4:
                # Offset keypoints back to original coordinates
                keypoints = self._offset_keypoints(kp_result['keypoints'], offset)

                # Calculate angle
                angle = self._calculate_angle_from_keypoints(keypoints, calibrator)
                angle_map[bbox_id] = angle

                if debug:
                    print(f"[KEYPOINT] class={class_id} angle={angle}° keypoints={len(keypoints)}")
            else:
                if debug:
                    kp_count = len(kp_result.get('keypoints', [])) if kp_result else 0
                    print(f"[KEYPOINT] No keypoints for class={class_id} (got {kp_count}, need 4)")

        return angle_map

    def _filter_big_objects(self, detections) -> List[Tuple[int, np.ndarray, int]]:
        """
        Filter detections for big objects (class_id in big_object_classes).

        Returns:
            List of (index, bbox, class_id) tuples
        """
        big_objects = []

        for idx, (bbox, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
            if int(class_id) in self.big_object_classes:
                big_objects.append((idx, bbox, int(class_id)))

        return big_objects

    def _crop_with_margin(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        Crop detection with margin, handling image boundaries.

        Args:
            frame: BGR image
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            Tuple of (cropped_image, (x_offset, y_offset))
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add margin, clamping to image boundaries
        x1_margin = max(0, int(x1) - self.margin)
        y1_margin = max(0, int(y1) - self.margin)
        x2_margin = min(w, int(x2) + self.margin)
        y2_margin = min(h, int(y2) + self.margin)

        # Crop
        cropped = frame[y1_margin:y2_margin, x1_margin:x2_margin]

        if cropped.size == 0:
            return None, (0, 0)

        return cropped, (x1_margin, y1_margin)

    def _run_keypoint_inference(self, cropped: np.ndarray) -> Optional[dict]:
        """
        Run YOLO pose inference using direct array input (no file I/O).
        
        This is much faster than the previous file-based approach.

        Args:
            cropped: BGR cropped image (numpy array)

        Returns:
            Dict with 'keypoints' and 'keypoint_confidences', or None
        """
        model = self._get_model()
        if model is None:
            return None

        try:
            # Direct array inference - no file I/O needed!
            # YOLO accepts numpy arrays directly
            results = model(cropped, verbose=False)

            # Parse results
            if results and len(results) > 0:
                result = results[0]
                if result.keypoints is not None and len(result.keypoints) > 0:
                    kpts_xy = result.keypoints.xy.cpu().numpy()
                    kpts_conf = result.keypoints.conf.cpu().numpy()

                    if len(kpts_xy) > 0:
                        return {
                            'keypoints': kpts_xy[0].tolist(),
                            'keypoint_confidences': kpts_conf[0].tolist(),
                        }
        except Exception as e:
            print(f"[KEYPOINT] Inference error: {e}")

        return None

    def _offset_keypoints(
        self,
        keypoints: List[List[float]],
        offset: Tuple[int, int],
    ) -> List[List[float]]:
        """
        Offset keypoints back to original frame coordinates.

        Args:
            keypoints: List of [x, y] points in cropped coordinates
            offset: (x_offset, y_offset) from cropping

        Returns:
            List of [x, y] points in original coordinates
        """
        x_offset, y_offset = offset
        return [[kp[0] + x_offset, kp[1] + y_offset] for kp in keypoints]

    def _calculate_angle_from_keypoints(
        self,
        keypoints: List[List[float]],
        calibrator=None,
    ) -> float:
        """
        Calculate rotation angle from 4-corner keypoints.

        Keypoint order (from checkpoints/module.py):
        - [0] front_left
        - [1] front_right
        - [2] back_left
        - [3] back_right

        Uses front edge (keypoints[0] -> keypoints[1]) for angle calculation.

        Args:
            keypoints: List of 4 [x, y] corner points
            calibrator: Optional FisheyeCalibrator for world coord transform

        Returns:
            Angle in degrees (0-359), normalized
        """
        if len(keypoints) < 2:
            return 0.0

        front_left = keypoints[0]
        front_right = keypoints[1]

        # Calculate angle using world coordinates if calibrator available
        if calibrator is not None:
            try:
                world_fl = calibrator.pixel_to_world(front_left[0], front_left[1])
                world_fr = calibrator.pixel_to_world(front_right[0], front_right[1])

                if world_fl and world_fr:
                    dx = world_fr[0] - world_fl[0]
                    dy = world_fr[1] - world_fl[1]
                else:
                    # Fallback to pixel coordinates
                    dx = front_right[0] - front_left[0]
                    dy = front_right[1] - front_left[1]
            except Exception:
                dx = front_right[0] - front_left[0]
                dy = front_right[1] - front_left[1]
        else:
            # Use pixel coordinates
            dx = front_right[0] - front_left[0]
            dy = front_right[1] - front_left[1]

        # Calculate angle
        angle = math.degrees(math.atan2(dy, dx))

        # Normalize to 0-359 (clockwise from right)
        # Adjust by -180 to match position_angle_calculator.py convention
        angle = (angle - 180) % 360

        return round(angle)

    def _generate_bbox_id(self, bbox: np.ndarray) -> str:
        """
        Generate bbox-based ID matching id_helper.py format.

        Args:
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            String ID in format "x1_y1_x2_y2"
        """
        x1, y1, x2, y2 = bbox
        return f"{int(round(x1))}_{int(round(y1))}_{int(round(x2))}_{int(round(y2))}"

    def cleanup(self):
        """Clean up resources."""
        # No temp files to clean up anymore (direct array inference)
        pass

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'openvino_path': self.openvino_path,
            'using_openvino': self._using_openvino,
            'model_loaded': self._model is not None,
        }
