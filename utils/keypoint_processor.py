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
"""

import os
import math
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import (
    KEYPOINT_MODEL_PATH,
    KEYPOINT_CROP_MARGIN,
    KEYPOINT_TEMP_DIR,
    KEYPOINT_BIG_OBJECT_CLASSES,
    KEYPOINT_MIN_CONFIDENCE,
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
    """

    def __init__(
        self,
        model_path: str = KEYPOINT_MODEL_PATH,
        margin: int = KEYPOINT_CROP_MARGIN,
        temp_dir: str = KEYPOINT_TEMP_DIR,
        big_object_classes: set = None,
        min_confidence: float = KEYPOINT_MIN_CONFIDENCE,
    ):
        """
        Initialize the keypoint processor.

        Args:
            model_path: Path to YOLO pose model checkpoint
            margin: Pixels of margin around bbox when cropping
            temp_dir: Directory for temporary crop files
            big_object_classes: Set of class IDs to process (default: {1, 4})
            min_confidence: Minimum keypoint confidence threshold
        """
        self.model_path = model_path
        self.margin = margin
        self.temp_dir = temp_dir
        self.big_object_classes = big_object_classes or KEYPOINT_BIG_OBJECT_CLASSES
        self.min_confidence = min_confidence
        self._model = None  # Lazy-loaded

    def _get_model(self):
        """Lazy-load the YOLO pose model."""
        if self._model is None:
            if not os.path.exists(self.model_path):
                print(f"[KEYPOINT] Warning: Model not found at {self.model_path}")
                return None
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                print(f"[KEYPOINT] Loaded pose model from {self.model_path}")
            except Exception as e:
                print(f"[KEYPOINT] Error loading model: {e}")
                return None
        return self._model

    def process(
        self,
        detections,
        frame: np.ndarray,
        calibrator=None,
    ) -> Dict[str, float]:
        """
        Process detections and return angle map for big objects.

        Args:
            detections: supervision.Detections with xyxy, class_id
            frame: BGR frame (in original/resized coordinates)
            calibrator: Optional FisheyeCalibrator for world coord angle calculation

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

        if not big_objects:
            return angle_map

        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

        # Process each big object
        for idx, bbox, class_id in big_objects:
            # Generate bbox-based ID
            bbox_id = self._generate_bbox_id(bbox)

            # Crop with margin
            cropped, offset = self._crop_with_margin(frame, bbox)

            if cropped is None or cropped.size == 0:
                continue

            # Run keypoint inference
            kp_result = self._run_keypoint_inference(cropped)

            if kp_result and len(kp_result.get('keypoints', [])) >= 4:
                # Offset keypoints back to original coordinates
                keypoints = self._offset_keypoints(kp_result['keypoints'], offset)

                # Calculate angle
                angle = self._calculate_angle_from_keypoints(keypoints, calibrator)
                angle_map[bbox_id] = angle

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
        Run YOLO pose inference using file-based approach.

        Args:
            cropped: BGR cropped image

        Returns:
            Dict with 'keypoints' and 'keypoint_confidences', or None
        """
        model = self._get_model()
        if model is None:
            return None

        # Save to temp file
        temp_path = os.path.join(self.temp_dir, f"crop_{time.time_ns()}.jpg")

        try:
            cv2.imwrite(temp_path, cropped)

            # Run inference
            results = model(temp_path, verbose=False)

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
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

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
        """Clean up temporary files and resources."""
        if os.path.exists(self.temp_dir):
            try:
                for f in os.listdir(self.temp_dir):
                    if f.startswith("crop_") and f.endswith(".jpg"):
                        os.remove(os.path.join(self.temp_dir, f))
            except Exception:
                pass
