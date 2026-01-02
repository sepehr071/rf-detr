"""
Preprocessing Module
====================
Frame preprocessing pipeline for RF-DETR inference.

This module consolidates the preprocessing steps that were duplicated
between run_detection_loop() and run_image_inference() in main.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import numpy as np
import cv2


@dataclass
class PreprocessedFrame:
    """
    Container for preprocessed frame data.

    Holds all the information needed to map detections back to
    original coordinate space after inference.
    """
    frame_rgb: np.ndarray               # RGB frame for model input
    frame_masked: np.ndarray            # BGR masked frame (for tracker/visualization)
    frame_resized: np.ndarray           # Full resized frame (for saving raw image)
    original_height: int                # Height of resized frame (before crop)
    original_width: int                 # Width of resized frame (before crop)
    x_offset: int                       # ROI crop X offset
    y_offset: int                       # ROI crop Y offset
    scale: float                        # Resize scale factor
    scaled_roi: Optional[object] = None # Scaled ROI manager (if applicable)
    timings: Dict[str, float] = field(default_factory=dict)  # Timing breakdown


def preprocess_frame(
    frame: np.ndarray,
    roi: Optional["ROIManager"] = None,
    max_size: int = 1280,
    collect_timings: bool = False
) -> PreprocessedFrame:
    """
    Apply full preprocessing pipeline to a frame.

    Pipeline: resize -> crop to ROI -> apply polygon mask -> convert to RGB

    This function consolidates the preprocessing logic that was duplicated
    in main.py between run_detection_loop() (lines 311-344) and
    run_image_inference() (lines 610-641).

    Args:
        frame: Input BGR frame from camera/file
        roi: ROIManager instance (optional)
        max_size: Maximum dimension for resize (default: 1280)
        collect_timings: Whether to collect timing information

    Returns:
        PreprocessedFrame with all preprocessing results
    """
    import time
    from model import resize_for_inference

    timings = {}

    # === RESIZE IF NEEDED ===
    t0 = time.time() if collect_timings else 0
    frame_resized, scale = resize_for_inference(frame, max_size)
    original_h, original_w = frame_resized.shape[:2]
    if collect_timings:
        timings['resize'] = time.time() - t0

    # === CROP TO ROI (more efficient than masking) ===
    t0 = time.time() if collect_timings else 0
    x_offset, y_offset = 0, 0
    scaled_roi = None

    if roi and roi.is_complete():
        # Scale ROI if image was resized
        if scale != 1.0:
            scaled_roi = roi.get_scaled(scale)
        else:
            scaled_roi = roi

        # Crop to ROI bounding box
        frame_cropped, x_offset, y_offset = scaled_roi.crop_to_roi(frame_resized)
        if collect_timings:
            timings['crop'] = time.time() - t0

        # Apply polygon mask within cropped area (for non-rectangular ROI edges)
        t0 = time.time() if collect_timings else 0
        frame_masked = scaled_roi.apply_mask_to_cropped(frame_cropped, x_offset, y_offset)
        if collect_timings:
            timings['mask'] = time.time() - t0
    else:
        frame_masked = frame_resized
        if collect_timings:
            timings['crop'] = time.time() - t0
            timings['mask'] = 0

    # === CONVERT TO RGB FOR MODEL ===
    t0 = time.time() if collect_timings else 0
    frame_rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
    if collect_timings:
        timings['convert'] = time.time() - t0

    return PreprocessedFrame(
        frame_rgb=frame_rgb,
        frame_masked=frame_masked,
        frame_resized=frame_resized,
        original_height=original_h,
        original_width=original_w,
        x_offset=x_offset,
        y_offset=y_offset,
        scale=scale,
        scaled_roi=scaled_roi,
        timings=timings
    )
