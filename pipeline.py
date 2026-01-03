"""
Inference Pipeline Module
=========================
Unified inference pipeline for both camera and image modes.

This module consolidates the inference logic that was duplicated between
run_detection_loop() and run_image_inference() in main.py.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

from config import (
    DEFAULT_CHECKPOINT,
    ROI_CONFIG_FILE,
    POST_NMS_THRESHOLD,
    POST_NMS_CLASS_AGNOSTIC,
    MAX_IMAGE_SIZE,
    VERBOSE_TILES_DIR,
    OPENVINO_ENABLED,
)


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline."""
    mode: str = "sahi"                    # full, sahi, hybrid
    confidence: float = 0.5
    nms_enabled: bool = False
    track_enabled: bool = False
    verbose: bool = False
    tiles_dir: Optional[str] = None       # Directory for saving SAHI tiles


@dataclass
class InferenceResult:
    """Result of a single inference pass."""
    detections: object                    # supervision.Detections (original coords)
    cropped_detections: object            # Detections in cropped coords (for viz)
    frame_display: np.ndarray             # BGR frame for visualization (cropped/masked)
    frame_resized: np.ndarray             # Full resized frame (for saving raw image)
    original_height: int                  # Height of resized frame
    original_width: int                   # Width of resized frame
    inference_time: float                 # Inference duration in seconds
    timings: Dict[str, float] = field(default_factory=dict)  # Detailed timing breakdown


class InferencePipeline:
    """
    Unified inference pipeline for RF-DETR detection.

    Handles the complete flow:
    preprocess -> inference -> NMS -> tracking -> coordinate offset -> ROI filter -> add IDs

    This class unifies the logic that was duplicated between
    run_detection_loop() and run_image_inference() in main.py.
    """

    def __init__(
        self,
        model: "DetectionModel",
        roi: Optional["ROIManager"] = None,
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize inference pipeline.

        Args:
            model: Loaded DetectionModel instance
            roi: ROIManager instance (optional)
            config: Pipeline configuration
        """
        self.model = model
        self.roi = roi
        self.config = config or PipelineConfig()
        self.tracker = None

        if self.config.track_enabled:
            from tracker import ObjectTracker
            self.tracker = ObjectTracker()

    def run(self, frame: np.ndarray) -> InferenceResult:
        """
        Run complete inference pipeline on a frame.

        Pipeline stages:
        1. Preprocess (resize, crop, mask, RGB convert)
        2. Inference (full/sahi/hybrid based on config.mode)
        3. Post-process (NMS if enabled)
        4. Track (if enabled)
        5. Offset coordinates back to original space
        6. Filter by ROI polygon
        7. Add bbox IDs

        Args:
            frame: Input BGR frame

        Returns:
            InferenceResult with detections and metadata
        """
        from utils.preprocessing import preprocess_frame
        from utils.id_helper import add_bbox_ids_to_detections
        from model import offset_detections

        timings = {}

        # === PREPROCESS ===
        t0 = time.time()
        preprocessed = preprocess_frame(
            frame,
            roi=self.roi,
            max_size=MAX_IMAGE_SIZE,
            collect_timings=True
        )
        timings.update(preprocessed.timings)

        # === RUN INFERENCE ===
        t0 = time.time()
        detections = self._run_inference(preprocessed.frame_rgb)
        inference_time = time.time() - t0
        timings['inference'] = inference_time

        # === APPLY NMS (if enabled) ===
        if self.config.nms_enabled:
            t0 = time.time()
            detections = detections.with_nms(
                threshold=POST_NMS_THRESHOLD,
                class_agnostic=POST_NMS_CLASS_AGNOSTIC
            )
            timings['nms'] = time.time() - t0

        # === UPDATE TRACKER (if enabled) ===
        if self.tracker is not None:
            t0 = time.time()
            # Pass BGR frame for tracking (frame_masked is before RGB conversion)
            detections = self.tracker.update(detections, preprocessed.frame_masked)
            timings['tracking'] = time.time() - t0

        # === KEEP CROPPED DETECTIONS FOR VISUALIZATION ===
        # Detections are currently in cropped coordinate space
        cropped_detections = detections

        # === OFFSET DETECTIONS BACK TO ORIGINAL COORDINATES ===
        t0 = time.time()
        x_offset = preprocessed.x_offset
        y_offset = preprocessed.y_offset
        if x_offset != 0 or y_offset != 0:
            detections = offset_detections(
                detections,
                x_offset,
                y_offset,
                preprocessed.original_height,
                preprocessed.original_width
            )
        timings['offset'] = time.time() - t0

        # === FILTER DETECTIONS BY ROI POLYGON ===
        t0 = time.time()
        scaled_roi = preprocessed.scaled_roi
        if scaled_roi and scaled_roi.is_complete():
            detections = scaled_roi.filter_detections(detections)
        timings['filter'] = time.time() - t0

        # === ADD BBOX IDS ===
        detections = add_bbox_ids_to_detections(detections)

        return InferenceResult(
            detections=detections,
            cropped_detections=cropped_detections,
            frame_display=preprocessed.frame_masked,
            frame_resized=preprocessed.frame_resized,
            original_height=preprocessed.original_height,
            original_width=preprocessed.original_width,
            inference_time=inference_time,
            timings=timings
        )

    def _run_inference(self, frame_rgb: np.ndarray) -> object:
        """
        Run model inference based on configured mode.

        Args:
            frame_rgb: RGB frame for model

        Returns:
            supervision.Detections object
        """
        mode = self.config.mode
        confidence = self.config.confidence
        verbose = self.config.verbose
        tiles_dir = self.config.tiles_dir

        if mode == "hybrid":
            return self.model.predict_hybrid(
                frame_rgb,
                confidence,
                verbose=verbose,
                tiles_dir=tiles_dir
            )
        elif mode == "sahi":
            return self.model.predict_sahi(
                frame_rgb,
                confidence,
                verbose=verbose,
                tiles_dir=tiles_dir
            )
        else:
            return self.model.predict(frame_rgb, confidence)

    def reset_tracker(self):
        """Reset tracker state (e.g., when switching videos)."""
        if self.tracker:
            self.tracker.reset()

    def update_confidence(self, confidence: float):
        """Update confidence threshold."""
        self.config.confidence = confidence


def create_pipeline(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    use_openvino: bool = OPENVINO_ENABLED,  # Default: True (from config)
    roi_config_file: str = ROI_CONFIG_FILE,
    load_roi: bool = True,
    config: Optional[PipelineConfig] = None
) -> InferencePipeline:
    """
    Factory function to create a fully configured pipeline.

    OpenVINO optimization is enabled by default for better performance.

    Args:
        checkpoint_path: Path to model checkpoint
        use_openvino: Whether to use OpenVINO optimization (default: True)
        roi_config_file: Path to ROI config file
        load_roi: Whether to load ROI (set False to skip)
        config: Pipeline configuration

    Returns:
        Configured InferencePipeline instance
    """
    from model import DetectionModel
    from roi import ROIManager

    # Load model (OpenVINO enabled by default)
    model = DetectionModel(checkpoint_path, use_openvino)

    # Load ROI
    roi = None
    if load_roi:
        roi = ROIManager(roi_config_file)
        if not roi.load():
            roi = None

    return InferencePipeline(model=model, roi=roi, config=config)
