"""
Model Module
============
RF-DETR model loading and inference.
"""

import os
import numpy as np
import cv2
from typing import Optional

import time

from config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE,
    SAHI_TILES_X,
    SAHI_TILES_Y,
    SAHI_OVERLAP,
    MAX_IMAGE_SIZE,
    VERBOSE_TILES_DIR,
    NMS_IOU_THRESHOLD,
    MERGE_IOU_THRESHOLD,
    MERGE_CLASS_AGNOSTIC,
    MERGE_STRATEGY,
    CLASS_REMAP,
    FILTER_OTHER_CLASS
)
from nms import class_aware_nms, non_max_merge


class DetectionModel:
    """
    RF-DETR-Seg model wrapper for inference.
    
    Supports both full-frame and SAHI tiled inference modes.
    """
    
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        use_openvino: bool = False
    ):
        """
        Initialize and load the detection model.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            use_openvino: Whether to apply OpenVINO optimization
        """
        self.checkpoint_path = checkpoint_path
        self.use_openvino = use_openvino
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the RF-DETR-Seg model."""
        from rfdetr import RFDETRSegPreview
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        print(f"Loading model from: {self.checkpoint_path}")
        self.model = RFDETRSegPreview(pretrain_weights=self.checkpoint_path)
        
        if self.use_openvino:
            self._apply_openvino()
    
    def _apply_openvino(self):
        """Apply OpenVINO optimization."""
        print("Optimizing model for Intel CPU with OpenVINO...")
        try:
            self.model.optimize_for_inference()
            print("✅ OpenVINO optimization applied")
        except Exception as e:
            print(f"⚠️ OpenVINO optimization failed: {e}")
            print("Continuing without optimization...")
    
    def predict(
        self,
        image: np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE
    ):
        """
        Run inference on a single image (full frame).

        Args:
            image: Input image (RGB format)
            confidence: Confidence threshold (0-1)

        Returns:
            supervision Detections object with remapped class IDs
        """
        detections = self.model.predict(image, threshold=confidence)
        return remap_class_ids(detections)
    
    def predict_sahi(
        self,
        image: np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE,
        tiles_x: int = SAHI_TILES_X,
        tiles_y: int = SAHI_TILES_Y,
        overlap: float = SAHI_OVERLAP,
        verbose: bool = False,
        tiles_dir: str = None
    ):
        """
        Run SAHI-style tiled inference for better small object detection.

        Splits image into overlapping tiles, runs inference on each,
        and merges results with class-aware NMS.

        Args:
            image: Input image (RGB format)
            confidence: Confidence threshold (0-1)
            tiles_x: Number of horizontal tiles
            tiles_y: Number of vertical tiles
            overlap: Overlap ratio between tiles (0-1)
            verbose: If True, save tile images to tiles_dir
            tiles_dir: Directory to save tile images (required if verbose=True)

        Returns:
            supervision Detections object with merged results
        """
        import supervision as sv

        h, w = image.shape[:2]

        # Calculate slice size to create exactly tiles_x × tiles_y tiles
        slice_w = int(w / (tiles_x - overlap * (tiles_x - 1)))
        slice_h = int(h / (tiles_y - overlap * (tiles_y - 1)))

        stride_x = int(slice_w * (1 - overlap))
        stride_y = int(slice_h * (1 - overlap))

        print(f"📐 SAHI: {tiles_x}x{tiles_y} tiles, size ~{slice_w}x{slice_h}, stride {stride_x}x{stride_y}")

        all_boxes = []
        all_masks = []
        all_classes = []
        all_confidences = []

        tile_count = 0
        row_idx = 0

        # Timestamp for verbose tile naming
        if verbose and tiles_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Generate tiles
        for y in range(0, h, stride_y):
            col_idx = 0
            for x in range(0, w, stride_x):
                # Calculate tile coordinates
                x1 = x
                y1 = y
                x2 = min(x + slice_w, w)
                y2 = min(y + slice_h, h)

                # Skip if tile is too small
                if (x2 - x1) < slice_w * 0.5 or (y2 - y1) < slice_h * 0.5:
                    col_idx += 1
                    continue

                tile_count += 1

                # Extract tile
                tile = image[y1:y2, x1:x2]

                # Save tile if verbose mode
                if verbose and tiles_dir:
                    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                    tile_path = os.path.join(tiles_dir, f"{timestamp}_tile_{row_idx}_{col_idx}.jpg")
                    cv2.imwrite(tile_path, tile_bgr)

                col_idx += 1

                # Run inference on tile
                detections = self.model.predict(tile, threshold=confidence)
                
                if len(detections) > 0:
                    # Offset boxes to full image coordinates
                    boxes = detections.xyxy.copy()
                    boxes[:, [0, 2]] += x1  # Offset x
                    boxes[:, [1, 3]] += y1  # Offset y
                    
                    # Clip to image bounds
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
                    
                    all_boxes.append(boxes)
                    all_classes.append(detections.class_id)
                    all_confidences.append(detections.confidence)
                    
                    # Handle masks
                    if detections.mask is not None:
                        for mask in detections.mask:
                            # Create full-size mask
                            full_mask = np.zeros((h, w), dtype=bool)
                            
                            # Get mask dimensions
                            mask_h, mask_w = mask.shape[:2]
                            
                            # Calculate where to place mask
                            place_h = min(mask_h, h - y1)
                            place_w = min(mask_w, w - x1)
                            
                            full_mask[y1:y1 + place_h, x1:x1 + place_w] = mask[:place_h, :place_w]
                            all_masks.append(full_mask)
            row_idx += 1

        print(f"📊 Processed {tile_count} tiles")
        
        # Combine all detections
        if len(all_boxes) == 0:
            return sv.Detections.empty()
        
        combined_boxes = np.vstack(all_boxes)
        combined_classes = np.concatenate(all_classes)
        combined_confidences = np.concatenate(all_confidences)
        combined_masks = np.array(all_masks) if all_masks else None
        
        # Apply class-aware NMS to remove duplicates
        final_indices = class_aware_nms(
            combined_boxes,
            combined_confidences,
            combined_classes
        )
        
        if len(final_indices) == 0:
            return sv.Detections.empty()
        
        # Filter to keep only NMS-selected detections
        filtered_boxes = combined_boxes[final_indices]
        filtered_classes = combined_classes[final_indices]
        filtered_confidences = combined_confidences[final_indices]
        filtered_masks = combined_masks[final_indices] if combined_masks is not None else None
        
        detections = sv.Detections(
            xyxy=filtered_boxes,
            class_id=filtered_classes,
            confidence=filtered_confidences,
            mask=filtered_masks
        )

        return remap_class_ids(detections)

    def predict_hybrid(
        self,
        image: np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE,
        verbose: bool = False,
        tiles_dir: str = None
    ):
        """
        Run hybrid inference combining full-image and SAHI-tiled inference.

        This approach improves detection accuracy by:
        - Full-image inference: Better for large objects with global context
        - SAHI-tiled inference: Better for small objects with local detail

        Results are merged using class-aware NMS.

        Args:
            image: Input image (RGB format)
            confidence: Confidence threshold (0-1)
            verbose: If True, save tile images to tiles_dir
            tiles_dir: Directory to save tile images (required if verbose=True)

        Returns:
            supervision Detections object with merged results
        """
        print("🔀 Hybrid mode: Running full + SAHI inference...")

        # 1. Full image inference
        print("  → Full image inference...")
        full_detections = self.predict(image, confidence)
        print(f"  → Full: {len(full_detections)} detections")

        # 2. SAHI tiled inference
        print("  → SAHI tiled inference...")
        sahi_detections = self.predict_sahi(
            image, confidence,
            verbose=verbose, tiles_dir=tiles_dir
        )
        print(f"  → SAHI: {len(sahi_detections)} detections")

        # 3. Merge results using configured strategy
        merged = merge_detections(full_detections, sahi_detections)
        strategy_name = "NMM" if MERGE_STRATEGY == "nmm" else "NMS"
        agnostic_str = "class-agnostic" if MERGE_CLASS_AGNOSTIC else "class-aware"
        print(f"  → Merged: {len(merged)} detections ({strategy_name}, {agnostic_str}, IoU={MERGE_IOU_THRESHOLD})")

        return merged


def merge_detections(
    detections1,
    detections2,
    strategy: str = MERGE_STRATEGY,
    iou_threshold: float = MERGE_IOU_THRESHOLD,
    class_agnostic: bool = MERGE_CLASS_AGNOSTIC
):
    """
    Merge two detection sets using configurable strategy.

    Args:
        detections1: First supervision Detections object
        detections2: Second supervision Detections object
        strategy: Merge strategy - "nms" or "nmm" (non-max merging)
        iou_threshold: IoU threshold for merging/suppression
        class_agnostic: If True, ignore class labels when merging

    Returns:
        Merged supervision Detections object
    """
    import supervision as sv

    # Handle empty cases
    if len(detections1) == 0:
        return detections2
    if len(detections2) == 0:
        return detections1

    # Combine boxes, classes, confidences
    combined_boxes = np.vstack([detections1.xyxy, detections2.xyxy])
    combined_classes = np.concatenate([detections1.class_id, detections2.class_id])
    combined_confidences = np.concatenate([detections1.confidence, detections2.confidence])

    # Combine masks if both have them
    combined_masks = None
    if detections1.mask is not None and detections2.mask is not None:
        combined_masks = np.vstack([detections1.mask, detections2.mask])
    elif detections1.mask is not None:
        # Pad detections2 with empty masks
        h, w = detections1.mask[0].shape
        empty_masks = np.zeros((len(detections2), h, w), dtype=bool)
        combined_masks = np.vstack([detections1.mask, empty_masks])
    elif detections2.mask is not None:
        # Pad detections1 with empty masks
        h, w = detections2.mask[0].shape
        empty_masks = np.zeros((len(detections1), h, w), dtype=bool)
        combined_masks = np.vstack([empty_masks, detections2.mask])

    # Apply merge strategy
    if strategy == "nmm":
        # Non-Max Merging: combines overlapping boxes
        merged_boxes, merged_confidences, merged_classes = non_max_merge(
            combined_boxes,
            combined_confidences,
            combined_classes,
            iou_threshold
        )

        if len(merged_boxes) == 0:
            return sv.Detections.empty()

        # NMM doesn't preserve indices, so we can't filter masks directly
        # For now, set masks to None when using NMM
        return sv.Detections(
            xyxy=merged_boxes,
            class_id=merged_classes,
            confidence=merged_confidences,
            mask=None  # Masks not supported with NMM yet
        )
    else:
        # NMS: keeps highest confidence, discards overlapping
        final_indices = class_aware_nms(
            combined_boxes,
            combined_confidences,
            combined_classes,
            iou_threshold,
            class_agnostic
        )

        if len(final_indices) == 0:
            return sv.Detections.empty()

        # Filter to keep only NMS-selected detections
        filtered_boxes = combined_boxes[final_indices]
        filtered_classes = combined_classes[final_indices]
        filtered_confidences = combined_confidences[final_indices]
        filtered_masks = combined_masks[final_indices] if combined_masks is not None else None

        return sv.Detections(
            xyxy=filtered_boxes,
            class_id=filtered_classes,
            confidence=filtered_confidences,
            mask=filtered_masks
        )


def remap_class_ids(
    detections,
    class_remap: dict = CLASS_REMAP,
    filter_other: bool = FILTER_OTHER_CLASS
):
    """
    Remap detection class IDs from model output to new scheme.

    Args:
        detections: supervision Detections object
        class_remap: dict mapping old_id -> new_id
        filter_other: if True, remove detections with new_id = -1

    Returns:
        Detections with remapped class IDs
    """
    import supervision as sv

    if len(detections) == 0:
        return detections

    # Remap class IDs
    new_class_ids = np.array([
        class_remap.get(int(cid), -1) for cid in detections.class_id
    ])

    # Filter out "Other" class if requested
    if filter_other:
        keep_mask = new_class_ids != -1
        if not np.any(keep_mask):
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=detections.xyxy[keep_mask],
            class_id=new_class_ids[keep_mask],
            confidence=detections.confidence[keep_mask],
            mask=detections.mask[keep_mask] if detections.mask is not None else None
        )

    # Return all detections with remapped IDs
    return sv.Detections(
        xyxy=detections.xyxy,
        class_id=new_class_ids,
        confidence=detections.confidence,
        mask=detections.mask
    )


def offset_detections(
    detections,
    x_offset: int,
    y_offset: int,
    original_height: int,
    original_width: int
):
    """
    Offset detection coordinates back to original image space.

    Used when inference was run on a cropped image and coordinates
    need to be mapped back to the full original image.

    Args:
        detections: supervision Detections object
        x_offset: X offset of crop from original image
        y_offset: Y offset of crop from original image
        original_height: Height of original image
        original_width: Width of original image

    Returns:
        Detections with coordinates in original image space
    """
    import supervision as sv

    if len(detections) == 0:
        return detections

    # Offset bounding boxes
    boxes = detections.xyxy.copy()
    boxes[:, [0, 2]] += x_offset
    boxes[:, [1, 3]] += y_offset

    # Handle masks - need to place them in full-size mask array
    offset_masks = None
    if detections.mask is not None:
        offset_masks = []
        for mask in detections.mask:
            full_mask = np.zeros((original_height, original_width), dtype=bool)
            mask_h, mask_w = mask.shape[:2]

            # Calculate placement bounds
            y1 = y_offset
            x1 = x_offset
            y2 = min(y_offset + mask_h, original_height)
            x2 = min(x_offset + mask_w, original_width)

            # Place mask in full image
            full_mask[y1:y2, x1:x2] = mask[:y2-y1, :x2-x1]
            offset_masks.append(full_mask)

        offset_masks = np.array(offset_masks)

    return sv.Detections(
        xyxy=boxes,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=offset_masks
    )


def resize_for_inference(
    image: np.ndarray,
    max_size: int = MAX_IMAGE_SIZE
) -> tuple:
    """
    Resize image if it exceeds max size.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    h, w = image.shape[:2]
    
    if max(w, h) <= max_size:
        return image, 1.0
    
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"📐 Resized to: {new_w}x{new_h} (scale: {scale:.2f})")
    
    return resized, scale


def export_to_openvino(checkpoint_path: str, output_dir: str = "openvino_model") -> Optional[str]:
    """
    Export model to OpenVINO IR format for faster inference.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save OpenVINO model
        
    Returns:
        Path to exported model, or None if export failed
    """
    try:
        import torch
        from rfdetr import RFDETRSegPreview
        
        print("Exporting model to OpenVINO format...")
        
        # Load model
        model = RFDETRSegPreview(pretrain_weights=checkpoint_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to ONNX first
        onnx_path = os.path.join(output_dir, "model.onnx")
        dummy_input = torch.randn(1, 3, 504, 504)
        
        torch.onnx.export(
            model.model,
            dummy_input,
            onnx_path,
            opset_version=14,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        
        print(f"✅ ONNX model saved to: {onnx_path}")
        
        # Convert to OpenVINO
        from openvino.tools import mo
        from openvino.runtime import serialize
        
        ov_model = mo.convert_model(onnx_path)
        
        ir_path = os.path.join(output_dir, "model.xml")
        serialize(ov_model, ir_path)
        
        print(f"✅ OpenVINO model saved to: {ir_path}")
        
        return ir_path
        
    except Exception as e:
        print(f"❌ OpenVINO export failed: {e}")
        return None
