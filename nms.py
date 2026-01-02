"""
NMS Module
==========
Non-Maximum Suppression utilities for detection post-processing.
"""

import numpy as np
from typing import List

from config import NMS_IOU_THRESHOLD


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = NMS_IOU_THRESHOLD
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression to bounding boxes.
    
    Args:
        boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2]
        scores: numpy array of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression (lower = more aggressive)
    
    Returns:
        numpy array of indices to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    
    # Sort by score (highest first)
    order = np.argsort(scores)[::-1]
    
    keep: List[int] = []
    
    while len(order) > 0:
        # Keep the highest scoring box
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        remaining = order[1:]
        
        # Get coordinates
        xx1 = np.maximum(boxes[i, 0], boxes[remaining, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[remaining, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[remaining, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[remaining, 3])
        
        # Compute intersection
        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        intersection = inter_w * inter_h
        
        # Compute areas
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_remaining = (boxes[remaining, 2] - boxes[remaining, 0]) * \
                        (boxes[remaining, 3] - boxes[remaining, 1])
        
        # Compute IoU
        union = area_i + area_remaining - intersection
        iou = intersection / (union + 1e-6)
        
        # Keep boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = remaining[mask]
    
    return np.array(keep, dtype=int)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    class_agnostic: bool = False
) -> np.ndarray:
    """
    Apply class-aware or class-agnostic Non-Maximum Suppression.

    Args:
        boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2]
        scores: numpy array of shape (N,) with confidence scores
        class_ids: numpy array of shape (N,) with class IDs
        iou_threshold: IoU threshold for suppression
        class_agnostic: If True, ignore class labels and run single NMS on all boxes

    Returns:
        numpy array of indices to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # Class-agnostic: treat all boxes as same class
    if class_agnostic:
        return nms_boxes(boxes, scores, iou_threshold)

    # Class-aware: process each class separately
    final_indices: List[int] = []

    for class_id in np.unique(class_ids):
        # Get indices for this class
        class_mask = class_ids == class_id
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        # Apply NMS for this class
        keep_indices = nms_boxes(class_boxes, class_scores, iou_threshold)

        # Map back to original indices
        final_indices.extend(class_indices[keep_indices])

    return np.array(final_indices, dtype=int)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.

    Args:
        box1: numpy array [x1, y1, x2, y2]
        box2: numpy array [x1, y1, x2, y2]

    Returns:
        IoU value (0-1)
    """
    # Compute intersection
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    inter_w = max(0, xx2 - xx1)
    inter_h = max(0, yy2 - yy1)
    intersection = inter_w * inter_h

    # Compute areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute IoU
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def non_max_merge(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = NMS_IOU_THRESHOLD
) -> tuple:
    """
    Non-Max Merging - combines overlapping boxes instead of discarding.

    Unlike NMS which keeps only the highest confidence box, NMM merges
    overlapping detections into a single consolidated result. This is
    better for handling partial detections and size variations.

    For each pair of overlapping boxes (IoU > threshold):
    - Merged box = union bounding box (encompasses both)
    - Merged confidence = weighted average by area
    - Merged class = class of higher confidence box

    Args:
        boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2]
        scores: numpy array of shape (N,) with confidence scores
        class_ids: numpy array of shape (N,) with class IDs
        iou_threshold: IoU threshold for merging

    Returns:
        Tuple of (merged_boxes, merged_scores, merged_class_ids)
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert to lists for easier manipulation
    boxes = boxes.copy().tolist()
    scores = scores.copy().tolist()
    class_ids = class_ids.copy().tolist()

    # Sort by confidence (highest first)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    boxes = [boxes[i] for i in indices]
    scores = [scores[i] for i in indices]
    class_ids = [class_ids[i] for i in indices]

    merged_boxes = []
    merged_scores = []
    merged_classes = []

    while len(boxes) > 0:
        # Take the highest confidence box
        best_box = boxes.pop(0)
        best_score = scores.pop(0)
        best_class = class_ids.pop(0)

        # Find all boxes that overlap with this one
        i = 0
        while i < len(boxes):
            iou = compute_iou(np.array(best_box), np.array(boxes[i]))

            if iou > iou_threshold:
                # Merge the boxes
                other_box = boxes.pop(i)
                other_score = scores.pop(i)
                class_ids.pop(i)  # Discard other class (keep best)

                # Compute areas for weighted confidence
                area1 = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                area2 = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])

                # Merge box = union (encompassing both)
                best_box = [
                    min(best_box[0], other_box[0]),  # x1
                    min(best_box[1], other_box[1]),  # y1
                    max(best_box[2], other_box[2]),  # x2
                    max(best_box[3], other_box[3]),  # y2
                ]

                # Merge confidence = weighted average by area
                total_area = area1 + area2
                if total_area > 0:
                    best_score = (best_score * area1 + other_score * area2) / total_area
                # best_class stays the same (higher confidence box's class)
            else:
                i += 1

        merged_boxes.append(best_box)
        merged_scores.append(best_score)
        merged_classes.append(best_class)

    return (
        np.array(merged_boxes),
        np.array(merged_scores),
        np.array(merged_classes, dtype=int)
    )
