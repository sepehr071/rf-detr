"""
Label I/O Module
================
Save and load detection labels in text format.

This module extracts the label serialization logic that was embedded
in main.py (lines 47-111).
"""

import numpy as np
import cv2
from typing import List, Optional


def save_detections_to_txt(
    detections,
    output_path: str,
    image_height: int,
    image_width: int
) -> int:
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

    Returns:
        Number of detections saved
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


def load_detections_from_txt(
    input_path: str,
    image_height: int,
    image_width: int
) -> List[dict]:
    """
    Load detections from a text file.

    Args:
        input_path: Path to input .txt file
        image_height: Image height for denormalization
        image_width: Image width for denormalization

    Returns:
        List of detection dictionaries with keys:
        - class_id: int
        - polygon: List[Tuple[float, float]] (denormalized)
        - bbox: Tuple[float, float, float, float] (denormalized xyxy)
    """
    detections = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split by pipe to separate polygon from bbox
            parts = line.split('|')
            if len(parts) != 2:
                continue

            polygon_part = parts[0].strip()
            bbox_part = parts[1].strip()

            # Parse polygon part (class_id + coordinates)
            polygon_values = polygon_part.split()
            class_id = int(polygon_values[0])
            coords = [float(v) for v in polygon_values[1:]]

            # Convert to polygon points (denormalize)
            polygon = []
            for i in range(0, len(coords), 2):
                x = coords[i] * image_width
                y = coords[i + 1] * image_height
                polygon.append((x, y))

            # Parse bbox (denormalize)
            bbox_values = [float(v) for v in bbox_part.split()]
            bbox = (
                bbox_values[0] * image_width,
                bbox_values[1] * image_height,
                bbox_values[2] * image_width,
                bbox_values[3] * image_height
            )

            detections.append({
                'class_id': class_id,
                'polygon': polygon,
                'bbox': bbox
            })

    return detections
