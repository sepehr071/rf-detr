"""
ROI Module
==========
Region of Interest management for shelf detection.
"""

import os
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional

from config import ROI_CONFIG_FILE


class ROIManager:
    """
    Manages Region of Interest (ROI) for shelf detection.
    
    The ROI is defined by 4 corner points forming a polygon.
    Only detections within this region are processed.
    """
    
    def __init__(self, config_file: str = ROI_CONFIG_FILE):
        """
        Initialize ROI manager.
        
        Args:
            config_file: Path to ROI configuration JSON file
        """
        self.config_file = config_file
        self.points: List[Tuple[int, int]] = []
        self._mask_cache: Optional[np.ndarray] = None
        self._mask_shape: Optional[Tuple[int, int]] = None
    
    def load(self) -> bool:
        """
        Load ROI from config file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.config_file):
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.points = [tuple(p) for p in data.get('points', [])]
                print(f"✅ Loaded ROI with {len(self.points)} points from {self.config_file}")
                return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Failed to load ROI: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save ROI to config file.
        
        Returns:
            True if saved successfully
        """
        data = {
            'points': self.points,
            'description': 'ROI corners: top-left, top-right, bottom-right, bottom-left'
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ ROI saved to {self.config_file}")
            return True
        except IOError as e:
            print(f"❌ Failed to save ROI: {e}")
            return False
    
    def add_point(self, x: int, y: int) -> bool:
        """
        Add a corner point (max 4).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point added, False if already have 4 points
        """
        if len(self.points) < 4:
            self.points.append((x, y))
            self._invalidate_cache()
            print(f"Point {len(self.points)}: ({x}, {y})")
            return True
        return False
    
    def reset(self):
        """Reset all points."""
        self.points = []
        self._invalidate_cache()
        print("🔄 Points reset")
    
    def is_complete(self) -> bool:
        """Check if we have all 4 points."""
        return len(self.points) == 4
    
    def _invalidate_cache(self):
        """Invalidate cached mask."""
        self._mask_cache = None
        self._mask_shape = None
    
    def get_mask(self, height: int, width: int) -> Optional[np.ndarray]:
        """
        Create binary mask for ROI.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Binary mask (uint8), or None if ROI not complete
        """
        if not self.is_complete():
            return None
        
        # Use cached mask if dimensions match
        if self._mask_cache is not None and self._mask_shape == (height, width):
            return self._mask_cache
        
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(self.points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Cache the mask
        self._mask_cache = mask
        self._mask_shape = (height, width)
        
        return mask
    
    def get_bounding_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding rectangle of ROI.
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max), or None if ROI not complete
        """
        if not self.is_complete():
            return None
        
        pts = np.array(self.points)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        return int(x_min), int(y_min), int(x_max), int(y_max)
    
    def apply_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply ROI mask to image (black out areas outside ROI).
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Masked image with areas outside ROI blacked out
        """
        if not self.is_complete():
            return image
        
        h, w = image.shape[:2]
        mask = self.get_mask(h, w)
        
        result = image.copy()
        result[mask == 0] = 0

        return result

    def crop_to_roi(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Crop image to ROI bounding rectangle.

        This is more efficient than masking as it removes unused pixels
        before processing, reducing computation in SAHI.

        Args:
            image: Input image (BGR or RGB)

        Returns:
            Tuple of (cropped_image, x_offset, y_offset)
            The offsets can be used to map coordinates back to original image.
        """
        if not self.is_complete():
            return image, 0, 0

        rect = self.get_bounding_rect()
        if rect is None:
            return image, 0, 0

        x_min, y_min, x_max, y_max = rect

        # Clip to image bounds
        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Crop the image
        cropped = image[y_min:y_max, x_min:x_max].copy()

        crop_h, crop_w = cropped.shape[:2]
        print(f"📐 Cropped to ROI: {crop_w}x{crop_h} (offset: {x_min}, {y_min})")

        return cropped, x_min, y_min

    def get_cropped_mask(self, crop_height: int, crop_width: int, x_offset: int, y_offset: int) -> Optional[np.ndarray]:
        """
        Get ROI mask adjusted for cropped image coordinates.

        Args:
            crop_height: Height of cropped image
            crop_width: Width of cropped image
            x_offset: X offset from original image
            y_offset: Y offset from original image

        Returns:
            Binary mask for the cropped region
        """
        if not self.is_complete():
            return None

        # Adjust polygon points to cropped coordinates
        adjusted_pts = np.array([
            (x - x_offset, y - y_offset) for x, y in self.points
        ], dtype=np.int32)

        # Create mask for cropped image size
        mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
        cv2.fillPoly(mask, [adjusted_pts], 255)

        return mask

    def apply_mask_to_cropped(self, cropped_image: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
        """
        Apply ROI polygon mask to a cropped image.

        Args:
            cropped_image: Already cropped image
            x_offset: X offset used for cropping
            y_offset: Y offset used for cropping

        Returns:
            Cropped image with areas outside polygon blacked out
        """
        if not self.is_complete():
            return cropped_image

        h, w = cropped_image.shape[:2]
        mask = self.get_cropped_mask(h, w, x_offset, y_offset)

        if mask is None:
            return cropped_image

        result = cropped_image.copy()
        result[mask == 0] = 0

        return result

    def filter_detections(self, detections, offset: Tuple[int, int] = (0, 0)):
        """
        Filter detections to only keep those inside ROI.
        
        Args:
            detections: supervision Detections object
            offset: (x, y) offset if image was cropped
            
        Returns:
            Filtered Detections object
        """
        if not self.is_complete() or len(detections) == 0:
            return detections
        
        import supervision as sv
        
        roi_pts = np.array(self.points)
        keep_indices = []
        
        for i, box in enumerate(detections.xyxy):
            # Get box center (accounting for offset)
            cx = (box[0] + box[2]) / 2 + offset[0]
            cy = (box[1] + box[3]) / 2 + offset[1]
            
            # Check if center is inside ROI polygon
            if self._point_in_polygon(cx, cy, roi_pts):
                keep_indices.append(i)
        
        if len(keep_indices) == 0:
            return sv.Detections.empty()
        
        keep_indices = np.array(keep_indices)
        
        return sv.Detections(
            xyxy=detections.xyxy[keep_indices],
            class_id=detections.class_id[keep_indices],
            confidence=detections.confidence[keep_indices],
            mask=detections.mask[keep_indices] if detections.mask is not None else None
        )
    
    def _point_in_polygon(self, x: float, y: float, polygon: np.ndarray) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.
        
        Args:
            x: X coordinate of point
            y: Y coordinate of point
            polygon: Array of polygon vertices
            
        Returns:
            True if point is inside polygon
        """
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def get_scaled(self, scale: float) -> 'ROIManager':
        """
        Get a new ROIManager with scaled points.
        
        Args:
            scale: Scale factor to apply to points
            
        Returns:
            New ROIManager with scaled points
        """
        scaled_roi = ROIManager(self.config_file)
        scaled_roi.points = [(int(x * scale), int(y * scale)) for x, y in self.points]
        return scaled_roi
    
    def draw_overlay(
        self,
        image: np.ndarray,
        alpha: float = 0.3,
        show_points: bool = True
    ) -> np.ndarray:
        """
        Draw ROI overlay on image.
        
        Args:
            image: Input image
            alpha: Transparency for fill (0-1)
            show_points: Whether to show corner points
            
        Returns:
            Image with ROI overlay drawn
        """
        from config import ROI_COLOR_POINT, ROI_COLOR_LINE, ROI_COLOR_FILL, ROI_COLOR_TEXT
        
        result = image.copy()
        
        # Draw points
        if show_points:
            for i, (x, y) in enumerate(self.points):
                cv2.circle(result, (x, y), 8, ROI_COLOR_POINT, -1)
                cv2.putText(
                    result, str(i + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOR_TEXT, 2
                )
        
        # Draw lines between points
        if len(self.points) >= 2:
            for i in range(len(self.points)):
                if i < len(self.points) - 1:
                    cv2.line(result, self.points[i], self.points[i + 1], ROI_COLOR_LINE, 2)
            
            # Close polygon if complete
            if self.is_complete():
                cv2.line(result, self.points[-1], self.points[0], ROI_COLOR_LINE, 2)
                
                # Fill with transparent overlay
                overlay = result.copy()
                pts = np.array(self.points, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], ROI_COLOR_FILL)
                result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        return result
