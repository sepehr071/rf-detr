"""
Visualization Module
====================
Drawing and display functions for detection results.
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, Optional

from config import (
    CLASS_NAMES,
    CLASS_COLORS,
    WINDOW_NAME,
    FONT_SCALE,
    FONT_THICKNESS,
    PANEL_WIDTH
)


FONT = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    """
    Visualization class for drawing detection results.

    Handles drawing bounding boxes, masks, labels, and summary panels.
    """

    # Default color for unknown class IDs
    DEFAULT_COLOR = (128, 128, 128)  # Gray

    def __init__(
        self,
        class_names: dict = CLASS_NAMES,
        class_colors: dict = CLASS_COLORS,
        window_name: str = WINDOW_NAME
    ):
        """
        Initialize visualizer.

        Args:
            class_names: Dict mapping class_id -> name
            class_colors: Dict mapping class_id -> BGR color
            window_name: OpenCV window name
        """
        self.class_names = class_names
        self.class_colors = class_colors
        self.window_name = window_name
        self.window_created = False
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections,
        show_masks: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes, masks, and labels on image.
        
        Args:
            image: Input image (BGR format)
            detections: supervision Detections object
            show_masks: Whether to draw segmentation masks
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        if len(detections) == 0:
            return annotated
        
        # Draw masks first (so boxes appear on top)
        if show_masks and detections.mask is not None:
            mask_overlay = annotated.copy()

            for mask, class_id in zip(detections.mask, detections.class_id):
                color = self.class_colors.get(int(class_id), self.DEFAULT_COLOR)
                mask_overlay[mask] = color

            # Blend with original
            annotated = cv2.addWeighted(annotated, 0.6, mask_overlay, 0.4, 0)

        # Check if tracker IDs are available
        has_tracker_ids = (
            hasattr(detections, 'tracker_id') and
            detections.tracker_id is not None and
            len(detections.tracker_id) > 0
        )

        # Draw boxes and labels
        for i, (box, class_id, conf) in enumerate(zip(
            detections.xyxy,
            detections.class_id,
            detections.confidence
        )):
            x1, y1, x2, y2 = map(int, box)
            cid = int(class_id)
            color = self.class_colors.get(cid, self.DEFAULT_COLOR)
            class_name = self.class_names.get(cid, 'Unknown')

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Build label: "#ID ClassName" if tracking, else "ClassName: conf"
            if has_tracker_ids and i < len(detections.tracker_id):
                tracker_id = detections.tracker_id[i]
                if tracker_id >= 0:
                    label = f"#{tracker_id} {class_name}"
                else:
                    label = f"{class_name}: {conf:.2f}"
            else:
                label = f"{class_name}: {conf:.2f}"

            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, FONT, FONT_SCALE, FONT_THICKNESS
            )

            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color, -1
            )

            # Draw label text
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS
            )

        return annotated
    
    def draw_summary(
        self,
        image: np.ndarray,
        detections,
        fps: float,
        mode: str,
        inference_time: float
    ) -> np.ndarray:
        """
        Draw detection summary panel on right side of image.
        
        Args:
            image: Annotated image
            detections: supervision Detections object
            fps: Current frames per second
            mode: Inference mode ('full' or 'sahi')
            inference_time: Time taken for inference in seconds
            
        Returns:
            Image with summary panel added
        """
        h, w = image.shape[:2]
        
        # Count detections per class
        class_counts: Dict[int, int] = defaultdict(int)
        if len(detections) > 0:
            for class_id in detections.class_id:
                class_counts[class_id] += 1
        
        # Create summary panel
        panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        y_offset = 30
        line_height = 28
        
        # Title
        cv2.putText(
            panel, "DETECTION SUMMARY", (10, y_offset),
            FONT, 0.7, (255, 255, 255), 2
        )
        y_offset += line_height + 10
        
        # Mode and FPS
        cv2.putText(
            panel, f"Mode: {mode.upper()}", (10, y_offset),
            FONT, 0.5, (200, 200, 200), 1
        )
        y_offset += line_height
        
        cv2.putText(
            panel, f"FPS: {fps:.1f}", (10, y_offset),
            FONT, 0.5, (200, 200, 200), 1
        )
        y_offset += line_height
        
        cv2.putText(
            panel, f"Inference: {inference_time*1000:.0f}ms", (10, y_offset),
            FONT, 0.5, (200, 200, 200), 1
        )
        y_offset += line_height + 10
        
        # Separator
        cv2.line(panel, (10, y_offset), (PANEL_WIDTH - 10, y_offset), (100, 100, 100), 1)
        y_offset += 15
        
        # Total count
        total = sum(class_counts.values())
        cv2.putText(
            panel, f"TOTAL: {total}", (10, y_offset),
            FONT, 0.7, (0, 255, 0), 2
        )
        y_offset += line_height + 10
        
        # Per-class counts
        cv2.putText(
            panel, "Per Class:", (10, y_offset),
            FONT, 0.5, (200, 200, 200), 1
        )
        y_offset += line_height

        # Sort class IDs for consistent display order (excluding -1 which goes last)
        sorted_class_ids = sorted([cid for cid in self.class_names.keys() if cid >= 0])
        if -1 in self.class_names:
            sorted_class_ids.append(-1)  # Add "Other" at the end

        for class_id in sorted_class_ids:
            class_name = self.class_names.get(class_id, 'Unknown')
            count = class_counts.get(class_id, 0)
            color = self.class_colors.get(class_id, self.DEFAULT_COLOR)

            # Color indicator
            cv2.rectangle(panel, (10, y_offset - 12), (25, y_offset + 2), color, -1)

            # Class name and count
            text = f"{class_name}: {count}"
            text_color = (255, 255, 255) if count > 0 else (100, 100, 100)
            cv2.putText(panel, text, (32, y_offset), FONT, 0.45, text_color, 1)

            y_offset += line_height - 4
        
        # Separator
        y_offset += 10
        cv2.line(panel, (10, y_offset), (PANEL_WIDTH - 10, y_offset), (100, 100, 100), 1)
        y_offset += 15
        
        # Controls
        cv2.putText(panel, "CONTROLS:", (10, y_offset), FONT, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        cv2.putText(panel, "Q - Quit", (10, y_offset), FONT, 0.45, (150, 150, 150), 1)
        y_offset += line_height - 6
        cv2.putText(panel, "S - Screenshot", (10, y_offset), FONT, 0.45, (150, 150, 150), 1)
        y_offset += line_height - 6
        cv2.putText(panel, "M - Toggle masks", (10, y_offset), FONT, 0.45, (150, 150, 150), 1)
        y_offset += line_height - 6
        cv2.putText(panel, "+/- Confidence", (10, y_offset), FONT, 0.45, (150, 150, 150), 1)
        
        # Combine image and panel
        combined = np.hstack([image, panel])
        
        return combined
    
    def create_window(self, width: int, height: int):
        """
        Create display window.
        
        Args:
            width: Window width
            height: Window height
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width + PANEL_WIDTH, height)
        self.window_created = True
    
    def show(self, image: np.ndarray):
        """
        Display image in window.
        
        Args:
            image: Image to display
        """
        if not self.window_created:
            h, w = image.shape[:2]
            self.create_window(w - PANEL_WIDTH, h)
        
        cv2.imshow(self.window_name, image)
    
    def handle_input(self, wait_ms: int = 1) -> str:
        """
        Handle keyboard input.
        
        Args:
            wait_ms: Milliseconds to wait for key press
            
        Returns:
            Key pressed as string, empty string if no key
        """
        key = cv2.waitKey(wait_ms) & 0xFF
        
        if key == 255:  # No key pressed
            return ''
        
        return chr(key)
    
    def destroy(self):
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()
        self.window_created = False
    
    def save_screenshot(self, image: np.ndarray, prefix: str = "screenshot") -> str:
        """
        Save screenshot to disk.
        
        Args:
            image: Image to save
            prefix: Filename prefix
            
        Returns:
            Path to saved file
        """
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        print(f"📸 Screenshot saved: {filename}")
        return filename


def draw_text_overlay(
    image: np.ndarray,
    text: str,
    position: tuple = (10, 30),
    color: tuple = (255, 255, 255),
    background: Optional[tuple] = None
) -> np.ndarray:
    """
    Draw text with optional background on image.
    
    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position
        color: Text color (BGR)
        background: Background color (BGR), None for transparent
        
    Returns:
        Image with text drawn
    """
    result = image.copy()
    
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, 0.8, 2)
    
    x, y = position
    
    if background is not None:
        cv2.rectangle(
            result,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + baseline + 5),
            background, -1
        )
    
    cv2.putText(result, text, position, FONT, 0.8, color, 2)
    
    return result
