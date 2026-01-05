"""
ROI Calibration Tool
====================
Interactive tool for selecting the Region of Interest (shelf area).

Usage:
    # Calibrate using camera
    python calibrate.py
    python calibrate.py --camera 0
    
    # Calibrate using an image
    python calibrate.py --image sample.jpg

Controls:
    - Click: Add corner point (need 4 points)
    - R: Reset points
    - S: Save ROI and exit
    - Q: Quit without saving
"""

import os
import sys
import cv2
import argparse

from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    ROI_CONFIG_FILE
)
from roi import ROIManager
from camera import ImageCapture, load_image, find_available_camera
from visualization import draw_text_overlay


def calibrate_roi(
    source,
    is_camera: bool = True,
    config_file: str = ROI_CONFIG_FILE
):
    """
    Interactive ROI calibration.
    
    Args:
        source: Camera index (int) or image path (str)
        is_camera: Whether source is a camera
        config_file: Path to save ROI configuration
    """
    roi = ROIManager(config_file)
    
    # Mouse callback
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi.add_point(x, y)
    
    # Open source
    if is_camera:
        camera = ImageCapture(source, CAMERA_WIDTH, CAMERA_HEIGHT)
        if not camera.open():
            print(f"❌ Failed to open camera {source}")
            return None
        
        frame = camera.capture_single()
        if frame is None:
            print("❌ Failed to capture frame")
            camera.release()
            return None
    else:
        frame = load_image(source)
        if frame is None:
            return None
        camera = None
    
    # Create window
    window_name = "ROI Calibration - Click 4 corners of shelf"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    h, w = frame.shape[:2]
    cv2.resizeWindow(window_name, min(w, 1400), min(h, 900))
    
    print("\n" + "=" * 60)
    print("ROI CALIBRATION")
    print("=" * 60)
    print("Click 4 corners of the shelf (clockwise from top-left):")
    print("  1. Top-left")
    print("  2. Top-right")
    print("  3. Bottom-right")
    print("  4. Bottom-left")
    print("-" * 60)
    print("Controls:")
    print("  R - Reset points")
    print("  S - Save ROI and exit")
    print("  Q - Quit without saving")
    print("=" * 60 + "\n")
    
    while True:
        # Get fresh frame if camera
        if camera is not None:
            frame = camera.capture_single()
            if frame is None:
                continue
        
        # Draw ROI overlay
        display = roi.draw_overlay(frame, alpha=0.3, show_points=True)
        
        # Draw status text
        if roi.is_complete():
            status = "ROI Complete! Press S to save, R to reset"
            color = (0, 255, 0)
        else:
            status = f"Click point {len(roi.points) + 1} of 4"
            color = (0, 255, 255)
        
        display = draw_text_overlay(
            display, status, (10, 30), color, (0, 0, 0)
        )
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1 if camera else 0) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("❌ Calibration cancelled")
            break
        elif key == ord('r') or key == ord('R'):
            roi.reset()
        elif key == ord('s') or key == ord('S'):
            if roi.is_complete():
                roi.save()
                print("✅ ROI saved successfully!")
                break
            else:
                print("⚠️ Need 4 points to save ROI")
    
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()
    
    return roi if roi.is_complete() else None


def show_current_roi(config_file: str = ROI_CONFIG_FILE):
    """
    Display the currently saved ROI configuration.
    
    Args:
        config_file: Path to ROI configuration file
    """
    if not os.path.exists(config_file):
        print(f"❌ No ROI configuration found at: {config_file}")
        return
    
    roi = ROIManager(config_file)
    if roi.load():
        print("\n📍 Current ROI Configuration:")
        print(f"   Config file: {config_file}")
        print(f"   Points: {roi.points}")
        
        if roi.is_complete():
            rect = roi.get_bounding_rect()
            print(f"   Bounding rect: x={rect[0]}, y={rect[1]}, w={rect[2]-rect[0]}, h={rect[3]-rect[1]}")
    else:
        print(f"❌ Failed to load ROI from: {config_file}")


def main():
    """Main entry point for calibration tool."""
    parser = argparse.ArgumentParser(description="ROI Calibration Tool")
    
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image file path for calibration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=ROI_CONFIG_FILE,
        help="ROI configuration file path"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show current ROI configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Show current ROI if requested
    if args.show:
        show_current_roi(args.config)
        return
    
    # Determine source
    if args.image:
        source = args.image
        is_camera = False
        
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
    elif args.camera is not None:
        source = args.camera
        is_camera = True
    else:
        # Auto-detect camera (includes permission checks and quality validation)
        print("Auto-detecting camera...")
        source = find_available_camera()
        is_camera = True
    
    # Run calibration
    calibrate_roi(source, is_camera, args.config)


if __name__ == "__main__":
    main()
