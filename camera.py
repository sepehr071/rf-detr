"""
Camera Module
=============
Handles image capture from camera with single-frame capture mode.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from config import CAMERA_WIDTH, CAMERA_HEIGHT


def find_available_camera(max_index: int = 10) -> int:
    """
    Scan for first available working camera.

    Tests each camera index from 0 to max_index-1, returning the first
    one that successfully opens and captures a frame.

    Args:
        max_index: Maximum camera index to check (default: 10)

    Returns:
        Index of first working camera, or 0 if none found
    """
    print("Scanning for available cameras...")

    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()  # Test actual frame capture
            cap.release()
            if ret:
                print(f"Found working camera at index {i}")
                return i

    print("No camera found, defaulting to index 0")
    return 0


class ImageCapture:
    """
    Image capture class for single-frame capture from camera.
    
    Instead of continuous video streaming, this captures one image at a time
    for processing, then captures the next image after processing completes.
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT
    ):
        """
        Initialize camera capture.
        
        Args:
            camera_index: Camera device index (default: 0)
            width: Capture width (default: 1280)
            height: Capture height (default: 960)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False
        
    def open(self) -> bool:
        """
        Open camera connection.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_index}")
            return False
        
        # Configure resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Minimize buffer to reduce latency (hint to driver, may not work on all systems)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Verify actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Camera opened: {actual_w}x{actual_h}")
        
        if actual_w != self.width or actual_h != self.height:
            print(f"⚠️ Requested {self.width}x{self.height}, got {actual_w}x{actual_h}")
            self.width = actual_w
            self.height = actual_h
        
        self.is_open = True
        return True

    def flush_buffer(self, num_frames: int = 5):
        """
        Discard buffered frames to get fresh capture.

        This is important when inference is slow - without flushing,
        cap.read() returns old buffered frames instead of current view.

        Args:
            num_frames: Number of frames to discard (default: 5)
        """
        if self.cap is None:
            return
        for _ in range(num_frames):
            self.cap.grab()

    def capture_single(self) -> Optional[np.ndarray]:
        """
        Capture a single image from camera.

        Flushes the buffer first to ensure we get the current frame,
        not an old buffered frame from when inference was running.

        Returns:
            numpy array of captured image (BGR format), or None if capture failed
        """
        if not self.is_open or self.cap is None:
            print("❌ Camera not opened. Call open() first.")
            return None

        # Flush buffer to get current frame (important when inference is slow)
        self.flush_buffer()

        ret, frame = self.cap.read()
        
        if not ret:
            print("❌ Failed to capture frame")
            return None
        
        return frame
    
    def capture_rgb(self) -> Optional[np.ndarray]:
        """
        Capture a single image and convert to RGB.
        
        Returns:
            numpy array of captured image (RGB format), or None if capture failed
        """
        frame = self.capture_single()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get current capture resolution.
        
        Returns:
            Tuple of (width, height)
        """
        return self.width, self.height
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_open = False
        print("✅ Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy array of image (BGR format), or None if load failed
    """
    import os
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    h, w = frame.shape[:2]
    print(f"✅ Image loaded: {w}x{h}")
    
    return frame
