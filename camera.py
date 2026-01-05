"""
Camera Module
=============
Handles image capture from camera with single-frame capture mode.
"""

import cv2
import numpy as np
import platform
import time
from typing import Optional, Tuple, List

from config import CAMERA_WIDTH, CAMERA_HEIGHT


def get_camera_backend() -> int:
    """
    Get platform-specific OpenCV camera backend.

    Returns:
        OpenCV backend constant for current platform
    """
    system = platform.system().lower()
    if system == "windows":
        return cv2.CAP_DSHOW
    elif system == "linux":
        return cv2.CAP_V4L2
    elif system == "darwin":
        return cv2.CAP_AVFOUNDATION
    else:
        return -1  # Auto-detect


def validate_camera_quality(frame: np.ndarray) -> bool:
    """
    Validate camera frame quality to filter virtual/broken cameras.

    Checks:
    - Brightness (not too dark or too bright)
    - Variance (not uniform/blank)
    - Edge density (has actual visual content)

    Args:
        frame: BGR frame from camera

    Returns:
        True if frame passes quality checks
    """
    if frame is None:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check brightness (reject too dark or too bright)
    brightness = np.mean(gray)
    if brightness < 5 or brightness > 250:
        return False

    # Check variance (reject uniform/blank frames)
    variance = np.var(gray)
    if variance < 100:
        return False

    # Check edge density (reject frames without visual content)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    if edge_density < 0.01:
        return False

    return True


def scan_available_cameras(max_index: int = 4) -> List[int]:
    """
    Scan for available real cameras with quality validation.

    Tests each camera index and validates with multiple frames
    to filter out virtual cameras and broken devices.

    Args:
        max_index: Maximum camera index to check

    Returns:
        List of working camera indices
    """
    backend = get_camera_backend()
    available_cameras = []

    print("[CAMERA] Scanning for available cameras...")

    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            valid_frames = 0
            # Test 5 frames to ensure camera is real
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and validate_camera_quality(frame):
                    valid_frames += 1
                time.sleep(0.1)

            if valid_frames >= 3:
                available_cameras.append(i)
                print(f"[CAMERA] Found real camera at index {i}")
            else:
                print(f"[CAMERA] Rejected virtual/broken camera at index {i}")

        cap.release()
        time.sleep(0.1)

    return available_cameras


def find_available_camera(max_index: int = 4, preferred_index: Optional[int] = None) -> int:
    """
    Find first available working camera with quality validation.

    Scans cameras, validates quality, and returns the best available.
    Supports preferred index that will be tried first if available.

    Args:
        max_index: Maximum camera index to scan
        preferred_index: Preferred camera index to try first

    Returns:
        Index of first working camera, or 0 if none found
    """
    backend = get_camera_backend()
    available_cameras = scan_available_cameras(max_index)

    if not available_cameras:
        print("[CAMERA] No real cameras found! Defaulting to index 0")
        return 0

    # Try preferred index first if specified and available
    if preferred_index is not None and preferred_index in available_cameras:
        print(f"[CAMERA] Trying preferred camera index {preferred_index}")
        cap = cv2.VideoCapture(preferred_index, backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if ret and frame is not None and validate_camera_quality(frame):
            print(f"[CAMERA] Connected to preferred camera index {preferred_index}")
            cap.release()
            return preferred_index
        cap.release()

    # Try other available cameras
    for cam_index in available_cameras:
        if cam_index != preferred_index:
            print(f"[CAMERA] Trying camera index {cam_index}")
            cap = cv2.VideoCapture(cam_index, backend)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            if ret and frame is not None and validate_camera_quality(frame):
                print(f"[CAMERA] Connected to real camera index {cam_index}")
                cap.release()
                return cam_index
            cap.release()

    # Fallback to first available if all validation fails
    if available_cameras:
        print(f"[CAMERA] Using first available camera index {available_cameras[0]}")
        return available_cameras[0]

    print("[CAMERA] No camera found, defaulting to index 0")
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
        # Use platform-specific backend (V4L2 for Linux, DSHOW for Windows)
        backend = get_camera_backend()
        self.cap = cv2.VideoCapture(self.camera_index, backend)

        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_index}")
            return False

        # Minimize buffer to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Configure resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
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
