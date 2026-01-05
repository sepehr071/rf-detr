"""
Camera Module
=============
Handles image capture from camera with single-frame capture mode.
"""

import cv2
import numpy as np
import platform
import time
import glob
import os
import subprocess
from typing import Optional, Tuple, List

from config import CAMERA_WIDTH, CAMERA_HEIGHT


# Delay between camera operations to prevent USB bus destabilization
CAMERA_RELEASE_DELAY = 0.5  # seconds
CAMERA_OPEN_DELAY = 0.3     # seconds


def check_camera_permissions() -> Tuple[bool, str]:
    """
    Check if current user has camera permissions on Linux.

    On Linux, user must be in 'video' group to access /dev/video* devices.

    Returns:
        Tuple of (has_permission, message)
    """
    if platform.system().lower() != "linux":
        return True, "Non-Linux system, permissions OK"

    # Check if user is in video group
    try:
        import grp
        import pwd
        username = pwd.getpwuid(os.getuid()).pw_name
        groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]
        # Also add primary group
        primary_gid = pwd.getpwuid(os.getuid()).pw_gid
        primary_group = grp.getgrgid(primary_gid).gr_name
        groups.append(primary_group)

        if 'video' in groups:
            print(f"[CAMERA] User '{username}' is in 'video' group - permissions OK")
            return True, f"User in video group"
        else:
            msg = f"User '{username}' is NOT in 'video' group. Run: sudo usermod -aG video {username}"
            print(f"[CAMERA] WARNING: {msg}")
            return False, msg
    except Exception as e:
        print(f"[CAMERA] Could not check permissions: {e}")
        return True, "Could not verify permissions"


def check_uvcvideo_module() -> Tuple[bool, str]:
    """
    Check if uvcvideo kernel module is loaded on Linux.

    Returns:
        Tuple of (is_loaded, message)
    """
    if platform.system().lower() != "linux":
        return True, "Non-Linux system"

    try:
        result = subprocess.run(
            ['lsmod'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if 'uvcvideo' in result.stdout:
            print("[CAMERA] uvcvideo kernel module is loaded")
            return True, "uvcvideo loaded"
        else:
            msg = "uvcvideo module not loaded. Run: sudo modprobe uvcvideo"
            print(f"[CAMERA] WARNING: {msg}")
            return False, msg
    except Exception as e:
        print(f"[CAMERA] Could not check uvcvideo module: {e}")
        return True, "Could not verify module"


def configure_uvcvideo_for_stability():
    """
    Print instructions for configuring uvcvideo module for stability.

    This helps prevent USB camera disconnects on Linux.
    """
    if platform.system().lower() != "linux":
        return

    print("[CAMERA] === USB Camera Stability Tips ===")
    print("[CAMERA] If cameras disappear after scanning, create /etc/modprobe.d/uvcvideo.conf:")
    print("[CAMERA]   options uvcvideo nodrop=1 timeout=5000 quirks=0x80")
    print("[CAMERA] Then reload: sudo rmmod uvcvideo && sudo modprobe uvcvideo")
    print("[CAMERA] Or disable USB autosuspend: echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend")
    print("[CAMERA] ===================================")


def get_camera_backend() -> int:
    """
    Get platform-specific OpenCV camera backend.

    Returns:
        OpenCV backend constant for current platform
    """
    system = platform.system().lower()
    print(f"[CAMERA] Detected platform: {system}")

    if system == "windows":
        print("[CAMERA] Using DSHOW backend (Windows)")
        return cv2.CAP_DSHOW
    elif system == "linux":
        print("[CAMERA] Using V4L2 backend (Linux)")
        return cv2.CAP_V4L2
    elif system == "darwin":
        print("[CAMERA] Using AVFoundation backend (macOS)")
        return cv2.CAP_AVFOUNDATION
    else:
        print("[CAMERA] Using auto-detect backend")
        return -1  # Auto-detect


def validate_camera_quality(frame: np.ndarray, verbose: bool = False) -> bool:
    """
    Validate camera frame quality to filter virtual/broken cameras.

    Checks:
    - Brightness (not too dark or too bright)
    - Variance (not uniform/blank)
    - Edge density (has actual visual content)

    Args:
        frame: BGR frame from camera
        verbose: Print detailed validation info

    Returns:
        True if frame passes quality checks
    """
    if frame is None:
        if verbose:
            print("[VALIDATE] Frame is None")
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check brightness (reject too dark or too bright)
    brightness = np.mean(gray)
    if verbose:
        print(f"[VALIDATE] Brightness: {brightness:.1f} (valid: 5-250)")
    if brightness < 5 or brightness > 250:
        if verbose:
            print("[VALIDATE] FAILED: Brightness out of range")
        return False

    # Check variance (reject uniform/blank frames)
    variance = np.var(gray)
    if verbose:
        print(f"[VALIDATE] Variance: {variance:.1f} (valid: >= 100)")
    if variance < 100:
        if verbose:
            print("[VALIDATE] FAILED: Variance too low (uniform frame)")
        return False

    # Check edge density (reject frames without visual content)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    if verbose:
        print(f"[VALIDATE] Edge density: {edge_density:.4f} (valid: >= 0.01)")
    if edge_density < 0.01:
        if verbose:
            print("[VALIDATE] FAILED: Edge density too low (no content)")
        return False

    if verbose:
        print("[VALIDATE] PASSED: Frame quality OK")
    return True


def enumerate_cameras_package() -> List[int]:
    """
    Try to enumerate cameras using cv2-enumerate-cameras package.

    Returns:
        List of camera indices, or empty list if package not available
    """
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        print("[CAMERA] Using cv2-enumerate-cameras package...")

        cameras = []
        for cam_info in enumerate_cameras(cv2.CAP_V4L2 if platform.system().lower() == "linux" else cv2.CAP_ANY):
            print(f"[CAMERA] Found: index={cam_info.index}, name={cam_info.name}, path={getattr(cam_info, 'path', 'N/A')}")
            cameras.append(cam_info.index)

        return cameras
    except ImportError:
        print("[CAMERA] cv2-enumerate-cameras not installed, using fallback method")
        return []
    except Exception as e:
        print(f"[CAMERA] cv2-enumerate-cameras error: {e}, using fallback method")
        return []


def scan_dev_video_devices() -> List[int]:
    """
    Scan /dev/video* devices on Linux.

    Returns:
        List of video device indices
    """
    if platform.system().lower() != "linux":
        return []

    print("[CAMERA] Scanning /dev/video* devices...")
    devices = sorted(glob.glob('/dev/video*'))
    indices = []

    for dev in devices:
        try:
            # Extract index from /dev/videoX
            idx = int(dev.replace('/dev/video', ''))
            indices.append(idx)
            print(f"[CAMERA] Found device: {dev} (index {idx})")
        except ValueError:
            continue

    return indices


def scan_available_cameras(max_index: int = 4) -> List[int]:
    """
    Scan for available real cameras with quality validation.

    Phase 0: Check permissions and kernel module
    Phase 1: Enumerate all candidate cameras (no testing)
    Phase 2: Test cameras sequentially with fail-fast logic

    Tries multiple enumeration methods:
    1. cv2-enumerate-cameras package (if installed)
    2. /dev/video* scanning (Linux only)
    3. Brute-force index scanning (fallback)

    Args:
        max_index: Maximum camera index to check (for fallback)

    Returns:
        List with single working camera index, or empty if none found
    """
    backend = get_camera_backend()

    print("[CAMERA] === Phase 0: System checks ===")
    check_camera_permissions()
    check_uvcvideo_module()

    print("[CAMERA] === Phase 1: Enumerate cameras ===")

    # Method 1: Try cv2-enumerate-cameras package
    candidates = enumerate_cameras_package()

    # Method 2: Try /dev/video* scanning on Linux
    if not candidates:
        candidates = scan_dev_video_devices()

    # Method 3: Fallback to brute-force index scanning
    if not candidates:
        print(f"[CAMERA] Using brute-force scan (indices 0-{max_index})...")
        candidates = list(range(max_index + 1))

    # Print full list for debugging
    print(f"[CAMERA] Candidate cameras found: {candidates}")

    if not candidates:
        print("[CAMERA] No camera candidates found")
        return []

    print("[CAMERA] === Phase 2: Test cameras (fail-fast) ===")
    print(f"[CAMERA] Using {CAMERA_OPEN_DELAY}s delay between operations to prevent USB issues")

    # Test each candidate camera with fail-fast logic
    for i in candidates:
        print(f"[CAMERA] Testing camera index {i}...")

        # Add delay before opening to prevent USB bus destabilization
        time.sleep(CAMERA_OPEN_DELAY)

        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            print(f"[CAMERA] Index {i}: Failed to open")
            cap.release()
            time.sleep(CAMERA_RELEASE_DELAY)
            continue

        # Set MJPG codec to prevent V4L2 timeout issues
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[CAMERA] Index {i}: Opened, testing frames...")

        valid_frames = 0
        skip_camera = False

        # Test up to 3 frames with fail-fast
        for frame_num in range(1, 4):  # 1, 2, 3
            ret, frame = cap.read()

            if ret and frame is not None:
                is_valid = validate_camera_quality(frame, verbose=(frame_num == 1))
                if is_valid:
                    valid_frames += 1
                    print(f"[CAMERA] Index {i}: Frame {frame_num} - VALID ({valid_frames} total)")

                    # Success: 2 valid frames = use this camera
                    if valid_frames >= 2:
                        cap.release()
                        time.sleep(CAMERA_RELEASE_DELAY)
                        print(f"[CAMERA] SUCCESS: Camera {i} ready (2 valid frames)")
                        return [i]
                else:
                    print(f"[CAMERA] Index {i}: Frame {frame_num} - INVALID")
                    # Fail-fast: if 2nd frame fails, skip to next camera
                    if frame_num == 2:
                        print(f"[CAMERA] Index {i}: Fail-fast triggered, trying next camera")
                        skip_camera = True
                        break
            else:
                print(f"[CAMERA] Index {i}: Frame {frame_num} - CAPTURE FAILED")
                # Fail-fast on capture failure at frame 2
                if frame_num == 2:
                    print(f"[CAMERA] Index {i}: Fail-fast triggered, trying next camera")
                    skip_camera = True
                    break

        cap.release()
        time.sleep(CAMERA_RELEASE_DELAY)  # Delay after release to stabilize USB bus

        if skip_camera:
            continue

        # If we got here with some valid frames but not 2, camera is marginal
        print(f"[CAMERA] Index {i}: REJECTED ({valid_frames} valid frames, need 2)")

    print("[CAMERA] === Scan complete. No valid cameras found ===")
    configure_uvcvideo_for_stability()  # Print tips if no camera found
    return []


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
    print("[CAMERA] === Finding available camera ===")
    backend = get_camera_backend()

    # Try preferred index first if specified
    if preferred_index is not None:
        print(f"[CAMERA] Testing preferred index: {preferred_index}")
        time.sleep(CAMERA_OPEN_DELAY)
        cap = cv2.VideoCapture(preferred_index, backend)
        if cap.isOpened():
            # Set MJPG codec to prevent V4L2 timeout issues
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            valid_frames = 0
            for i in range(3):  # Quick 3-frame test
                ret, frame = cap.read()
                if ret and frame is not None and validate_camera_quality(frame, verbose=(i == 0)):
                    valid_frames += 1
            cap.release()
            time.sleep(CAMERA_RELEASE_DELAY)
            if valid_frames >= 2:
                print(f"[CAMERA] SUCCESS: Using preferred camera index {preferred_index}")
                return preferred_index
            else:
                print(f"[CAMERA] Preferred index {preferred_index} failed validation")
        else:
            print(f"[CAMERA] Preferred index {preferred_index} failed to open")
            cap.release()
            time.sleep(CAMERA_RELEASE_DELAY)

    # Scan for first available camera
    print("[CAMERA] Scanning for first available camera...")
    available_cameras = scan_available_cameras(max_index)

    if available_cameras:
        cam_index = available_cameras[0]
        print(f"[CAMERA] SUCCESS: Using camera index {cam_index}")
        return cam_index

    print("[CAMERA] ERROR: No real cameras found!")
    print("[CAMERA] Defaulting to index 0 (may not work)")
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

        # Set MJPG codec to prevent V4L2 timeout issues on Linux
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

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
        """Release camera resources with delay to prevent USB issues."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            # Delay after release to stabilize USB bus
            time.sleep(CAMERA_RELEASE_DELAY)
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
