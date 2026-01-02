"""
camera_calibration.py

Fisheye camera calibration, extrinsics estimation, and ray-plane intersection
for precise product placement mapping on shelf coordinates.

Usage:
    1. Calibrate fisheye camera using checkerboard images
    2. Compute camera extrinsics using known shelf points
    3. Map pixels to world coordinates using ray-plane intersection
"""

import cv2
import numpy as np
import os
import json
import glob
from typing import Tuple, List, Optional, Dict


class FisheyeCalibrator:
    """Handles fisheye camera calibration and coordinate mapping."""
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize calibrator.
        
        Args:
            calibration_file: Path to load existing calibration data (JSON)
        """
        self.K = None  # Intrinsic matrix (3x3)
        self.D = None  # Distortion coefficients (4x1)
        self.R = None  # Rotation matrix (3x3)
        self.t = None  # Translation vector (3x1)
        self.image_size = None  # (width, height)
        
        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)
    
    def calibrate_camera(self,
                        checkerboard_images: List[str],
                        checkerboard_size: Tuple[int, int],
                        square_size: float,
                        image_size: Optional[Tuple[int, int]] = None) -> bool:
        """
        Calibrate fisheye camera using checkerboard images.
        
        Args:
            checkerboard_images: List of paths to checkerboard calibration images
            checkerboard_size: (cols, rows) - number of INNER corners (e.g., (9, 6) for 10x7 board)
            square_size: Size of checkerboard square in meters (e.g., 0.025 for 25mm)
            image_size: (width, height) - if None, will be detected from first image
            
        Returns:
            True if calibration successful
        """
        print(f"[CALIBRATION] Starting fisheye calibration with {len(checkerboard_images)} images...")
        
        # Prepare 3D object points (1, N, 3) for fisheye
        objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store 3D points and 2D image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        valid_images = 0
        
        for img_path in checkerboard_images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Could not read image: {img_path}")
                continue
            
            if image_size is None:
                h, w = img.shape[:2]
                image_size = (w, h)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                objpoints.append(objp)
                # Reshape to (1, N, 2) for fisheye
                imgpoints.append(corners_refined.reshape(1, -1, 2))
                valid_images += 1
                print(f"[CALIBRATION] ✓ Valid corners found in: {os.path.basename(img_path)}")
            else:
                print(f"[CALIBRATION] ✗ No corners in: {os.path.basename(img_path)}")
        
        if valid_images < 3:
            print(f"[ERROR] Need at least 3 valid images, got {valid_images}")
            return False
        
        print(f"[CALIBRATION] Processing {valid_images} valid images...")
        
        # Calibrate fisheye camera
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )
        
        # Initialize matrices
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))
        
        # Arrays for per-image rotation and translation
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(valid_images)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(valid_images)]
        
        try:
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                image_size,
                self.K,
                self.D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            
            self.image_size = image_size
            
            print(f"[CALIBRATION] ✓ Success! RMS error: {rms:.4f}")
            print(f"[CALIBRATION] Intrinsic matrix K:\n{self.K}")
            print(f"[CALIBRATION] Distortion coefficients D:\n{self.D.ravel()}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Calibration failed: {e}")
            return False
    
    def compute_extrinsics(self,
                          shelf_points_3d: np.ndarray,
                          shelf_points_2d: np.ndarray) -> bool:
        """
        Compute camera extrinsics (R, t) using known shelf points.
        
        Args:
            shelf_points_3d: Nx3 array of 3D points on shelf (Z=0) in meters
                            Example: [[0, 0, 0], [0.5, 0, 0], [0.5, 0.3, 0], [0, 0.3, 0]]
            shelf_points_2d: Nx2 array of corresponding pixel coordinates
                            Example: [[100, 200], [500, 195], [480, 400], [120, 405]]
            
        Returns:
            True if extrinsics computed successfully
        """
        if self.K is None or self.D is None:
            print("[ERROR] Camera must be calibrated first (K and D required)")
            return False
        
        if len(shelf_points_3d) < 4 or len(shelf_points_2d) < 4:
            print("[ERROR] Need at least 4 point correspondences")
            return False
        
        print(f"[EXTRINSICS] Computing camera pose using {len(shelf_points_3d)} points...")
        
        # Convert to proper format
        object_points = shelf_points_3d.astype(np.float64).reshape(-1, 1, 3)
        image_points = shelf_points_2d.astype(np.float64).reshape(-1, 1, 2)
        
        # Undistort the 2D points
        if self.D.size == 4:
            # Fisheye model
            undistorted_points = cv2.fisheye.undistortPoints(
                image_points,
                self.K,
                self.D,
                P=self.K
            )
        else:
            # Standard model (rational polynomial, etc.)
            undistorted_points = cv2.undistortPoints(
                image_points,
                self.K,
                self.D,
                P=self.K
            )
        
        # Solve PnP with zero distortion (points already undistorted)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            undistorted_points,
            self.K,
            np.zeros((4, 1)),  # Zero distortion (already undistorted)
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print("[ERROR] solvePnP failed")
            return False
        
        # Convert rotation vector to rotation matrix
        self.R, _ = cv2.Rodrigues(rvec)
        self.t = tvec.reshape(3, 1)
        
        print(f"[EXTRINSICS] ✓ Success!")
        print(f"[EXTRINSICS] Rotation matrix R:\n{self.R}")
        print(f"[EXTRINSICS] Translation vector t:\n{self.t.ravel()}")
        
        # Compute camera center in world coordinates
        camera_center = -self.R.T @ self.t
        print(f"[EXTRINSICS] Camera center in world: {camera_center.ravel()}")
        
        return True
    
    def pixel_to_world(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to world coordinates on shelf plane (Z=0).
        Uses ray-plane intersection.
        
        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate
            
        Returns:
            (x_world, y_world) in meters, or None if not calibrated or no intersection
        """
        if self.K is None or self.D is None or self.R is None or self.t is None:
            print("[ERROR] Full calibration required (K, D, R, t)")
            return None
        
        # 1. Undistort pixel to normalized camera coordinates
        pixel_point = np.array([[[px, py]]], dtype=np.float64)
        
        # Undistort pixel to normalized camera coordinates
        if self.D.size == 4:
            # Fisheye model
            undistorted = cv2.fisheye.undistortPoints(
                pixel_point,
                self.K,
                self.D
            )
        else:
            # Standard model
            undistorted = cv2.undistortPoints(
                pixel_point,
                self.K,
                self.D
            )
        
        x_n, y_n = undistorted[0, 0]
        
        # 2. Form ray in camera coordinates
        v_cam = np.array([x_n, y_n, 1.0]).reshape(3, 1)
        
        # 3. Compute camera center in world coordinates
        C = -self.R.T @ self.t
        
        # 4. Compute ray direction in world coordinates
        d = self.R.T @ v_cam
        d = d / np.linalg.norm(d)  # Normalize
        
        # 5. Define shelf plane: Z = 0
        # Plane normal: n = [0, 0, 1]
        # Plane point: p0 = [0, 0, 0]
        n = np.array([0, 0, 1]).reshape(3, 1)
        p0 = np.array([0, 0, 0]).reshape(3, 1)
        
        # 6. Ray-plane intersection
        # Compute parameter s: s = dot(n, (p0 - C)) / dot(n, d)
        denominator = np.dot(n.ravel(), d.ravel())
        
        if abs(denominator) < 1e-6:
            # Ray parallel to plane
            return None
        
        numerator = np.dot(n.ravel(), (p0 - C).ravel())
        s = numerator / denominator
        
        if s < 0:
            # Intersection behind camera
            return None
        
        # 7. Compute world point
        X_world = C + s * d
        
        return float(X_world[0, 0]), float(X_world[1, 0])
    
    def save_calibration(self, filepath: str):
        """Save calibration data to JSON file."""
        if self.K is None or self.D is None:
            print("[ERROR] No calibration data to save")
            return
        
        data = {
            "K": self.K.tolist(),
            "D": self.D.tolist(),
            "image_size": self.image_size
        }
        
        if self.R is not None:
            data["R"] = self.R.tolist()
        
        if self.t is not None:
            data["t"] = self.t.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[CALIBRATION] Saved to {filepath}")
    
    def load_calibration(self, filepath: str) -> bool:
        """Load calibration data from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.K = np.array(data["K"], dtype=np.float64)
            if self.K.ndim == 1 and self.K.size == 9:
                self.K = self.K.reshape(3, 3)
                
            self.D = np.array(data["D"], dtype=np.float64)
            self.image_size = tuple(data.get("image_size", [None, None]))
            
            if "R" in data:
                self.R = np.array(data["R"], dtype=np.float64)
            
            if "t" in data:
                self.t = np.array(data["t"], dtype=np.float64)
            
            print(f"[CALIBRATION] ✓ Loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load calibration: {e}")
            return False


class TemporalSmoother:
    """Apply temporal smoothing to world coordinates to reduce jitter."""
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.history = {}  # track_id -> deque of (x, y) coordinates
    
    def smooth(self, track_id: int, x: float, y: float) -> Tuple[float, float]:
        """
        Apply temporal smoothing to coordinates.
        
        Args:
            track_id: Object track ID
            x: Current x coordinate
            y: Current y coordinate
            
        Returns:
            (smoothed_x, smoothed_y)
        """
        from collections import deque
        
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window_size)
        
        self.history[track_id].append((x, y))
        
        # Compute average
        coords = list(self.history[track_id])
        avg_x = sum(c[0] for c in coords) / len(coords)
        avg_y = sum(c[1] for c in coords) / len(coords)
        
        return avg_x, avg_y
    
    def clear_track(self, track_id: int):
        """Remove history for a track."""
        if track_id in self.history:
            del self.history[track_id]
    
    def cleanup_stale_tracks(self, active_track_ids: set):
        """Remove history for tracks no longer active."""
        stale_ids = set(self.history.keys()) - active_track_ids
        for track_id in stale_ids:
            del self.history[track_id]


# ============================== CALIBRATION UTILITIES ============================== #

def create_calibration_images_interactive(camera_index: int = 0, 
                                         output_dir: str = "calibration_images",
                                         num_images: int = 20):
    """
    Interactive tool to capture calibration images.
    
    Args:
        camera_index: Camera device index
        output_dir: Directory to save images
        num_images: Target number of images to capture
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}")
        return
    
    count = 0
    print(f"[INFO] Press SPACE to capture image, ESC to quit")
    print(f"[INFO] Capture {num_images} images with checkerboard at different angles/positions")
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display instructions
        cv2.putText(frame, f"Images: {count}/{num_images}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Capture | ESC: Quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Calibration Capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            filename = os.path.join(output_dir, f"calib_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[SAVED] {filename}")
            count += 1
            cv2.waitKey(200)  # Brief pause
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} images in {output_dir}")


def run_calibration_example():
    """Example calibration workflow."""
    
    # 1. Capture calibration images
    print("\n=== Step 1: Capture Calibration Images ===")
    print("Run: create_calibration_images_interactive()")
    
    # 2. Calibrate camera
    print("\n=== Step 2: Calibrate Camera ===")
    calibrator = FisheyeCalibrator()
    
    # Get checkerboard images
    calib_images = glob.glob("calibration_images/*.jpg")
    
    if not calib_images:
        print("[ERROR] No calibration images found")
        return
    
    # Calibrate (adjust checkerboard_size and square_size to your board)
    success = calibrator.calibrate_camera(
        checkerboard_images=calib_images,
        checkerboard_size=(9, 6),  # 10x7 board has 9x6 inner corners
        square_size=0.025  # 25mm squares
    )
    
    if not success:
        return
    
    # Save intrinsics
    calibrator.save_calibration("camera_calibration.json")
    
    # 3. Compute extrinsics
    print("\n=== Step 3: Compute Extrinsics ===")
    print("Mark 4+ points on shelf and measure their positions")
    
    # Example: 4 corners of shelf (replace with actual measurements)
    shelf_points_3d = np.array([
        [0.0, 0.0, 0.0],      # Bottom-left corner
        [0.5, 0.0, 0.0],      # Bottom-right corner  
        [0.5, 0.35, 0.0],     # Top-right corner
        [0.0, 0.35, 0.0]      # Top-left corner
    ], dtype=np.float64)
    
    # Corresponding pixel coordinates (replace with actual pixels)
    shelf_points_2d = np.array([
        [228, 411],  # Bottom-left
        [699, 421],  # Bottom-right
        [912, 46],   # Top-right
        [151, 38]    # Top-left
    ], dtype=np.float64)
    
    calibrator.compute_extrinsics(shelf_points_3d, shelf_points_2d)
    
    # Save complete calibration
    calibrator.save_calibration("camera_calibration_full.json")
    
    print("\n=== Calibration Complete ===")


if __name__ == "__main__":
    print("Camera Calibration Module")
    print("Run run_calibration_example() to see example workflow")
