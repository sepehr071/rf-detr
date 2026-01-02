"""
Object Tracking Module
======================
BoT-SORT tracker wrapper for persistent object ID assignment.
"""

import numpy as np
from boxmot import BoTSORT

from config import (
    TRACKER_FRAME_RATE,
    TRACKER_BUFFER,
    TRACKER_HIGH_THRESH,
    TRACKER_MATCH_THRESH
)


class ObjectTracker:
    """
    Object tracker using BoT-SORT algorithm.

    Assigns persistent IDs to detected objects across frames,
    stabilizing detections even when objects briefly disappear.
    """

    def __init__(
        self,
        frame_rate: float = None,
        lost_track_buffer: int = None,
        track_high_thresh: float = None,
        match_thresh: float = None
    ):
        """
        Initialize BoT-SORT tracker.

        Args:
            frame_rate: Video frame rate (default from config)
            lost_track_buffer: Frames to keep lost tracks (default from config)
            track_high_thresh: Detection threshold for new tracks (default from config)
            match_thresh: Matching threshold (default from config)
        """
        self.frame_rate = frame_rate or TRACKER_FRAME_RATE
        self.lost_track_buffer = lost_track_buffer or TRACKER_BUFFER
        self.track_high_thresh = track_high_thresh or TRACKER_HIGH_THRESH
        self.match_thresh = match_thresh or TRACKER_MATCH_THRESH

        self.tracker = BoTSORT(
            reid_weights=None,
            device='cpu',
            half=False,
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=0.1,
            new_track_thresh=self.track_high_thresh,
            track_buffer=self.lost_track_buffer,
            match_thresh=self.match_thresh,
            frame_rate=self.frame_rate,
            with_reid=False
        )

    def update(self, detections, frame: np.ndarray):
        """
        Update tracker with new detections.

        Args:
            detections: supervision.Detections object with xyxy, confidence, class_id
            frame: Current frame (BGR numpy array)

        Returns:
            Updated supervision.Detections object with tracker_id field populated
        """
        if len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        # Convert to BoxMOT format: N x (x1, y1, x2, y2, conf, cls)
        dets = np.column_stack([
            detections.xyxy,
            detections.confidence,
            detections.class_id
        ])

        # Update tracker
        # Returns M x (x1, y1, x2, y2, id, conf, cls, idx)
        tracks = self.tracker.update(dets, frame)

        if len(tracks) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        # Map tracks back to original detections using idx (last column)
        tracker_ids = np.full(len(detections), -1, dtype=int)
        for track in tracks:
            idx = int(track[7])  # Original detection index
            if 0 <= idx < len(detections):
                tracker_ids[idx] = int(track[4])  # Track ID

        # Keep all detections but mark untracked ones with -1
        detections.tracker_id = tracker_ids

        return detections

    def reset(self):
        """Reset tracker state. Call when switching videos or resetting the system."""
        self.tracker = BoTSORT(
            reid_weights=None,
            device='cpu',
            half=False,
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=0.1,
            new_track_thresh=self.track_high_thresh,
            track_buffer=self.lost_track_buffer,
            match_thresh=self.match_thresh,
            frame_rate=self.frame_rate,
            with_reid=False
        )
