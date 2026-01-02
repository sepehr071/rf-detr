"""
Configuration constants for the Computer Vision Module.
"""
import numpy as np
# ===================== ROI / PERSPECTIVE POINTS ===================== #
SRC_POINTS = np.float32([
    [210, 286],      # top-left
    [430, 286],      # top-right
    [540, 478],      # bottom-right
    [85, 478]        # bottom-left
])
