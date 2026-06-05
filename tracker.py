"""MediaPipe hand tracking wrapper module.

Encapsulates the MediaPipe Hands solution to provide a clean interface
for hand detection, landmark extraction, and multi-hand tracking
with configurable confidence thresholds.
"""

from typing import List, Tuple, Optional, Dict, Any

import cv2
import mediapipe as mp
import numpy as np

from utils.math_utils import smooth_landmarks


# MediaPipe hand connections defining the skeletal mesh topology.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Landmark names for coordinate display.
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
