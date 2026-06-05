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


