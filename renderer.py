"""OpenCV rendering module for hand tracking visualization.

Draws skeletal mesh connections, keypoint dots, gesture labels,
FPS overlay, coordinate display, hand labels, and device info
onto video frames using OpenCV drawing primitives.
"""

from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from utils.math_utils import landmark_to_pixel
from tracker import HAND_CONNECTIONS, LANDMARK_NAMES


class HandRenderer:
    """Renders hand tracking visualization overlays on video frames.

    Draws the skeletal mesh, keypoint markers, text labels, and
