"""Rule-based gesture recognition from hand landmark geometry.

Detects gestures by analyzing finger extension states and relative
distances between specific hand landmarks. Supports pinch, fist,
open hand, peace sign, and thumbs up gestures.
"""

from typing import List, Tuple, Optional, Dict

from utils.math_utils import calculate_distance, calculate_angle


# MediaPipe hand landmark indices for reference.
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Finger tip and pip index pairs for extension checking.
FINGER_TIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIP_INDICES = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCP_INDICES = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def recognize_gesture(
    landmarks: List[Tuple[float, float, float]],
    pinch_threshold: float = 0.05,
    fist_threshold: float = 0.85,
) -> str:
    """Identify the current hand gesture from landmark positions.

    Analyzes the 21 hand landmarks to determine which gesture is
    being performed. Checks gestures in priority order: pinch first,
    then fist, thumbs up, peace sign, and finally open hand.

    Args:
        landmarks: List of 21 (x, y, z) tuples representing hand
