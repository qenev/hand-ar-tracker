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
