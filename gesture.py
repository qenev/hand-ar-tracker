"""Rule-based gesture recognition from hand landmark geometry.

Detects gestures by analyzing finger extension states and relative
distances between specific hand landmarks. Supports pinch, fist,
open hand, peace sign, and thumbs up gestures.
"""

from typing import List, Tuple, Optional, Dict

from utils.math_utils import calculate_distance, calculate_angle


