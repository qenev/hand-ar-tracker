"""Unit tests for the gesture recognition engine.

Tests gesture classification from mock hand landmark geometry.
"""

import math
from typing import List, Tuple

import pytest

from gesture import (
    recognize_gesture,
    get_finger_states,
    get_extended_finger_count,
    get_gesture_confidence,
)


def make_open_hand_landmarks() -> List[Tuple[float, float, float]]:
    """Helper to generate mock landmarks for an open hand gesture.

    All fingertips are located above their respective PIP joints (y-coordinate is smaller),
    and the thumb is extended.
    """
    # Initialize with wrist and basic palm
    lms = [(0.0, 0.5, 0.0)] * 21
    # INDEX
    lms[5] = (0.1, 0.4, 0.0)  # INDEX_MCP
    lms[6] = (0.1, 0.35, 0.0)  # INDEX_PIP
    lms[8] = (0.1, 0.2, 0.0)  # INDEX_TIP
    # MIDDLE
    lms[9] = (0.0, 0.4, 0.0)  # MIDDLE_MCP
