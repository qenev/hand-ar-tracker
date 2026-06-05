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
    lms[10] = (0.0, 0.35, 0.0)  # MIDDLE_PIP
    lms[12] = (0.0, 0.2, 0.0)  # MIDDLE_TIP
    # RING
    lms[13] = (-0.1, 0.4, 0.0)  # RING_MCP
    lms[14] = (-0.1, 0.35, 0.0)  # RING_PIP
    lms[16] = (-0.1, 0.2, 0.0)  # RING_TIP
    # PINKY
    lms[17] = (-0.2, 0.4, 0.0)  # PINKY_MCP
    lms[18] = (-0.2, 0.35, 0.0)  # PINKY_PIP
    lms[20] = (-0.2, 0.2, 0.0)  # PINKY_TIP
    # THUMB
    lms[1] = (0.2, 0.48, 0.0)  # THUMB_CMC
    lms[2] = (0.25, 0.46, 0.0)  # THUMB_MCP
    lms[4] = (0.35, 0.4, 0.0)  # THUMB_TIP
    return lms


def make_fist_landmarks() -> List[Tuple[float, float, float]]:
    """Helper to generate mock landmarks for a fist gesture.

    Fingertips are located below their respective PIP joints (y-coordinate is larger),
    and the thumb is curled in.
    """
    lms = [(0.0, 0.5, 0.0)] * 21
    # INDEX
    lms[5] = (0.1, 0.4, 0.0)  # INDEX_MCP
    lms[6] = (0.1, 0.35, 0.0)  # INDEX_PIP
    lms[8] = (0.1, 0.45, 0.0)  # INDEX_TIP
    # MIDDLE
    lms[9] = (0.0, 0.4, 0.0)  # MIDDLE_MCP
    lms[10] = (0.0, 0.35, 0.0)  # MIDDLE_PIP
    lms[12] = (0.0, 0.45, 0.0)  # MIDDLE_TIP
    # RING
    lms[13] = (-0.1, 0.4, 0.0)  # RING_MCP
    lms[14] = (-0.1, 0.35, 0.0)  # RING_PIP
    lms[16] = (-0.1, 0.45, 0.0)  # RING_TIP
    # PINKY
    lms[17] = (-0.2, 0.4, 0.0)  # PINKY_MCP
    lms[18] = (-0.2, 0.35, 0.0)  # PINKY_PIP
    lms[20] = (-0.2, 0.45, 0.0)  # PINKY_TIP
    # THUMB
    lms[1] = (0.2, 0.48, 0.0)  # THUMB_CMC
    lms[2] = (0.25, 0.46, 0.0)  # THUMB_MCP
    lms[4] = (0.22, 0.46, 0.0)  # THUMB_TIP
    return lms

