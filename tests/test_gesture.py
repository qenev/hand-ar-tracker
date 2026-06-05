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
