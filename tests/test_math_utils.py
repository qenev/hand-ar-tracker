"""Unit tests for the math utility functions.

Tests coordinate calculations, distance measurements, angles, vector normalization,
and landmark smoothing operations.
"""

import math
from typing import List, Tuple

import pytest

from utils.math_utils import (
    calculate_distance,
    calculate_distance_2d,
    calculate_angle,
    normalize_vector,
    landmark_to_pixel,
    smooth_landmarks,
    vector_between,
    vector_magnitude,
    dot_product,
    cross_product,
    midpoint,
    clamp,
)


def test_calculate_distance() -> None:
    """Test 3D Euclidean distance calculation."""
    pt1 = (0.0, 0.0, 0.0)
    pt2 = (3.0, 4.0, 0.0)
    assert math.isclose(calculate_distance(pt1, pt2), 5.0)

    pt3 = (-1.0, -1.0, -1.0)
    pt4 = (1.0, 1.0, 1.0)
    assert math.isclose(calculate_distance(pt3, pt4), math.sqrt(12.0))


def test_calculate_distance_2d() -> None:
    """Test 2D Euclidean distance calculation."""
    pt1 = (0.0, 0.0)
    pt2 = (3.0, 4.0)
    assert math.isclose(calculate_distance_2d(pt1, pt2), 5.0)

