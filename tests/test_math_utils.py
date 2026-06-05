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


def test_calculate_angle() -> None:
    """Test angle calculation between three points."""
    pt_a = (1.0, 0.0, 0.0)
    pt_b = (0.0, 0.0, 0.0)
    pt_c = (0.0, 1.0, 0.0)
    assert math.isclose(calculate_angle(pt_a, pt_b, pt_c), 90.0)

    pt_d = (-1.0, 0.0, 0.0)
    assert math.isclose(calculate_angle(pt_a, pt_b, pt_d), 180.0)

    assert math.isclose(calculate_angle(pt_b, pt_b, pt_c), 0.0)


def test_normalize_vector() -> None:
    """Test 3D vector normalization."""
    vec = (3.0, 0.0, 4.0)
    norm = normalize_vector(vec)
    assert math.isclose(norm[0], 0.6)
    assert math.isclose(norm[1], 0.0)
    assert math.isclose(norm[2], 0.8)

    zero_vec = (0.0, 0.0, 0.0)
    assert normalize_vector(zero_vec) == (0.0, 0.0, 0.0)


def test_landmark_to_pixel() -> None:
    """Test normalized coordinates conversion to pixel coordinates."""
    width = 640
    height = 480
    assert landmark_to_pixel(0.5, 0.5, width, height) == (320, 240)
    assert landmark_to_pixel(-0.1, 1.2, width, height) == (0, 479)


def test_smooth_landmarks() -> None:
    """Test exponential moving average landmark smoothing."""
    curr = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    assert smooth_landmarks(curr, None) == curr

    prev = [(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)]
    smoothed = smooth_landmarks(curr, prev, 0.5)
    assert smoothed[0] == (0.5, 1.0, 1.5)
    assert smoothed[1] == (3.0, 3.5, 4.0)

    # Mismatched lengths
    mismatched_prev = [(0.0, 0.0, 0.0)]
    assert smooth_landmarks(curr, mismatched_prev) == curr


def test_vector_between() -> None:
    """Test computing a vector between two points."""
    pt_a = (1.0, 2.0, 3.0)
    pt_b = (4.0, 6.0, 8.0)
    assert vector_between(pt_a, pt_b) == (3.0, 4.0, 5.0)


def test_vector_magnitude() -> None:
    """Test computing vector magnitude."""
    vec = (3.0, 4.0, 0.0)
    assert math.isclose(vector_magnitude(vec), 5.0)


def test_dot_product() -> None:
    """Test computing dot product of two vectors."""
    vec_a = (1.0, 2.0, 3.0)
    vec_b = (4.0, 5.0, 6.0)
    assert math.isclose(dot_product(vec_a, vec_b), 32.0)


def test_cross_product() -> None:
    """Test computing cross product of two vectors."""
    vec_a = (1.0, 0.0, 0.0)
    vec_b = (0.0, 1.0, 0.0)
    assert cross_product(vec_a, vec_b) == (0.0, 0.0, 1.0)


def test_midpoint() -> None:
    """Test computing midpoint between two points."""
    pt_a = (1.0, 2.0, 3.0)
    pt_b = (3.0, 4.0, 5.0)
    assert midpoint(pt_a, pt_b) == (2.0, 3.0, 4.0)

