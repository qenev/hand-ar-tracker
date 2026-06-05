"""Mathematical utility functions for hand landmark processing.

Provides distance calculations, angle measurements, vector operations,
and coordinate transformation helpers used throughout the tracking pipeline.
"""

import math
from typing import Tuple, List, Optional

import numpy as np


def calculate_distance(
    point_a: Tuple[float, float, float],
    point_b: Tuple[float, float, float],
) -> float:
    """Calculate the Euclidean distance between two 3D points.

    Args:
        point_a: First point as (x, y, z) tuple with normalized coordinates.
        point_b: Second point as (x, y, z) tuple with normalized coordinates.

    Returns:
        The Euclidean distance between the two points as a float.
    """
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_distance_2d(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
) -> float:
    """Calculate the Euclidean distance between two 2D points.

    Args:
        point_a: First point as (x, y) tuple.
        point_b: Second point as (x, y) tuple.

    Returns:
        The Euclidean distance between the two points as a float.
    """
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_angle(
    point_a: Tuple[float, float, float],
    point_b: Tuple[float, float, float],
    point_c: Tuple[float, float, float],
) -> float:
    """Calculate the angle at point_b formed by points a, b, and c.

    Uses the dot product formula to compute the angle in degrees
    at the vertex point_b between rays ba and bc.

    Args:
        point_a: First endpoint as (x, y, z) tuple.
        point_b: Vertex point as (x, y, z) tuple.
        point_c: Second endpoint as (x, y, z) tuple.

    Returns:
        The angle in degrees at point_b, clamped between 0 and 180.
    """
    vector_ba = (
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
        point_a[2] - point_b[2],
    )
    vector_bc = (
        point_c[0] - point_b[0],
        point_c[1] - point_b[1],
        point_c[2] - point_b[2],
    )
    dot_product = sum(a * b for a, b in zip(vector_ba, vector_bc))
    magnitude_ba = math.sqrt(sum(v * v for v in vector_ba))
    magnitude_bc = math.sqrt(sum(v * v for v in vector_bc))
    if magnitude_ba == 0.0 or magnitude_bc == 0.0:
        return 0.0
    cosine = dot_product / (magnitude_ba * magnitude_bc)
    cosine = max(-1.0, min(1.0, cosine))
    angle_radians = math.acos(cosine)
    return math.degrees(angle_radians)


def normalize_vector(
    vector: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Normalize a 3D vector to unit length.

    Args:
        vector: The input vector as (x, y, z) tuple.

    Returns:
        A unit vector in the same direction. Returns (0, 0, 0) if
        the input vector has zero magnitude.
    """
    magnitude = math.sqrt(sum(v * v for v in vector))
    if magnitude == 0.0:
        return (0.0, 0.0, 0.0)
    return (
        vector[0] / magnitude,
        vector[1] / magnitude,
        vector[2] / magnitude,
    )


def landmark_to_pixel(
    landmark_x: float,
    landmark_y: float,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int]:
    """Convert normalized landmark coordinates to pixel coordinates.

    MediaPipe returns landmarks in normalized [0, 1] coordinate space.
    This function maps them to actual pixel positions in the frame.

    Args:
        landmark_x: Normalized x coordinate from MediaPipe (0.0 to 1.0).
        landmark_y: Normalized y coordinate from MediaPipe (0.0 to 1.0).
        frame_width: Width of the video frame in pixels.
        frame_height: Height of the video frame in pixels.

    Returns:
        A tuple of (pixel_x, pixel_y) as integers clamped to frame bounds.
    """
    pixel_x = int(min(max(landmark_x * frame_width, 0), frame_width - 1))
    pixel_y = int(min(max(landmark_y * frame_height, 0), frame_height - 1))
    return (pixel_x, pixel_y)


def smooth_landmarks(
    current: List[Tuple[float, float, float]],
    previous: Optional[List[Tuple[float, float, float]]],
    smoothing_factor: float = 0.5,
) -> List[Tuple[float, float, float]]:
    """Apply exponential moving average smoothing to landmark positions.

    Reduces jitter in hand landmark positions by blending the current
    frame landmarks with the previous frame using a weighted average.

    Args:
        current: List of current frame landmark positions as (x, y, z) tuples.
        previous: List of previous frame landmark positions, or None if
            this is the first frame.
        smoothing_factor: Weight for the current frame (0.0 to 1.0).
            Higher values favor current frame data, lower values
            produce smoother but more delayed tracking.

    Returns:
        Smoothed landmark positions as a list of (x, y, z) tuples.
    """
    if previous is None or len(previous) != len(current):
        return current
    smoothed = []
