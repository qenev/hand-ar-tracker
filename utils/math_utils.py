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
