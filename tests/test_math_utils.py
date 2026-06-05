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
