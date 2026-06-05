"""Rolling average FPS counter for real-time performance monitoring.

Provides a frame rate calculator that maintains a sliding window
of frame timestamps to compute smooth, accurate FPS readings
with support for min, max, and average statistics.
"""

import time
from collections import deque
from typing import Optional


class FPSCounter:
    """Rolling window FPS counter with statistics tracking.

    Maintains a deque of frame timestamps and computes the rolling
    average frames per second over a configurable window size.

    Attributes:
        window_size: Number of frames to include in the rolling average.
        timestamps: Deque of frame arrival timestamps.
        min_fps: Lowest FPS value observed since last reset.
        max_fps: Highest FPS value observed since last reset.
    """

    def __init__(self, window_size: int = 30) -> None:
        """Initialize the FPS counter with a given window size.

