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
