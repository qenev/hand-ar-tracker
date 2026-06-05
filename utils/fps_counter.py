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

        Args:
            window_size: Number of recent frames to use for calculating
                the rolling average FPS. Larger values produce smoother
                readings but respond more slowly to changes.
        """
        self.window_size: int = max(1, window_size)
        self.timestamps: deque[float] = deque(maxlen=self.window_size)
        self.min_fps: float = float("inf")
        self.max_fps: float = 0.0
        self._frame_count: int = 0

    def tick(self) -> None:
        """Record a new frame timestamp.

        Should be called once per frame at the point where you want
        to measure the frame rate. Updates internal statistics.
        """
        current_time = time.perf_counter()
        self.timestamps.append(current_time)
        self._frame_count += 1
        current_fps = self.get_fps()
        if current_fps > 0.0 and self._frame_count > 1:
            self._update_statistics(current_fps)

    def _update_statistics(self, current_fps: float) -> None:
        """Update min and max FPS statistics.

        Args:
            current_fps: The current FPS reading to compare against
                stored min and max values.
        """
        if current_fps < self.min_fps:
            self.min_fps = current_fps
        if current_fps > self.max_fps:
            self.max_fps = current_fps

    def get_fps(self) -> float:
        """Calculate the current rolling average FPS.

        Computes the average frame rate over the timestamps stored
        in the rolling window.

        Returns:
            The current FPS as a float. Returns 0.0 if fewer than
            two frames have been recorded.
        """
        if len(self.timestamps) < 2:
            return 0.0
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span <= 0.0:
            return 0.0
        frame_count = len(self.timestamps) - 1
        return frame_count / time_span

    def get_fps_string(self) -> str:
        """Get a formatted string representation of the current FPS.
