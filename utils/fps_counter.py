"""Rolling average FPS counter for real-time performance monitoring.

Provides a frame rate calculator that maintains a sliding window
of frame timestamps to compute smooth, accurate FPS readings
with support for min, max, and average statistics.
"""

import time
