"""Pytest configuration and global fixtures.

In Windows Python 3.13 environment, MediaPipe does not include the legacy
solutions submodule. This conftest dynamically mocks these missing submodules
at test-time so that tests can import tracker.py and run without errors.
"""

import sys
from unittest.mock import MagicMock

# Create mock objects for the missing mediapipe submodules
mock_solutions = MagicMock()
mock_hands = MagicMock()

# Inject the mocks into sys.modules so imports and getattr operations succeed
sys.modules["mediapipe.solutions"] = mock_solutions
sys.modules["mediapipe.solutions.hands"] = mock_hands

# Also bind them to the mediapipe module if it is already loaded
try:
    import mediapipe as mp
    mp.solutions = mock_solutions
    mock_solutions.hands = mock_hands
except ImportError:
    pass
