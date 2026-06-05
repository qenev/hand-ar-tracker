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

