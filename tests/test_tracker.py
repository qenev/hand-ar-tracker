"""Unit tests for the HandTracker class and metadata.

Tests initialization, constants, and helper methods using mocks.
"""

from unittest.mock import MagicMock, patch
import pytest

from tracker import HandTracker, LANDMARK_NAMES, HAND_CONNECTIONS


def test_constants() -> None:
    """Test tracker metadata constants."""
    assert len(LANDMARK_NAMES) == 21
    assert "WRIST" in LANDMARK_NAMES
    assert len(HAND_CONNECTIONS) > 0


@patch("mediapipe.solutions.hands.Hands")
def test_tracker_initialization(mock_hands: MagicMock) -> None:
