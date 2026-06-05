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
    """Test HandTracker initialization with custom parameters."""
    tracker = HandTracker(
        max_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7,
    )
    assert tracker.max_hands == 1
    assert tracker.min_detection_confidence == 0.8
    assert tracker.min_tracking_confidence == 0.7
    assert tracker.get_landmark_count() == 21
    assert tracker.get_connection_list() == HAND_CONNECTIONS

