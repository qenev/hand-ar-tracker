"""MediaPipe hand tracking wrapper module.

Encapsulates the MediaPipe Hands solution to provide a clean interface
for hand detection, landmark extraction, and multi-hand tracking
with configurable confidence thresholds.
"""

from typing import List, Tuple, Optional, Dict, Any

import cv2
import mediapipe as mp
import numpy as np

from utils.math_utils import smooth_landmarks


# MediaPipe hand connections defining the skeletal mesh topology.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Landmark names for coordinate display.
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


class HandTracker:
    """Wrapper around MediaPipe Hands for hand detection and tracking.

    Provides a simplified interface for processing video frames,
    extracting hand landmarks, and managing tracking state across
    frames with optional landmark smoothing.

    Attributes:
        max_hands: Maximum number of hands to detect simultaneously.
        min_detection_confidence: Minimum confidence for initial detection.
        min_tracking_confidence: Minimum confidence for frame-to-frame tracking.
        hands: The underlying MediaPipe Hands solution instance.
        previous_landmarks: Stored landmarks from the previous frame
            for smoothing purposes.
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        """Initialize the hand tracker with detection parameters.

        Args:
            max_hands: Maximum number of hands to detect per frame.
                Valid range is 1 to 2.
            min_detection_confidence: Minimum confidence threshold for
                the hand detection model. Range 0.0 to 1.0.
            min_tracking_confidence: Minimum confidence threshold for
                the hand tracking model. Range 0.0 to 1.0.
        """
        self.max_hands: int = max_hands
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self._mp_hands = mp.solutions.hands
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.previous_landmarks: Dict[int, List[Tuple[float, float, float]]] = {}

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Optional[Any]:
        """Process a single video frame for hand detection.

        Converts the frame from BGR to RGB color space as required
        by MediaPipe, then runs the hand detection pipeline.

        Args:
            frame: Input video frame as a NumPy array in BGR format
                with shape (height, width, 3).

        Returns:
            The MediaPipe results object containing detected hands,
            or None if processing fails.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results

    def extract_landmarks(
        self,
        results: Any,
    ) -> List[List[Tuple[float, float, float]]]:
        """Extract landmark coordinates from MediaPipe results.

        Converts the MediaPipe landmark protobuf objects into plain
        Python tuples for easier downstream processing.

        Args:
            results: MediaPipe results object from process_frame.

        Returns:
            A list of hands, where each hand is a list of 21
            (x, y, z) coordinate tuples in normalized space.
            Returns an empty list if no hands are detected.
        """
        if results is None or results.multi_hand_landmarks is None:
            self.previous_landmarks.clear()
            return []
        all_hands: List[List[Tuple[float, float, float]]] = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = self._parse_single_hand(hand_landmarks)
            all_hands.append(landmarks)
        return all_hands

    def _parse_single_hand(
        self,
        hand_landmarks: Any,
    ) -> List[Tuple[float, float, float]]:
        """Parse landmarks for a single detected hand.

        Args:
