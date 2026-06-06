"""MediaPipe hand tracking wrapper module.

Encapsulates the MediaPipe Hands solution to provide a clean interface
for hand detection, landmark extraction, and multi-hand tracking
with configurable confidence thresholds.
"""

from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

try:
    import mediapipe as mp
    try:
        from mediapipe.solutions import hands as mp_hands
    except ImportError:
        from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    mp = None
    mp_hands = None

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
        model_complexity: MediaPipe model complexity (0=lite, 1=full).
        hands: The underlying MediaPipe Hands solution instance.
        previous_landmarks: Stored landmarks from the previous frame
            for smoothing purposes.
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
    ) -> None:
        """Initialize the hand tracker with detection parameters.

        Args:
            max_hands: Maximum number of hands to detect per frame.
                Valid range is 1 to 2.
            min_detection_confidence: Minimum confidence threshold for
                the hand detection model. Range 0.0 to 1.0.
            min_tracking_confidence: Minimum confidence threshold for
                the hand tracking model. Range 0.0 to 1.0.
            model_complexity: MediaPipe model complexity.
                0 = lite (fast), 1 = full (accurate). Default is 1.
        """
        self.max_hands: int = max_hands
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.model_complexity: int = model_complexity

        # Determine if MediaPipe is available; if not, use OpenCV fallback
        self.use_opencv: bool = mp is None
        if self.use_opencv:
            # OpenCV fallback does not require initialization
            self._last_opencv_landmarks: list = []
            self._last_opencv_handedness: list = []
            print("[INFO] MediaPipe not available – using OpenCV hand detection fallback.")
        else:
            self._mp_hands = mp_hands
            self.hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            print(f"[INFO] MediaPipe Hands initialised (model_complexity={model_complexity}).")
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
        if self.use_opencv:
            # OpenCV fallback: detect hand and store landmarks
            from utils.opencv_hand_detection import detect_hand
            landmarks = detect_hand(frame)
            # Store for later extraction
            self._last_opencv_landmarks = landmarks
            self._last_opencv_handedness = ["Hand"] * len(landmarks)  # placeholder label
            return None  # No MediaPipe results
        else:
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
        if self.use_opencv:
            # Return landmarks from OpenCV detection
            return self._last_opencv_landmarks
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
            hand_landmarks: MediaPipe hand landmarks protobuf object
                containing 21 landmark positions.

        Returns:
            A list of 21 (x, y, z) coordinate tuples.
        """
        landmarks: List[Tuple[float, float, float]] = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((
                landmark.x,
                landmark.y,
                landmark.z,
            ))
        return landmarks

    def extract_handedness(
        self,
        results: Any,
    ) -> List[str]:
        """Extract hand labels (Left/Right) from MediaPipe results.

        MediaPipe labels are mirrored by default since the camera
        provides a mirror view. This function returns the raw labels
        from the detection results.

        Args:
            results: MediaPipe results object from process_frame.

        Returns:
            A list of hand label strings ("Left" or "Right") in the
            same order as the detected hands. Returns an empty list
            if no hands are detected.
        """
        if self.use_opencv:
            # Return placeholder handedness for each detected hand
            return self._last_opencv_handedness
        if results is None or results.multi_handedness is None:
            return []
        labels: List[str] = []
        for handedness in results.multi_handedness:
            label = handedness.classification[0].label
            labels.append(label)
        return labels

    def get_smoothed_landmarks(
        self,
        hand_index: int,
        current_landmarks: List[Tuple[float, float, float]],
        smoothing_factor: float = 0.7,
    ) -> List[Tuple[float, float, float]]:
        """Apply temporal smoothing to hand landmarks.

        Uses exponential moving average smoothing between the current
        and previous frame landmarks to reduce tracking jitter.

        Args:
            hand_index: Index of the hand (0 or 1) for tracking
                previous frame state.
            current_landmarks: Current frame landmark positions.
            smoothing_factor: Blending weight for current frame.
                Higher = more responsive, lower = smoother.

        Returns:
            Smoothed landmark positions as a list of (x, y, z) tuples.
        """
        previous = self.previous_landmarks.get(hand_index)
        smoothed = smooth_landmarks(
            current_landmarks,
            previous,
            smoothing_factor,
        )
        self.previous_landmarks[hand_index] = smoothed
        return smoothed

    def get_landmark_count(self) -> int:
        """Get the number of landmarks per hand.

        Returns:
            The constant number of landmarks per hand (always 21).
        """
        return 21

    def get_connection_list(self) -> List[Tuple[int, int]]:
        """Get the list of connections defining the hand skeleton.

        Returns:
            A list of (start_index, end_index) tuples defining
            which landmarks should be connected by lines.
        """
        return HAND_CONNECTIONS.copy()

    def release(self) -> None:
        """Release MediaPipe resources.

        Should be called when the tracker is no longer needed
        to free GPU memory and processing resources.
        """
        if not self.use_opencv:
            self.hands.close()
        self.previous_landmarks.clear()
