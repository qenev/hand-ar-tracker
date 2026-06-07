"""MediaPipe hand tracking wrapper module.

Encapsulates the MediaPipe Hands solution to provide a clean interface
for hand detection, landmark extraction, and multi-hand tracking
with configurable confidence thresholds.
"""

from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

_MP_AVAILABLE = False
mp = None
mp_hands = None
mp_selfie = None


def _load_mediapipe_hands() -> bool:
    """Import MediaPipe Hands from any supported module path.

    Returns:
        True when MediaPipe Hands is available, False otherwise.
    """
    global mp, mp_hands, mp_selfie, _MP_AVAILABLE
    try:
        import mediapipe as mp_module
        mp = mp_module
    except ImportError:
        return False

    for loader in (
        lambda: (mp.solutions.hands, mp.solutions.selfie_segmentation),
        lambda: (
            __import__("mediapipe.python.solutions.hands", fromlist=["Hands"]),
            __import__("mediapipe.python.solutions.selfie_segmentation", fromlist=["SelfieSegmentation"])
        ),
    ):
        try:
            mp_hands, mp_selfie = loader()
            _MP_AVAILABLE = True
            return True
        except (AttributeError, ImportError, ModuleNotFoundError):
            continue
    return False


_load_mediapipe_hands()

from utils.math_utils import smooth_landmarks


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


class HandTracker:
    """Wrapper around MediaPipe Hands for hand detection and tracking."""

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
    ) -> None:
        """Initialize the hand tracker with detection parameters."""
        if not _MP_AVAILABLE or mp_hands is None:
            raise RuntimeError(
                "MediaPipe Hands is not available. "
                "Run install.bat to install mediapipe==0.10.14."
            )

        self.max_hands: int = max_hands
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.model_complexity: int = model_complexity

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        # Initialize SelfieSegmentation model
        self.selfie = mp_selfie.SelfieSegmentation(model_selection=0) # 0 for general/fast model
        # Initialize FaceMesh solution for facial landmarks
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.previous_face_landmarks: List[Tuple[float, float]] = []
        # Store previous hand landmarks for smoothing (keyed by hand index)
        self.previous_landmarks: Dict[int, List[Tuple[float, float, float]]] = {}

        print(
            f"[INFO] MediaPipe Hands & SelfieSegmentation initialised "
            f"(model_complexity={model_complexity})."
        )

    def process_frame(self, frame: np.ndarray) -> Optional[Any]:
        """Process a single video frame for hand detection."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results

    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run selfie segmentation and background subtraction to detect person + objects."""
        # 1. Selfie segmentation for the person
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.selfie.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        person_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        if results.segmentation_mask is not None:
            person_mask = (results.segmentation_mask * 255).astype(np.uint8)
            _, person_mask = cv2.threshold(person_mask, 128, 255, cv2.THRESH_BINARY)

        # 2. Background subtraction to detect objects
        if not hasattr(self, "bg_subtractor"):
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=16, detectShadows=False
            )
        
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.005)
        
        # Clean background noise using morphological opening/closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Combine person mask and object foreground mask
        combined_mask = cv2.bitwise_or(person_mask, fg_mask)
        return combined_mask

    def extract_landmarks(
        self,
        results: Any,
    ) -> List[List[Tuple[float, float, float]]]:
        """Extract landmark coordinates from MediaPipe results."""
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
        """Parse landmarks for a single detected hand."""
        landmarks: List[Tuple[float, float, float]] = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks

    def extract_handedness(self, results: Any) -> List[str]:
        """Extract hand labels (Left/Right) from MediaPipe results."""
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
        """Apply temporal smoothing to hand landmarks."""
        previous = self.previous_landmarks.get(hand_index)
        smoothed = smooth_landmarks(
            current_landmarks,
            previous,
            smoothing_factor,
        )
        self.previous_landmarks[hand_index] = smoothed
        return smoothed

    def get_landmark_count(self) -> int:
        """Get the number of landmarks per hand."""
        return 21

    def get_connection_list(self) -> List[Tuple[int, int]]:
        """Get the list of connections defining the hand skeleton."""
        return HAND_CONNECTIONS.copy()

    def release(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
        self.selfie.close()
        self.face_mesh.close()
        self.previous_landmarks.clear()

    def extract_face_landmarks(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """Extract facial landmarks using MediaPipe FaceMesh.

        Returns a list of (x, y) normalized coordinates for the detected face.
        If no face is detected, returns an empty list.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True
        if results.multi_face_landmarks:
            # Use first face only
            face_landmarks = results.multi_face_landmarks[0]
            return [(lm.x, lm.y) for lm in face_landmarks.landmark]
        return []
