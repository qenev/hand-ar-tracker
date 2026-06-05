"""Rule-based gesture recognition from hand landmark geometry.

Detects gestures by analyzing finger extension states and relative
distances between specific hand landmarks. Supports pinch, fist,
open hand, peace sign, and thumbs up gestures.
"""

from typing import List, Tuple, Optional, Dict

from utils.math_utils import calculate_distance, calculate_angle


# MediaPipe hand landmark indices for reference.
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Finger tip and pip index pairs for extension checking.
FINGER_TIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIP_INDICES = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCP_INDICES = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def recognize_gesture(
    landmarks: List[Tuple[float, float, float]],
    pinch_threshold: float = 0.05,
    fist_threshold: float = 0.85,
) -> str:
    """Identify the current hand gesture from landmark positions.

    Analyzes the 21 hand landmarks to determine which gesture is
    being performed. Checks gestures in priority order: pinch first,
    then fist, thumbs up, peace sign, and finally open hand.

    Args:
        landmarks: List of 21 (x, y, z) tuples representing hand
            landmark positions in normalized coordinates.
        pinch_threshold: Maximum distance between thumb tip and index
            tip to register as a pinch gesture.
        fist_threshold: Minimum ratio of curled fingers to total
            fingers required to register as a fist.

    Returns:
        A string label for the detected gesture, or "Unknown" if
        no known gesture pattern is matched.
    """
    if len(landmarks) != 21:
        return "Unknown"
    finger_states = get_finger_states(landmarks)
    if _is_pinch(landmarks, pinch_threshold):
        return "Pinch"
    if _is_fist(finger_states, fist_threshold):
        return "Fist"
    if _is_thumbs_up(finger_states, landmarks):
        return "Thumbs Up"
    if _is_peace_sign(finger_states):
        return "Peace"
    if _is_open_hand(finger_states):
        return "Open Hand"
    return "Unknown"


def get_finger_states(
    landmarks: List[Tuple[float, float, float]],
) -> List[bool]:
    """Determine the extended/curled state of each finger.

    For the thumb, uses a horizontal distance comparison relative
    to the palm. For other fingers, compares tip position to PIP
    joint position along the y-axis.

    Args:
        landmarks: List of 21 (x, y, z) tuples for hand landmarks.

    Returns:
        A list of 5 booleans corresponding to [thumb, index, middle,
        ring, pinky]. True means the finger is extended.
    """
    states: List[bool] = []
    thumb_extended = _is_thumb_extended(landmarks)
    states.append(thumb_extended)
    for i in range(1, 5):
        tip = landmarks[FINGER_TIP_INDICES[i]]
        pip_joint = landmarks[FINGER_PIP_INDICES[i]]
        extended = tip[1] < pip_joint[1]
        states.append(extended)
    return states


def _is_thumb_extended(
    landmarks: List[Tuple[float, float, float]],
) -> bool:
    """Check if the thumb is extended outward from the palm.

    Uses the horizontal distance between the thumb tip and the
    thumb CMC joint compared to the index finger MCP position.

    Args:
        landmarks: List of 21 (x, y, z) tuples for hand landmarks.

    Returns:
        True if the thumb appears to be extended outward.
    """
    thumb_tip = landmarks[THUMB_TIP]
    thumb_mcp = landmarks[THUMB_MCP]
    index_mcp = landmarks[INDEX_MCP]
    thumb_tip_dist = abs(thumb_tip[0] - index_mcp[0])
    thumb_mcp_dist = abs(thumb_mcp[0] - index_mcp[0])
