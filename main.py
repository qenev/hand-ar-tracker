"""Main entry point for the hand AR tracker application.

Initializes the webcam capture, hand tracker, gesture recognizer,
renderer, and FPS counter. Runs the main processing loop that
captures frames, detects hands, recognizes gestures, and renders
the augmented reality overlay in real time.
"""

import sys
from typing import Dict, Any, Optional

import cv2
import yaml
import numpy as np

from tracker import HandTracker
from renderer import HandRenderer
from gesture import recognize_gesture
from utils.fps_counter import FPSCounter
from utils.device_utils import select_device, get_device_label


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and validate the configuration from a YAML file.

    Reads the config.yaml file and merges with default values
    for any missing fields to ensure all required settings exist.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing all configuration settings with
        defaults applied for any missing values.
    """
    defaults = _get_default_config()
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            user_config = yaml.safe_load(config_file)
        if user_config is None:
            return defaults
        return _merge_configs(defaults, user_config)
    except FileNotFoundError:
        print(f"[WARNING] Config file '{config_path}' not found, using defaults.")
        return defaults


def _get_default_config() -> Dict[str, Any]:
    """Return the default configuration dictionary.

    Provides sensible defaults for all configuration parameters
    so the application can run without a config file.

    Returns:
        Dictionary with default values for all config sections.
    """
    return {
        "camera": {
            "index": 0,
            "width": 1280,
            "height": 720,
            "fps": 30,
        },
        "tracking": {
            "max_hands": 2,
            "min_detection_confidence": 0.7,
            "min_tracking_confidence": 0.6,
        },
        "renderer": {
            "skeleton_color": [0, 255, 0],
            "keypoint_color": [0, 0, 255],
            "keypoint_radius": 6,
            "skeleton_thickness": 2,
            "show_fps": True,
            "show_coordinates": True,
            "show_gesture_label": True,
            "show_hand_label": True,
            "show_device_label": True,
        },
        "gestures": {
            "enabled": True,
            "pinch_threshold": 0.05,
            "fist_threshold": 0.85,
        },
        "device": "auto",
    }


def _merge_configs(
    defaults: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Recursively merge user config overrides into default config.

    For nested dictionaries, merges at each level. For other types,
    the override value replaces the default.

    Args:
        defaults: Default configuration dictionary.
        overrides: User-provided configuration overrides.

    Returns:
        Merged configuration dictionary with user values taking
        precedence over defaults.
    """
    merged = defaults.copy()
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def initialize_camera(config: Dict[str, Any]) -> cv2.VideoCapture:
    """Initialize the webcam video capture device.

    Opens the camera specified in the config and sets the requested
    resolution and frame rate.

    Args:
        config: Configuration dictionary containing camera settings
            under the 'camera' key.

    Returns:
        An opened cv2.VideoCapture object ready for frame capture.

    Raises:
        SystemExit: If the camera cannot be opened.
    """
    cam_config = config["camera"]
    cap = cv2.VideoCapture(cam_config["index"])
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check camera index in config.yaml.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_config["fps"])
    print(f"[INFO] Camera opened: index={cam_config['index']}, "
          f"resolution={cam_config['width']}x{cam_config['height']}")
    return cap


def initialize_tracker(config: Dict[str, Any]) -> HandTracker:
    """Create and configure the hand tracker instance.

    Args:
        config: Configuration dictionary containing tracking settings
            under the 'tracking' key.

    Returns:
        A configured HandTracker instance ready for frame processing.
    """
    track_config = config["tracking"]
    tracker = HandTracker(
        max_hands=track_config["max_hands"],
        min_detection_confidence=track_config["min_detection_confidence"],
        min_tracking_confidence=track_config["min_tracking_confidence"],
    )
    print("[INFO] Hand tracker initialized.")
    return tracker


def initialize_renderer(config: Dict[str, Any]) -> HandRenderer:
    """Create and configure the hand renderer instance.

    Args:
        config: Configuration dictionary containing renderer settings
            under the 'renderer' key.

    Returns:
        A configured HandRenderer instance ready for frame rendering.
    """
    rend_config = config["renderer"]
    renderer = HandRenderer(
        skeleton_color=tuple(rend_config["skeleton_color"]),
        keypoint_color=tuple(rend_config["keypoint_color"]),
        keypoint_radius=rend_config["keypoint_radius"],
        skeleton_thickness=rend_config["skeleton_thickness"],
        show_fps=rend_config["show_fps"],
        show_coordinates=rend_config["show_coordinates"],
        show_gesture_label=rend_config["show_gesture_label"],
        show_hand_label=rend_config["show_hand_label"],
        show_device_label=rend_config["show_device_label"],
    )
    print("[INFO] Renderer initialized.")
    return renderer


def process_hands(
    frame: np.ndarray,
    tracker: HandTracker,
    renderer: HandRenderer,
    config: Dict[str, Any],
) -> np.ndarray:
    """Process a single frame for hand detection and rendering.

    Runs the full pipeline: detection, landmark extraction,
    gesture recognition, and overlay rendering for all detected hands.

    Args:
        frame: Input video frame as BGR NumPy array.
        tracker: Configured HandTracker instance.
        renderer: Configured HandRenderer instance.
        config: Full configuration dictionary.

    Returns:
        The frame with all hand visualizations rendered.
    """
    results = tracker.process_frame(frame)
    all_landmarks = tracker.extract_landmarks(results)
    hand_labels = tracker.extract_handedness(results)
    if len(all_landmarks) == 0:
        frame = renderer.draw_no_hands_message(frame)
        return frame
    gesture_config = config["gestures"]
    for i, landmarks in enumerate(all_landmarks):
        smoothed = tracker.get_smoothed_landmarks(i, landmarks)
        label = hand_labels[i] if i < len(hand_labels) else ""
        gesture = _get_gesture(smoothed, gesture_config)
        frame = renderer.draw_hand(frame, smoothed, label, gesture)
    return frame


def _get_gesture(
    landmarks: list,
    gesture_config: Dict[str, Any],
) -> str:
    """Determine the gesture for a set of hand landmarks.

    Args:
        landmarks: List of 21 (x, y, z) coordinate tuples.
        gesture_config: Gesture configuration with thresholds.

    Returns:
        Gesture name string, or empty string if gestures are disabled.
    """
    if not gesture_config.get("enabled", True):
        return ""
    return recognize_gesture(
        landmarks,
        pinch_threshold=gesture_config.get("pinch_threshold", 0.05),
        fist_threshold=gesture_config.get("fist_threshold", 0.85),
    )


def run_main_loop(
    cap: cv2.VideoCapture,
    tracker: HandTracker,
    renderer: HandRenderer,
