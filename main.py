"""Main entry point for the hand AR tracker application.

Initializes the webcam capture, hand tracker, gesture recognizer,
renderer, and FPS counter. Runs the main processing loop that
captures frames, detects hands, recognizes gestures, and renders
the augmented reality overlay in real time.
"""

import sys
import time
import math
from typing import Dict, Any, Optional, Tuple

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
            "fps": 60,
            "rotate_180": True,
        },
        "tracking": {
            "max_hands": 2,
            "min_detection_confidence": 0.8,
            "min_tracking_confidence": 0.7,
            "model_complexity": 1,
        },
        "renderer": {
            "skeleton_color": [0, 255, 0],
            "keypoint_color": [0, 0, 255],
            "keypoint_radius": 6,
            "skeleton_thickness": 2,
            "show_fps": True,
            "show_coordinates": False,
            "show_gesture_label": False,
            "show_hand_label": False,
            "show_device_label": True,
        },
        "gestures": {
            "enabled": False,
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
        model_complexity=track_config.get("model_complexity", 1),
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
        keypoint_radius=rend_config.get("keypoint_radius", 8),
        skeleton_thickness=rend_config["skeleton_thickness"],
        show_fps=rend_config["show_fps"],
        show_coordinates=rend_config.get("show_coordinates", False),
        show_gesture_label=rend_config.get("show_gesture_label", False),
        show_hand_label=rend_config.get("show_hand_label", False),
        show_device_label=rend_config["show_device_label"],
    )
    print("[INFO] Renderer initialized.")
    return renderer


def process_hands(
    frame: np.ndarray,
    tracker: HandTracker,
    renderer: HandRenderer,
    config: Dict[str, Any],
    show_gui: bool = True,
    clap_active: bool = False,
    time_val: float = 0.0,
) -> Tuple[np.ndarray, bool]:
    """Process a single frame for hand detection and rendering."""
    if not hasattr(process_hands, "hands_close"):
        process_hands.hands_close = False

    results = tracker.process_frame(frame)
    all_landmarks = tracker.extract_landmarks(results)
    hand_labels = tracker.extract_handedness(results)

    if len(all_landmarks) == 0:
        if show_gui:
            frame = renderer.draw_no_hands_message(frame)
        return frame, clap_active

    # Smooth ALL landmarks once here — reused for both ribbon and skeleton
    smoothed_all = [
        tracker.get_smoothed_landmarks(i, lm)
        for i, lm in enumerate(all_landmarks)
    ]

    # Pinch detection — both index fingertips (landmark 8) close together
    if len(smoothed_all) == 2:
        tip1 = smoothed_all[0][8]
        tip2 = smoothed_all[1][8]
        dist = math.sqrt((tip1[0] - tip2[0])**2 + (tip1[1] - tip2[1])**2)
        if dist < 0.07:
            if not process_hands.hands_close:
                clap_active = not clap_active
                process_hands.hands_close = True
                print(f"[INFO] Pinch detected! Holographic ribbon: {'ON' if clap_active else 'OFF'}")
        elif dist > 0.14:
            process_hands.hands_close = False
    else:
        process_hands.hands_close = False

    # Render holographic ribbon (uses already-smoothed landmarks)
    if clap_active and len(smoothed_all) == 2:
        sorted_indices = sorted(range(len(smoothed_all)), key=lambda idx: smoothed_all[idx][0][0])
        frame = renderer.draw_holographic_ribbon(
            frame, smoothed_all[sorted_indices[0]], smoothed_all[sorted_indices[1]], time_val
        )

    # Draw skeleton overlay (uses same smoothed landmarks — no double smoothing)
    if show_gui:
        gesture_config = config["gestures"]
        for i, smoothed in enumerate(smoothed_all):
            label = hand_labels[i] if i < len(hand_labels) else ""
            gesture = _get_gesture(smoothed, gesture_config)
            frame = renderer.draw_hand(frame, smoothed, label, gesture)

    return frame, clap_active


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
    fps_counter: FPSCounter,
    device_label: str,
    config: Dict[str, Any],
) -> None:
    """Execute the main webcam processing loop.

    Continuously captures frames, processes them for hand detection
    and gesture recognition, renders overlays, and displays results.
    Exits on 'q' key press or Escape key.

    Args:
        cap: Opened video capture device.
        tracker: Configured hand tracker instance.
        renderer: Configured hand renderer instance.
        fps_counter: FPS counter for performance monitoring.
        device_label: String label for the active compute device.
        config: Full configuration dictionary.
    """
    print("[INFO] Starting main loop. Press 'q' or ESC to quit.")
    rotate_180 = config.get("camera", {}).get("rotate_180", False)
    window_name = "Hand AR Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Store button coordinates for interactive click detection
    btn_bbox = [140, 10, 260, 35]
    show_gui = True      # FPS, flip button, device label
    show_skeleton = True # Hand skeleton dots and lines
    clap_active = False

    def on_mouse(event, x, y, flags, param):
        nonlocal rotate_180
        # Only handle clicks if GUI is visible
        if not show_gui:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_bbox[0] <= x <= btn_bbox[2] and btn_bbox[1] <= y <= btn_bbox[3]:
                rotate_180 = not rotate_180
                print(f"[INFO] Camera rotation toggled. Active: {rotate_180}")

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to capture frame, retrying...")
            continue
        # Mirror horizontally (standard webcam view)
        frame = cv2.flip(frame, 1)
        # Optional 180-degree rotation for upside-down cameras
        if rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        fps_counter.tick()
        
        # Process hands, detect pinches, and render skeletons/ribbon
        frame, clap_active = process_hands(
            frame, tracker, renderer, config,
            show_gui=show_skeleton, clap_active=clap_active, time_val=time.time()
        )
        
        if show_gui:
            frame = renderer.draw_fps(frame, fps_counter.get_fps_string())
            
            # Render flip button and update its exact bounding box
            frame, bbox = renderer.draw_flip_button(frame, rotate_180)
            btn_bbox[0], btn_bbox[1] = bbox[0]
            btn_bbox[2], btn_bbox[3] = bbox[1]

            frame = renderer.draw_device_label(frame, device_label)
            
        cv2.imshow(window_name, frame)
        
        # Capture keyboard input (waitKeyEx captures extended keys)
        key = cv2.waitKeyEx(1)
        if key != -1:
            ascii_key = key & 0xFF
            # q = quit
            if ascii_key == ord("q"):
                break
            # ESC = toggle hand skeleton visibility
            if ascii_key == 27:
                show_skeleton = not show_skeleton
                print(f"[INFO] Skeleton visibility: {'ON' if show_skeleton else 'OFF'}")
            # Delete = toggle all GUI (HUD: FPS, flip button, device label)
            if key in [3014656, 46, 127, 65535, 0x2E0000]:
                show_gui = not show_gui
                print(f"[INFO] HUD visibility: {'ON' if show_gui else 'OFF'}")


def cleanup(
    cap: cv2.VideoCapture,
    tracker: HandTracker,
) -> None:
    """Release all resources and close windows.

    Properly releases the camera capture device, MediaPipe tracker
    resources, and all OpenCV windows.

    Args:
        cap: Video capture device to release.
        tracker: Hand tracker instance to close.
    """
    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Application closed.")


def main() -> None:
    """Application entry point.

    Loads configuration, initializes all components, runs the
    main processing loop, and performs cleanup on exit.
    """
    print("Hand AR Tracker - Starting...")
    config = load_config()
    device = select_device(str(config.get("device", "auto")))
    device_label = get_device_label(device)
    cap = initialize_camera(config)
    tracker = initialize_tracker(config)
    renderer = initialize_renderer(config)
    fps_counter = FPSCounter(window_size=30)
    try:
        run_main_loop(
            cap, tracker, renderer,
            fps_counter, device_label, config,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cleanup(cap, tracker)


if __name__ == "__main__":
    main()
