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
