"""OpenCV rendering module for hand tracking visualization.

Draws skeletal mesh connections, keypoint dots, gesture labels,
FPS overlay, coordinate display, hand labels, and device info
onto video frames using OpenCV drawing primitives.
"""

from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from utils.math_utils import landmark_to_pixel
from tracker import HAND_CONNECTIONS, LANDMARK_NAMES


class HandRenderer:
    """Renders hand tracking visualization overlays on video frames.

    Draws the skeletal mesh, keypoint markers, text labels, and
    diagnostic information onto OpenCV frames based on detected
    hand landmarks and configuration settings.

    Attributes:
        skeleton_color: BGR color tuple for skeleton lines.
        keypoint_color: BGR color tuple for keypoint dots.
        keypoint_radius: Pixel radius for keypoint circles.
        skeleton_thickness: Pixel thickness for skeleton lines.
        show_fps: Whether to display the FPS counter overlay.
        show_coordinates: Whether to display landmark coordinates.
        show_gesture_label: Whether to display detected gesture names.
        show_hand_label: Whether to display Left/Right hand labels.
        show_device_label: Whether to display the active device.
    """

    def __init__(
        self,
        skeleton_color: Tuple[int, int, int] = (0, 255, 0),
        keypoint_color: Tuple[int, int, int] = (0, 0, 255),
        keypoint_radius: int = 6,
        skeleton_thickness: int = 2,
        show_fps: bool = True,
        show_coordinates: bool = True,
        show_gesture_label: bool = True,
        show_hand_label: bool = True,
        show_device_label: bool = True,
    ) -> None:
        """Initialize the renderer with display configuration.

        Args:
            skeleton_color: BGR color for skeleton connection lines.
            keypoint_color: BGR color for landmark keypoint dots.
            keypoint_radius: Radius in pixels for keypoint circles.
            skeleton_thickness: Line thickness in pixels for skeleton.
            show_fps: Enable FPS counter display in top-left corner.
            show_coordinates: Enable landmark coordinate display.
            show_gesture_label: Enable gesture name display near hand.
            show_hand_label: Enable Left/Right label display.
            show_device_label: Enable device label in top-right corner.
        """
        self.skeleton_color = skeleton_color
        self.keypoint_color = keypoint_color
        self.keypoint_radius = keypoint_radius
        self.skeleton_thickness = skeleton_thickness
        self.show_fps = show_fps
        self.show_coordinates = show_coordinates
        self.show_gesture_label = show_gesture_label
        self.show_hand_label = show_hand_label
        self.show_device_label = show_device_label
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._font_thickness = 1
        self._text_color = (255, 255, 255)
        self._bg_color = (0, 0, 0)

    def draw_hand(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[float, float, float]],
        hand_label: str = "",
