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
        gesture_label: str = "",
    ) -> np.ndarray:
        """Draw a complete hand visualization on the frame.

        Renders the skeleton, keypoints, hand label, and gesture
        label for a single detected hand.

        Args:
            frame: Input video frame as BGR NumPy array.
            landmarks: List of 21 (x, y, z) normalized coordinates.
            hand_label: Left/Right hand label string.
            gesture_label: Detected gesture name string.

        Returns:
            The frame with hand visualization drawn on it.
        """
        height, width = frame.shape[:2]
        pixel_coords = self._compute_pixel_coords(
            landmarks, width, height
        )
        frame = self._draw_skeleton(frame, pixel_coords)
        frame = self._draw_keypoints(frame, pixel_coords)
        if self.show_hand_label and hand_label:
            frame = self._draw_hand_label(
                frame, pixel_coords, hand_label
            )
        if self.show_gesture_label and gesture_label:
            frame = self._draw_gesture_label(
                frame, pixel_coords, gesture_label
            )
        if self.show_coordinates:
            frame = self._draw_coordinates(
                frame, landmarks, pixel_coords
            )
        return frame

    def _compute_pixel_coords(
        self,
        landmarks: List[Tuple[float, float, float]],
        width: int,
        height: int,
    ) -> List[Tuple[int, int]]:
        """Convert normalized landmarks to pixel coordinates.

        Args:
            landmarks: List of (x, y, z) normalized coordinates.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            List of (pixel_x, pixel_y) integer coordinate tuples.
        """
        coords: List[Tuple[int, int]] = []
        for lm in landmarks:
            px, py = landmark_to_pixel(lm[0], lm[1], width, height)
            coords.append((px, py))
        return coords

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        pixel_coords: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Draw the skeletal mesh connecting hand landmarks.

        Draws lines between connected landmarks as defined by
        the HAND_CONNECTIONS topology.

        Args:
            frame: Input video frame as BGR NumPy array.
            pixel_coords: List of pixel coordinate tuples for each landmark.

        Returns:
            Frame with skeleton lines drawn.
        """
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):
                start_point = pixel_coords[start_idx]
                end_point = pixel_coords[end_idx]
                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    self.skeleton_color,
                    self.skeleton_thickness,
                    cv2.LINE_AA,
                )
        return frame

    def _draw_keypoints(
        self,
        frame: np.ndarray,
        pixel_coords: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Draw circular markers on each hand landmark.

        Draws filled circles at each of the 21 hand keypoint positions
        using the configured color and radius.

        Args:
            frame: Input video frame as BGR NumPy array.
            pixel_coords: List of pixel coordinate tuples for each landmark.

        Returns:
            Frame with keypoint dots drawn.
        """
        for point in pixel_coords:
            cv2.circle(
                frame,
                point,
                self.keypoint_radius,
                self.keypoint_color,
                cv2.FILLED,
                cv2.LINE_AA,
            )
        return frame

    def _draw_hand_label(
        self,
        frame: np.ndarray,
        pixel_coords: List[Tuple[int, int]],
        label: str,
    ) -> np.ndarray:
        """Draw the Left/Right hand label near the wrist.

        Places a text label slightly above the wrist landmark
        position with a dark background for readability.

        Args:
            frame: Input video frame as BGR NumPy array.
            pixel_coords: List of pixel coordinate tuples.
            label: Hand label string ("Left" or "Right").

        Returns:
            Frame with hand label drawn.
        """
        if len(pixel_coords) == 0:
            return frame
        wrist = pixel_coords[0]
        text_position = (wrist[0] - 20, wrist[1] - 20)
        frame = self._draw_text_with_background(
            frame, label, text_position, scale=0.7
        )
        return frame

    def _draw_gesture_label(
        self,
        frame: np.ndarray,
        pixel_coords: List[Tuple[int, int]],
        gesture: str,
    ) -> np.ndarray:
        """Draw the detected gesture name above the hand.

        Places the gesture label above the middle finger MCP landmark
        with a semi-transparent background for readability.

        Args:
            frame: Input video frame as BGR NumPy array.
            pixel_coords: List of pixel coordinate tuples.
            gesture: Detected gesture name string.

        Returns:
            Frame with gesture label drawn.
        """
        if len(pixel_coords) < 10:
            return frame
        anchor = pixel_coords[9]
        text_position = (anchor[0] - 30, anchor[1] - 40)
        frame = self._draw_text_with_background(
            frame, gesture, text_position, scale=0.8
        )
        return frame

    def _draw_coordinates(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[float, float, float]],
        pixel_coords: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Draw coordinate labels next to key landmarks.

        Shows the normalized (x, y, z) coordinates as small text
        labels next to the fingertip landmarks only, to avoid
        cluttering the display.

        Args:
            frame: Input video frame as BGR NumPy array.
            landmarks: List of (x, y, z) normalized coordinates.
            pixel_coords: List of pixel coordinate tuples.

        Returns:
            Frame with coordinate labels drawn.
        """
        tip_indices = [4, 8, 12, 16, 20]
        for idx in tip_indices:
            if idx < len(landmarks) and idx < len(pixel_coords):
                lm = landmarks[idx]
                px, py = pixel_coords[idx]
                coord_text = f"({lm[0]:.2f},{lm[1]:.2f})"
                text_pos = (px + 10, py + 5)
                cv2.putText(
                    frame,
                    coord_text,
                    text_pos,
                    self._font,
                    0.35,
                    self._text_color,
                    1,
                    cv2.LINE_AA,
                )
        return frame

    def draw_fps(
        self,
        frame: np.ndarray,
        fps_text: str,
    ) -> np.ndarray:
        """Draw the FPS counter in the top-left corner.

        Renders the FPS string with a dark background rectangle
        for readability against any video content.

        Args:
            frame: Input video frame as BGR NumPy array.
            fps_text: Formatted FPS string to display.

        Returns:
            Frame with FPS overlay drawn.
        """
        if not self.show_fps:
            return frame
        position = (10, 30)
        frame = self._draw_text_with_background(
            frame, fps_text, position, scale=0.7
        )
        return frame

    def draw_device_label(
        self,
        frame: np.ndarray,
        device_text: str,
    ) -> np.ndarray:
        """Draw the active device label in the top-right corner.

        Shows which compute device (CPU/CUDA/MPS) is currently
        active as a text overlay.

        Args:
            frame: Input video frame as BGR NumPy array.
            device_text: Device label string to display.

        Returns:
            Frame with device label drawn.
        """
        if not self.show_device_label:
            return frame
        text_size = cv2.getTextSize(
            device_text, self._font, 0.6, 1
        )[0]
        width = frame.shape[1]
        position = (width - text_size[0] - 15, 30)
        frame = self._draw_text_with_background(
            frame, device_text, position, scale=0.6
        )
        return frame

    def draw_no_hands_message(
        self,
