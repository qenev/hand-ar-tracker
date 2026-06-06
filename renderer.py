"""OpenCV rendering module for hand tracking visualization.

Draws skeletal mesh connections and keypoint dots onto video frames
using OpenCV drawing primitives, matching the clean skeleton-only
style (red circles + green lines) shown in the reference photo.
"""

from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from utils.math_utils import landmark_to_pixel
from tracker import HAND_CONNECTIONS, LANDMARK_NAMES


class HandRenderer:
    """Renders hand tracking visualization overlays on video frames.

    Draws the skeletal mesh and keypoint markers onto OpenCV frames
    based on detected hand landmarks and configuration settings.
    The default style matches the reference photo: bright red filled
    circles on each landmark, connected by bright green lines.

    Attributes:
        skeleton_color: BGR color tuple for skeleton lines.
        keypoint_color: BGR color tuple for keypoint dots.
        keypoint_radius: Pixel radius for keypoint circles.
        skeleton_thickness: Pixel thickness for skeleton lines.
        show_fps: Whether to display the FPS counter overlay.
        show_gesture_label: Whether to display detected gesture names.
        show_hand_label: Whether to display Left/Right hand labels.
        show_device_label: Whether to display the active device.
    """

    def __init__(
        self,
        skeleton_color: Tuple[int, int, int] = (0, 255, 0),
        keypoint_color: Tuple[int, int, int] = (0, 0, 255),
        keypoint_radius: int = 8,
        skeleton_thickness: int = 2,
        show_fps: bool = True,
        show_coordinates: bool = False,
        show_gesture_label: bool = False,
        show_hand_label: bool = False,
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

        Renders the skeleton lines first (under the dots), then the
        filled keypoint circles on top, for a clean layered look
        matching the reference photo.

        Args:
            frame: Input video frame as BGR NumPy array.
            landmarks: List of 21 (x, y, z) normalized coordinates.
            hand_label: Left/Right hand label string.
            gesture_label: Detected gesture name string.

        Returns:
            The frame with hand visualization drawn on it.
        """
        height, width = frame.shape[:2]
        pixel_coords = self._compute_pixel_coords(landmarks, width, height)

        # Draw skeleton lines first so dots render on top
        frame = self._draw_skeleton(frame, pixel_coords)
        # Draw filled keypoint circles on top of lines
        frame = self._draw_keypoints(frame, pixel_coords)

        if self.show_hand_label and hand_label:
            frame = self._draw_hand_label(frame, pixel_coords, hand_label)
        if self.show_gesture_label and gesture_label:
            frame = self._draw_gesture_label(frame, pixel_coords, gesture_label)
        if self.show_coordinates:
            frame = self._draw_coordinates(frame, landmarks, pixel_coords)
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

        Draws anti-aliased lines between connected landmarks as defined
        by the HAND_CONNECTIONS topology.

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
        using the configured color and radius, with a thin white border
        to improve visibility against any background.

        Args:
            frame: Input video frame as BGR NumPy array.
            pixel_coords: List of pixel coordinate tuples for each landmark.

        Returns:
            Frame with keypoint dots drawn.
        """
        for point in pixel_coords:
            # Main filled dot
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
        text_position = (wrist[0] - 20, wrist[1] + 30)
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

    def draw_flip_button(
        self,
        frame: np.ndarray,
        rotate_180: bool,
    ) -> Tuple[np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Draw a clickable button next to the FPS counter to flip the camera.

        Args:
            frame: Input video frame as BGR NumPy array.
            rotate_180: Current state of the 180-degree rotation.

        Returns:
            A tuple of (frame, (btn_top_left, btn_bottom_right)) where the
            coordinates define the bounding box of the interactive button.
        """
        btn_text = "Flip 180: ON" if rotate_180 else "Flip 180: OFF"
        scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(btn_text, self._font, scale, thickness)[0]
        
        pad_x = 10
        pad_y = 6
        x_min = 140
        y_min = 10
        x_max = x_min + text_size[0] + 2 * pad_x
        y_max = y_min + text_size[1] + 2 * pad_y
        
        # BGR: Red border/accent if active, grey if inactive
        overlay = frame.copy()
        bg_color = (0, 0, 180) if rotate_180 else (60, 60, 60)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), bg_color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        border_color = (0, 0, 255) if rotate_180 else (180, 180, 180)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), border_color, 1, cv2.LINE_AA)
        
        text_x = x_min + pad_x
        text_y = y_max - pad_y - 2
        cv2.putText(
            frame,
            btn_text,
            (text_x, text_y),
            self._font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return frame, ((x_min, y_min), (x_max, y_max))

    def draw_device_label(
        self,
        frame: np.ndarray,
        device_text: str,
    ) -> np.ndarray:
        """Draw the active device label in the top-right corner.

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
        frame: np.ndarray,
    ) -> np.ndarray:
        """Draw a message when no hands are detected.

        Args:
            frame: Input video frame as BGR NumPy array.

        Returns:
            Frame with the informational message drawn.
        """
        height, width = frame.shape[:2]
        message = "No hands detected – show hands to camera"
        text_size = cv2.getTextSize(
            message, self._font, 0.7, 1
        )[0]
        x = (width - text_size[0]) // 2
        y = height - 30
        frame = self._draw_text_with_background(
            frame, message, (x, y), scale=0.7
        )
        return frame

    def _draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        scale: float = 0.5,
    ) -> np.ndarray:
        """Draw text with a semi-transparent dark background.

        Args:
            frame: Input video frame as BGR NumPy array.
            text: Text string to render.
            position: (x, y) position for the text baseline origin.
            scale: Font scale multiplier for text size.

        Returns:
            Frame with background-highlighted text drawn.
        """
        text_size = cv2.getTextSize(
            text, self._font, scale, self._font_thickness
        )[0]
        pad = 5
        x, y = position
        bg_start = (x - pad, y - text_size[1] - pad)
        bg_end = (x + text_size[0] + pad, y + pad)
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_start, bg_end, self._bg_color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(
            frame,
            text,
            position,
            self._font,
            scale,
            self._text_color,
            self._font_thickness,
            cv2.LINE_AA,
        )
        return frame

    def draw_holographic_ribbon(
        self,
        frame: np.ndarray,
        landmarks_left: List[Tuple[float, float, float]],
        landmarks_right: List[Tuple[float, float, float]],
        time_val: float,
    ) -> np.ndarray:
        """Draw a bending holographic ribbon between two hands (fast single-warp version)."""
        h_frame, w_frame = frame.shape[:2]

        coords_left = self._compute_pixel_coords(landmarks_left, w_frame, h_frame)
        coords_right = self._compute_pixel_coords(landmarks_right, w_frame, h_frame)

        if len(coords_left) < 21 or len(coords_right) < 21:
            return frame

        # Anchor points: index fingertip (8) top, thumb tip (4) bottom
        left_top  = np.array(coords_left[8],  dtype=np.float32)
        left_bot  = np.array(coords_left[4],  dtype=np.float32)
        right_top = np.array(coords_right[8], dtype=np.float32)
        right_bot = np.array(coords_right[4], dtype=np.float32)

        # ── Build LUT and cached meshgrid once ────────────────────────────────
        if not hasattr(self, '_holographic_lut'):
            self._holographic_lut = self._create_holographic_lut()

        TEX_W, TEX_H = 160, 80          # small texture — fast
        S = 12                          # number of horizontal slices
        N = S + 1                       # curved edge sample points aligned with slices

        if not hasattr(self, '_tex_xx'):
            x = np.linspace(0, 12, TEX_W)
            y = np.linspace(0, 8,  TEX_H)
            self._tex_xx, self._tex_yy = np.meshgrid(x, y)

        # ── Animate only the time-dependent parts ─────────────────────────────
        tv = time_val
        v = (
            np.sin(self._tex_xx * 1.8 + np.cos(self._tex_yy * 1.0) + tv * 5.0)
            + np.sin(self._tex_yy * 2.5 + np.sin(self._tex_xx * 0.7) + tv * 3.0)
            + np.sin(np.sqrt((self._tex_xx - 6.0)**2 + (self._tex_yy - 4.0)**2) * 2.0 - tv * 4.0)
            + 3.0
        ) / 6.0
        texture = self._holographic_lut[(np.clip(v, 0, 1) * 255).astype(np.uint8)]

        # ── Build ribbon polygon (top edge + bottom edge) ─────────────────────
        vec  = right_top - left_top
        dist = float(np.linalg.norm(vec)) or 1.0
        px   = -(vec[1] / dist)   # perpendicular unit vector
        py   =  (vec[0] / dist)
        if py < 0:
            px, py = -px, -py

        ts = np.linspace(0, 1, N)
        sag    = np.sin(ts * np.pi) * (dist * 0.15 + 20.0)
        ripple = np.sin(ts * 6.0 - tv * 6.0) * 10.0
        disp   = sag + ripple

        top_pts = (1 - ts)[:, None] * left_top + ts[:, None] * right_top \
                  + disp[:, None] * np.array([px, py])
        bot_pts = (1 - ts)[:, None] * left_bot  + ts[:, None] * right_bot  \
                  + disp[:, None] * np.array([px, py])

        # ── Draw Ribbon Segment by Segment ─────────────────────────────────────
        # S slices is plenty for a smooth curved ribbon and runs extremely fast
        for i in range(S):
            t0 = i / S
            t1 = (i + 1) / S

            # Extract corners for this slice
            p_top0 = top_pts[i]
            p_top1 = top_pts[i + 1]
            p_bot1 = bot_pts[i + 1]
            p_bot0 = bot_pts[i]

            dst_quad = np.array([p_top0, p_top1, p_bot1, p_bot0], dtype=np.float32)

            # Local bounding box for this slice
            x_min = max(0, int(np.min(dst_quad[:, 0])))
            x_max = min(w_frame, int(np.max(dst_quad[:, 0])) + 1)
            y_min = max(0, int(np.min(dst_quad[:, 1])))
            y_max = min(h_frame, int(np.max(dst_quad[:, 1])) + 1)

            if x_max > x_min and y_max > y_min:
                crop_w = x_max - x_min
                crop_h = y_max - y_min

                dst_quad_local = dst_quad - np.array([x_min, y_min], dtype=np.float32)
                src_quad = np.array([
                    [t0 * (TEX_W - 1), 0],
                    [t1 * (TEX_W - 1), 0],
                    [t1 * (TEX_W - 1), TEX_H - 1],
                    [t0 * (TEX_W - 1), TEX_H - 1]
                ], dtype=np.float32)

                M = cv2.getPerspectiveTransform(src_quad, dst_quad_local)
                warped_slice = cv2.warpPerspective(texture, M, (crop_w, crop_h))

                # Hard mask for this slice
                mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, dst_quad_local.astype(np.int32), 255)

                # Copy slice onto frame
                cv2.copyTo(warped_slice, mask, frame[y_min:y_max, x_min:x_max])

        return frame

    def _create_holographic_lut(self) -> np.ndarray:
        # Shape (256, 3) for fast fancy-indexing
        lut = np.zeros((256, 3), dtype=np.uint8)
        points = [
            (0.0,  [240, 240, 255]),
            (0.15, [60,  180, 255]),
            (0.3,  [20,  60,  220]),
            (0.45, [200, 30,  200]),
            (0.6,  [255, 120, 20]),
            (0.75, [200, 220, 30]),
            (0.9,  [80,  255, 200]),
            (1.0,  [240, 240, 255]),
        ]
        for i in range(256):
            t = i / 255.0
            for idx in range(len(points) - 1):
                t0, c0 = points[idx]
                t1, c1 = points[idx + 1]
                if t0 <= t <= t1:
                    f = (t - t0) / (t1 - t0 + 1e-9)
                    lut[i] = [int(c0[j] + f * (c1[j] - c0[j])) for j in range(3)]
                    break
        return lut

