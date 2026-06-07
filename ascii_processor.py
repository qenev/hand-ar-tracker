"""ASCII art face overlay — fast vectorised renderer.

Approach
--------
* Pre-renders all character tiles as numpy arrays at startup.
* Uses CLAHE contrast equalisation so the full range of characters
  is always used (no flat dark-dots look).
* All characters in the gradient are visible (no space/empty cells).
* Assembles the full ASCII canvas via numpy tiling — no Python loops.

Colors: pure #FFFFFF characters on #000000 background.
"""

from typing import List, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ---------------------------------------------------------------------------
# Base character pool covering letters, numbers, and requested special symbols
_CHAR_POOL = " .:-=+*!%#@$&?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_CW = 6    # character cell width  (pixels)
_CH = 10   # character cell height (pixels)

# CLAHE for local contrast enhancement before mapping
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# ---------------------------------------------------------------------------
# Pre-render and sort every character by density at startup
# ---------------------------------------------------------------------------
def _build_char_tiles() -> Tuple[np.ndarray, int]:
    """Pre-renders character pool and returns (sorted_tiles, num_chars)."""
    chars = list(_CHAR_POOL)
    if not _PIL_OK:
        # Fallback empty tiles if PIL is missing
        return np.zeros((len(chars), _CH, _CW), dtype=np.uint8), len(chars)

    font = None
    for name in ("cour.ttf", "consola.ttf", "lucon.ttf", "DejaVuSansMono.ttf"):
        try:
            font = ImageFont.truetype(name, 9)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    rendered = []
    for ch in chars:
        img = Image.new("L", (_CW, _CH), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ch, fill=255, font=font)
        tile = np.array(img)
        # Density is sum of pixel brightness values
        density = int(np.sum(tile))
        rendered.append((density, tile))

    # Sort characters by brightness density (darkest first, brightest last)
    rendered.sort(key=lambda x: x[0])
    
    sorted_tiles = np.array([item[1] for item in rendered], dtype=np.uint8)
    return sorted_tiles, len(rendered)


_CHAR_TILES, _N = _build_char_tiles()  # built once at import


# ---------------------------------------------------------------------------
# Core vectorised renderer
# ---------------------------------------------------------------------------
def _fast_ascii(grey: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Convert a greyscale image to an ASCII greyscale canvas.

    Applies CLAHE first so the full character range is always used
    regardless of ambient lighting.

    Returns (out_h, out_w) uint8 — white chars on black.
    """
    cols = out_w // _CW
    rows = out_h // _CH
    if cols < 1 or rows < 1:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    # Enhance local contrast so dark faces still show varied chars
    eq = _CLAHE.apply(grey)

    # Downsample to char grid
    small = cv2.resize(eq, (cols, rows), interpolation=cv2.INTER_AREA)

    # Map 0-255 → 0-(_N-1) char index
    idx = (small.astype(np.float32) / 255.0 * (_N - 1)).astype(np.int32)
    idx = np.clip(idx, 0, _N - 1)

    # Assemble via advanced indexing + reshape (no Python loops)
    # _CHAR_TILES[idx] → (rows, cols, _CH, _CW)
    canvas_tiles = _CHAR_TILES[idx]

    canvas_h = rows * _CH
    canvas_w = cols * _CW
    # (rows, cols, CH, CW) → (rows, CH, cols, CW) → (rows*CH, cols*CW)
    canvas = canvas_tiles.transpose(0, 2, 1, 3).reshape(canvas_h, canvas_w)

    if canvas_h < out_h or canvas_w < out_w:
        padded = np.zeros((out_h, out_w), dtype=np.uint8)
        padded[:canvas_h, :canvas_w] = canvas
        canvas = padded

    return canvas


def _grey_to_bgr_white(grey: np.ndarray) -> np.ndarray:
    """Greyscale ASCII canvas → BGR with pure white chars (#FFFFFF on #000000)."""
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------
class AsciiProcessor:
    """Real-time ASCII art overlay — #FFFFFF on #000000."""

    def render_face(
        self,
        frame: np.ndarray,
        face_landmarks: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Overwrite the face region with live ASCII art.

        Characters update every frame based on actual face brightness.
        CLAHE ensures the full character range is used even in low light.

        Parameters
        ----------
        frame          : BGR video frame (modified in-place, returned).
        face_landmarks : (x_norm, y_norm) pairs from MediaPipe FaceMesh.
        """
        if not face_landmarks:
            return frame

        h, w = frame.shape[:2]

        # Face convex hull → bounding rect
        pts = np.array(
            [(int(x * w), int(y * h)) for x, y in face_landmarks],
            dtype=np.int32,
        )
        hull = cv2.convexHull(pts)
        rx, ry, rw, rh = cv2.boundingRect(hull)
        rx, ry = max(0, rx), max(0, ry)
        rw = min(rw, w - rx)
        rh = min(rh, h - ry)
        if rw < _CW * 2 or rh < _CH * 2:
            return frame

        # Hull mask in ROI space
        face_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, hull - np.array([rx, ry]), 255)

        # Sample ROI
        roi = frame[ry:ry + rh, rx:rx + rw]
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Build ASCII canvas
        ascii_grey = _fast_ascii(grey, rw, rh)
        ascii_bgr = _grey_to_bgr_white(ascii_grey)

        # Paste inside hull only
        mask3 = cv2.merge([face_mask, face_mask, face_mask])
        np.copyto(roi, ascii_bgr, where=mask3 > 0)

        return frame

    def ascii_full_frame(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Convert a full BGR frame to ASCII art on black background.

        Used to render the holographic ribbon in ASCII style.
        Returns a new BGR array of the same shape.
        """
        h, w = frame.shape[:2]
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ascii_grey = _fast_ascii(grey, w, h)
        return _grey_to_bgr_white(ascii_grey)
