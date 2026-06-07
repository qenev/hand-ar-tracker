import cv2
import numpy as np

def detect_hand(frame: np.ndarray) -> list:
    """Detect hand landmarks using a simple OpenCV heuristic.

    This fallback is used when MediaPipe is unavailable. The implementation:
    1. Convert the BGR frame to HSV color space.
    2. Apply a skin‑color mask using a broad HSV range.
    3. Find contours and select the largest one, assuming it is the hand.
    4. Compute the convex hull of the contour.
    5. Sample up to 21 points from the hull (or evenly from the bounding box) and
       return them as normalized (x, y, 0) coordinates where x and y are in
       [0, 1] relative to the frame width/height.
    If no suitable contour is found, an empty list is returned.
    """
    # Step 1: Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Step 2: Skin color mask (this range works reasonably for many skin tones)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Apply some morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Step 3: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    # Choose the largest contour assuming it is the hand
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        # Too small to be a hand
        return []
    # Step 4: Compute convex hull
    hull = cv2.convexHull(largest, returnPoints=True)
    hull_points = hull.squeeze().tolist()
    if not isinstance(hull_points[0], list):
        # When hull has a single point, ensure it's a list of points
        hull_points = [hull_points]
    # Step 5: Sample up to 21 points
    num_points = min(21, len(hull_points))
    # Evenly sample indices
    indices = np.linspace(0, len(hull_points) - 1, num=num_points, dtype=int)
    sampled = [hull_points[i] for i in indices]
    height, width = frame.shape[:2]
    # Normalize coordinates and add dummy z=0
    normalized = [(x / width, y / height, 0.0) for x, y in sampled]
    # If fewer than 21 points, pad with the last point
    while len(normalized) < 21:
        normalized.append(normalized[-1])
    return [normalized]
