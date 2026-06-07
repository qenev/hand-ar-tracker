    def draw_holographic_ribbon(
        self,
        frame: np.ndarray,
        landmarks_left: List[Tuple[float, float, float]],
        landmarks_right: List[Tuple[float, float, float]],
        time_val: float,
    ) -> np.ndarray:
        """Apply the scanline effect on people/foreground and the starfield backdrop on walls and backgrounds, matching the photo 1:1."""
        h_frame, w_frame = frame.shape[:2]

        coords_left = self._compute_pixel_coords(landmarks_left, w_frame, h_frame)
        coords_right = self._compute_pixel_coords(landmarks_right, w_frame, h_frame)

        if len(coords_left) < 21 or len(coords_right) < 21:
            return frame

        left_top  = np.array(coords_left[8],  dtype=np.float32)
        left_bot  = np.array(coords_left[4],  dtype=np.float32)
        right_top = np.array(coords_right[8], dtype=np.float32)
        right_bot = np.array(coords_right[4], dtype=np.float32)

        poly = np.array([left_top, right_top, right_bot, left_bot], dtype=np.int32)
        x0 = max(0, int(poly[:, 0].min()))
        x1 = min(w_frame, int(poly[:, 0].max()) + 1)
        y0 = max(0, int(poly[:, 1].min()))
        y1 = min(h_frame, int(poly[:, 1].max()) + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        bw, bh = x1 - x0, y1 - y0

        # Create localized masks for ribbon pane
        poly_local = poly - np.array([x0, y0], dtype=np.int32)
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_local], 255)

        # ── 1. Create Space Backdrop (Navy Background + Starfield) ────────────
        pane = np.zeros((bh, bw, 3), dtype=np.uint8)
        pane[:] = (32, 28, 24) # Dark slate/navy space backing (matching 1st photo)
        
        # Add random starfield noise particles inside the pane
        np.random.seed(42)
        noise = np.random.rand(bh, bw)
        stars_mask = (noise > 0.993)
        pane[stars_mask] = (225, 235, 245) # White/blue stars

        # ── 2. Segment the Person/Foreground dynamically from BGR ROI ────────
        roi = frame[y0:y1, x0:x1]
        
        # Convert to grayscale & compute simple gradient/edge thresholds to find the person
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce background texture noise while preserving person outlines
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Simple adaptive thresholding & edge morphing to find the human shape
        edges = cv2.Canny(blurred, 35, 90)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Flood fill or grab contours to generate a silhouette of the user's face/body
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        person_mask = np.zeros((bh, bw), dtype=np.uint8)
        
        # Draw all significant foreground contours filled (removes tiny isolated background highlights)
        for c in contours:
            if cv2.contourArea(c) > 300:
                cv2.drawContours(person_mask, [c], -1, 255, -1)
        
        # Smooth the segmented mask slightly to avoid jagged edge noise
        person_mask = cv2.GaussianBlur(person_mask, (5, 5), 0)
        person_mask = (person_mask > 127).astype(np.uint8) * 255
        
        # Restrict the person mask to be inside the ribbon bounds
        person_mask = cv2.bitwise_and(person_mask, mask)

        # ── 3. Render Blue Glowing Scanline Effect ON the Person Silhouette ────
        y_indices, x_indices = np.where(person_mask > 0)
        if len(x_indices) > 0:
            # Wavy scanline offsets
            scanline = np.abs(np.sin(y_indices * 0.45 - time_val * 12.0))
            
            # Wavy horizontal distortion to the scanline edges to replicate point cloud vibration
            wave_displacement = np.sin(y_indices * 0.15 + time_val * 6.0) * 3.0
            warped_x = np.clip(x_indices + wave_displacement, 0, bw - 1).astype(np.int32)
            
            # Apply glowing blue colors on scanlines
            intensity = scanline * 0.8 + 0.2
            blue_val = (255 * intensity).astype(np.uint8)
            cyan_val = (195 * intensity).astype(np.uint8)
            
            pane[y_indices, warped_x, 0] = blue_val
            pane[y_indices, warped_x, 1] = cyan_val
            pane[y_indices, warped_x, 2] = 0

        # ── 4. Alpha blend the pane over the original background ROI ─────────
        blended = cv2.addWeighted(pane, 0.85, roi, 0.15, 0)
        np.copyto(roi, blended, where=mask[:, :, None] > 0)

        # ── 5. Glowing Cyan Outer Ribbon Border ────────────────────────────────
        contours_ribbon, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contours_ribbon, -1, (255, 245, 130), 2, cv2.LINE_AA) # Cyan-blue boundary
        cv2.drawContours(roi, contours_ribbon, -1, (255, 255, 255), 1, cv2.LINE_AA) # Outer white hairline

        return frame
