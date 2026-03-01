import cv2
import numpy as np

class RopeDetector:
    def __init__(self):
        self.min_pixel_ratio = 0.003  # ignore tiny noise
        self.min_edge_ratio = 0.001 #fast cpu guard
        self.min_lines = 4

        # ----- Temporal Stability -----
        self.detection_counter = 0
        self.required_stable_frames = 5 #must detect 5 consecutive frames

    def detect(self, frame, hsv_lower, hsv_upper, draw=True):
        """
        Args:
            frame (np.ndarray): BGR image
            draw (bool): whether to draw overlay

        Returns:
            rope_found (bool)
            rope_info (dict | None)
            overlay (np.ndarray)
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # --- Color mask ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        # --- Morphological cleanup ---
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        #pixel ratio guard
        pixel_ratio = np.count_nonzero(mask) / (h * w)
        if pixel_ratio < self.min_pixel_ratio:
            return False, None, overlay

        # --- Edge detection ---
        edges = cv2.Canny(mask, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (h * w)
        if edge_ratio < self.min_edge_ratio:
            return False, None, overlay

        # --- Line detection ---
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=80,
            maxLineGap=20
        )

        if lines is None:
            self.detection_counter = 0
            return False, None, overlay

        # --- Filter strong vertical lines only ---
        vertical_lines = []

        for l in lines:
            x1, y1, x2, y2 = l[0]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # Keep near-vertical lines only
            if dy > dx * 2:   # vertical dominance condition
                vertical_lines.append((x1, y1, x2, y2))

        if len(vertical_lines) < self.min_lines:
            self.detection_counter = 0
            return False, None, overlay

        # --- Temporal stability ---
        self.detection_counter += 1

        if self.detection_counter < self.required_stable_frames:
            return False, None, overlay

        # --- Compute rope center ---
        xs = []

        for (x1, y1, x2, y2) in vertical_lines:
            xs.extend([x1, x2])

            if draw:
                #(BGR- Red = (0,0,255), thickness- 3)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3) #draw lines for highlighting rope detected to red 

        
        x_mean = int(np.mean(xs))
        rope_side = (
            "left" if x_mean < w * 0.33 else
            "right" if x_mean > w * 0.66 else
            "center"
        )

        rope_info = {
            "x_center": x_mean,
            "position": rope_side,
            "line_count": len(lines),
            "pixel_ratio": round(pixel_ratio, 4)
        }

        return True, rope_info, overlay

    
