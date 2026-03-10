import cv2
import numpy as np


class RopeDetector:
    def __init__(self):
        self.min_pixel_ratio = 0.003
        self.min_edge_ratio = 0.001
        self.min_lines = 4

        # ----- Temporal Stability -----
        self.detection_counter = 0
        self.required_stable_frames = 5

    def detect(self, frame, hsv_lower, hsv_upper, draw=True):

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # ---------------- HSV MASK ----------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        pixel_ratio = np.count_nonzero(mask) / (h * w)
        if pixel_ratio < self.min_pixel_ratio:
            self.detection_counter = 0
            return False, None, overlay

        # ---------------- EDGE DETECTION ----------------
        edges = cv2.Canny(mask, 50, 150)

        edge_ratio = np.count_nonzero(edges) / (h * w)
        if edge_ratio < self.min_edge_ratio:
            self.detection_counter = 0
            return False, None, overlay

        # ---------------- HOUGH TRANSFORM ----------------
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

        # ---------------- ORIENTATION FILTER ----------------
        candidate_lines = []

        for l in lines:
            x1, y1, x2, y2 = l[0]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # Accept vertical OR horizontal rope
            if dy > dx * 2 or dx > dy * 2:
                candidate_lines.append((x1, y1, x2, y2))

        if len(candidate_lines) < self.min_lines:
            self.detection_counter = 0
            return False, None, overlay

        # ---------------- TEMPORAL STABILITY ----------------
        self.detection_counter += 1

        if self.detection_counter < self.required_stable_frames:
            return False, None, overlay

        # ---------------- CENTERLINE AVERAGING ----------------
        xs = []
        ys = []

        for (x1, y1, x2, y2) in candidate_lines:
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        x_mean = int(np.mean(xs))
        y_mean = int(np.mean(ys))

        dx_total = sum(abs(x2 - x1) for (x1, y1, x2, y2) in candidate_lines)
        dy_total = sum(abs(y2 - y1) for (x1, y1, x2, y2) in candidate_lines)

        # Determine rope orientation
        if dy_total > dx_total:
            orientation = "vertical"
            line = (x_mean, 0, x_mean, h)
        else:
            orientation = "horizontal"
            line = (0, y_mean, w, y_mean)

        # ---------------- DRAW FINAL STABLE LINE ----------------
        if draw:
            x1, y1, x2, y2 = line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # ---------------- POSITION ESTIMATION ----------------
        rope_side = (
            "left" if x_mean < w * 0.33 else
            "right" if x_mean > w * 0.66 else
            "center"
        )

        rope_info = {
            "x_center": x_mean,
            "y_center": y_mean,
            "orientation": orientation,
            "position": rope_side,
            "line_count": len(candidate_lines),
            "pixel_ratio": round(pixel_ratio, 4)
        }

        return True, rope_info, overlay