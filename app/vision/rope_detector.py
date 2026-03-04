import cv2
import numpy as np


class RopeDetector:
    def __init__(self):
        self.min_pixel_ratio = 0.005
        self.min_edge_ratio = 0.002
        self.min_lines = 3
        self.required_stable_frames = 3
        self.detection_counter = 0

    def detect(self, roi, hsv_lower, hsv_upper):
        """
        roi: cropped region from full frame
        returns:
            rope_found (bool)
            vertical_lines (list)
        """

        h, w = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        pixel_ratio = np.count_nonzero(mask) / (h * w)
        if pixel_ratio < self.min_pixel_ratio:
            self.detection_counter = 0
            return False, []

        edges = cv2.Canny(mask, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (h * w)

        if edge_ratio < self.min_edge_ratio:
            self.detection_counter = 0
            return False, []

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=50,
            maxLineGap=15
        )

        if lines is None:
            self.detection_counter = 0
            return False, []

        vertical_lines = []

        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dy > dx * 2:
                vertical_lines.append((x1, y1, x2, y2))

        if len(vertical_lines) < self.min_lines:
            self.detection_counter = 0
            return False, []

        self.detection_counter += 1

        if self.detection_counter < self.required_stable_frames:
            return False, []

        return True, vertical_lines