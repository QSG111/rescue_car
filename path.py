import cv2
import numpy as np


class PathAnalyzer:
    """路径判断模块。用地面亮区域近似可通行区域，追求简单稳定。"""

    def analyze(self, frame):
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.55): h, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        left = binary[:, : w // 3]
        center = binary[:, w // 3: 2 * w // 3]
        right = binary[:, 2 * w // 3:]

        left_ratio = float(np.count_nonzero(left)) / left.size
        center_ratio = float(np.count_nonzero(center)) / center.size
        right_ratio = float(np.count_nonzero(right)) / right.size

        best_dir = max(
            [("LEFT", left_ratio), ("FORWARD", center_ratio), ("RIGHT", right_ratio)],
            key=lambda item: item[1],
        )

        path_clear = center_ratio > 0.08 or best_dir[1] > 0.12

        return {
            "path_clear": path_clear,
            "best_direction": best_dir[0],
            "ratios": {
                "LEFT": left_ratio,
                "FORWARD": center_ratio,
                "RIGHT": right_ratio,
            },
            "mask": binary,
        }
