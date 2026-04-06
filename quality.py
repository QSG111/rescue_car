import cv2
import numpy as np


class QualityJudge:
    """画面质量判断模块。只做最基本的清晰度和亮度检测。"""

    def assess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))

        is_blurry = blur_score < 60
        is_too_dark = brightness < 45
        is_too_bright = brightness > 220
        is_good = not (is_blurry or is_too_dark or is_too_bright)

        return {
            "is_good": is_good,
            "blur_score": float(blur_score),
            "brightness": brightness,
            "is_blurry": is_blurry,
            "is_too_dark": is_too_dark,
            "is_too_bright": is_too_bright,
        }
