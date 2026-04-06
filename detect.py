import os

import cv2
import numpy as np


class Detector:
    """
    Primary color detection with low-frequency YOLO assistance.
    Color detection is always available. YOLO can be gated by image quality.
    """

    def __init__(self, yolo_model_path=None, yolo_stride=5, yolo_classes=None):
        self.frame_index = 0
        self.yolo_stride = max(1, int(yolo_stride))
        self.yolo_classes = yolo_classes or ["person", "sports ball", "bottle"]
        self.last_yolo_result = []
        self.yolo_model = self._load_yolo_model(yolo_model_path)

    def _load_yolo_model(self, yolo_model_path):
        if not yolo_model_path:
            return None
        if not os.path.exists(yolo_model_path):
            print(f"[Detector] YOLO weight missing: {yolo_model_path}")
            return None
        try:
            from ultralytics import YOLO

            print(f"[Detector] Loading YOLO weights: {yolo_model_path}")
            return YOLO(yolo_model_path)
        except Exception as exc:
            print(f"[Detector] YOLO init failed: {exc}")
            return None

    def detect(self, frame, team_color="red", allow_yolo=True):
        self.frame_index += 1

        color_targets = self._detect_color_targets(frame, team_color)
        color_target = self._pick_best_target(color_targets, frame.shape)
        safe_zone = self._detect_safe_zone(frame, team_color)

        yolo_enabled = bool(allow_yolo and self.yolo_model is not None)
        yolo_result = self._detect_yolo(frame, enabled=yolo_enabled)
        yolo_target = self._pick_best_target(yolo_result, frame.shape)

        target = color_target if color_target is not None else yolo_target

        return {
            "color_targets": color_targets,
            "safe_zone": safe_zone,
            "yolo": yolo_result,
            "target": target,
            "color_target": color_target,
            "yolo_target": yolo_target,
            "yolo_enabled": yolo_enabled,
        }

    def _detect_color_targets(self, frame, team_color):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        color_specs = [
            {
                "name": "red",
                "type": "normal",
                "priority": 3 if team_color == "red" else 1,
                "ranges": [
                    (np.array([0, 120, 70]), np.array([10, 255, 255])),
                    (np.array([160, 120, 70]), np.array([180, 255, 255])),
                ],
            },
            {
                "name": "blue",
                "type": "normal",
                "priority": 3 if team_color == "blue" else 1,
                "ranges": [
                    (np.array([100, 100, 60]), np.array([140, 255, 255])),
                ],
            },
            {
                "name": "black",
                "type": "core",
                "priority": 4,
                "ranges": [
                    (np.array([0, 0, 0]), np.array([180, 255, 55])),
                ],
            },
            {
                "name": "yellow",
                "type": "danger",
                "priority": 2,
                "ranges": [
                    (np.array([18, 90, 90]), np.array([38, 255, 255])),
                ],
            },
        ]

        candidates = []
        for spec in color_specs:
            mask = None
            for lower, upper in spec["ranges"]:
                current = cv2.inRange(hsv, lower, upper)
                mask = current if mask is None else cv2.bitwise_or(mask, current)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self._min_area_for_target(spec["name"]):
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                candidates.append(
                    {
                        "source": "color",
                        "label": spec["name"],
                        "target_type": spec["type"],
                        "priority": spec["priority"],
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "center_x": int(x + w / 2),
                        "center_y": int(y + h / 2),
                        "area": float(area),
                        "confidence": 1.0,
                    }
                )

        return candidates

    def _min_area_for_target(self, color_name):
        if color_name == "black":
            return 700
        return 500

    def _detect_safe_zone(self, frame, team_color):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((7, 7), np.uint8)

        if team_color == "red":
            mask1 = cv2.inRange(hsv, np.array([0, 100, 60]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 100, 60]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, np.array([100, 100, 60]), np.array([140, 255, 255]))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1e9
        _, frame_w = frame.shape[:2]
        frame_center_x = frame_w / 2

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = float(w * h)
            if rect_area <= 0:
                continue

            fill_ratio = area / rect_area
            aspect = w / max(h, 1)
            if fill_ratio < 0.45:
                continue
            if not (1.2 <= aspect <= 4.5):
                continue

            score = area - abs((x + w / 2) - frame_center_x) * 1.5
            if score > best_score:
                best_score = score
                best = {
                    "label": f"{team_color}_safe_zone",
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "center_x": int(x + w / 2),
                    "center_y": int(y + h / 2),
                    "area": float(area),
                    "fill_ratio": float(fill_ratio),
                }

        return best

    def _detect_yolo(self, frame, enabled):
        if not enabled:
            return []
        if self.frame_index % self.yolo_stride != 0:
            return self.last_yolo_result

        try:
            results = self.yolo_model(frame, verbose=False)
            detections = []
            names = results[0].names
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = names.get(cls_id, str(cls_id))
                    if label not in self.yolo_classes:
                        continue

                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    detections.append(
                        {
                            "source": "yolo",
                            "label": label,
                            "target_type": "assist",
                            "priority": 0,
                            "bbox": (int(x1), int(y1), w, h),
                            "center_x": int((x1 + x2) / 2),
                            "center_y": int((y1 + y2) / 2),
                            "area": float(max(w * h, 0)),
                            "confidence": conf,
                        }
                    )
            self.last_yolo_result = detections
        except Exception as exc:
            print(f"[Detector] YOLO inference failed, keep previous result: {exc}")

        return self.last_yolo_result

    def _pick_best_target(self, candidates, frame_shape):
        if not candidates:
            return None

        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2

        def score(item):
            dx = abs(item["center_x"] - center_x)
            dy = abs(item["center_y"] - center_y)
            return item["priority"] * 100000 + item["area"] - 2.0 * (dx + dy)

        return max(candidates, key=score)
