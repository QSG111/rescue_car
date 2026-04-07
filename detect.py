import os

import cv2
import numpy as np


class Detector:
    """
    以颜色检测为主，辅以低频 YOLO 检测。
    颜色检测始终可用，YOLO 是否启用由画面质量控制。
    """

    def __init__(self, yolo_model_path=None, yolo_stride=5, yolo_classes=None):
        self.frame_index = 0
        self.yolo_stride = max(1, int(yolo_stride))
        self.yolo_classes = yolo_classes or ["person", "sports ball", "bottle"]
        self.last_yolo_result = []
        self.yolo_model = self._load_yolo_model(yolo_model_path)
        self.start_delivered = False
        self.yellow_picked_count = 0
        self.normal_picked_count = 0
        self.current_load = 0
        self.current_has_yellow = False

    def _load_yolo_model(self, yolo_model_path):
        if not yolo_model_path:
            return None
        if not os.path.exists(yolo_model_path):
            print(f"[Detector] 未找到 YOLO 权重文件: {yolo_model_path}")
            return None
        try:
            from ultralytics import YOLO

            print(f"[Detector] 正在加载 YOLO 权重: {yolo_model_path}")
            return YOLO(yolo_model_path)
        except Exception as exc:
            print(f"[Detector] YOLO 初始化失败: {exc}")
            return None

    def detect(self, frame, team_color="red", allow_yolo=True):
        self.frame_index += 1

        color_targets = self._detect_color_targets(frame, team_color)
        color_target = self._pick_best_target(color_targets, frame.shape)
        safe_zone = self._detect_safe_zone(frame, team_color)
        opponent_color = "blue" if team_color == "red" else "red"
        opponent_safe_zone = self._detect_safe_zone(frame, opponent_color)

        yolo_enabled = bool(allow_yolo and self.yolo_model is not None)
        yolo_result = self._detect_yolo(frame, enabled=yolo_enabled)
        yolo_target = self._pick_best_target(yolo_result, frame.shape)

        target = color_target if color_target is not None else yolo_target

        return {
            "color_targets": color_targets,
            "safe_zone": safe_zone,
            "opponent_safe_zone": opponent_safe_zone,
            "yolo": yolo_result,
            "target": target,
            "color_target": color_target,
            "yolo_target": yolo_target,
            "yolo_enabled": yolo_enabled,
        }

    def can_target_label(self, label, team_color):
        own_ball_label = f"{team_color}_ball"
        opponent_ball_label = "blue_ball" if team_color == "red" else "red_ball"

        if self.current_load >= 3:
            return False
        if self.current_has_yellow:
            return False
        if label == opponent_ball_label:
            return False
        if not self.start_delivered:
            return label == own_ball_label
        if label == "yellow_ball":
            return self.current_load == 0 and self.yellow_picked_count < 2 and self.normal_picked_count > 0
        return label in {own_ball_label, "black_ball"}

    def should_force_deliver(self):
        return (not self.start_delivered) or self.current_has_yellow

    def register_pick_result(self, label):
        self.current_load += 1
        if label == "yellow_ball":
            self.yellow_picked_count += 1
            self.current_has_yellow = True
            return
        self.normal_picked_count += 1

    def register_delivery_complete(self):
        if self.current_load > 0:
            self.start_delivered = True
        self.current_load = 0
        self.current_has_yellow = False

    def _detect_color_targets(self, frame, team_color):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        color_specs = [
            {
                "name": "red_ball",
                "type": "normal",
                "priority": 100,
                "ranges": [
                    (np.array([0, 120, 70]), np.array([10, 255, 255])),
                    (np.array([160, 120, 70]), np.array([180, 255, 255])),
                ],
            },
            {
                "name": "blue_ball",
                "type": "normal",
                "priority": 100,
                "ranges": [
                    (np.array([100, 100, 60]), np.array([140, 255, 255])),
                ],
            },
            {
                "name": "black_ball",
                "type": "core",
                "priority": 80,
                "ranges": [
                    (np.array([0, 0, 0]), np.array([180, 255, 55])),
                ],
            },
            {
                "name": "yellow_ball",
                "type": "danger",
                "priority": 120,
                "ranges": [
                    (np.array([18, 90, 90]), np.array([38, 255, 255])),
                ],
            },
        ]

        candidates = []
        for spec in color_specs:
            if not self.can_target_label(spec["name"], team_color):
                continue

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
        if color_name == "black_ball":
            return 700
        return 500

    def _build_team_mask(self, hsv_frame, team_color):
        if team_color == "red":
            mask_low = cv2.inRange(
                hsv_frame,
                np.array([0, 100, 60], dtype=np.uint8),
                np.array([10, 255, 255], dtype=np.uint8),
            )
            mask_high = cv2.inRange(
                hsv_frame,
                np.array([170, 100, 60], dtype=np.uint8),
                np.array([180, 255, 255], dtype=np.uint8),
            )
            return cv2.bitwise_or(mask_low, mask_high)

        return cv2.inRange(
            hsv_frame,
            np.array([100, 100, 60], dtype=np.uint8),
            np.array([140, 255, 255], dtype=np.uint8),
        )

    def find_safety_zone(self, frame, team_color):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dilate_kernel = np.ones((5, 5), np.uint8)
        inner_kernel = np.ones((9, 9), np.uint8)

        purple_lower = np.array([120, 40, 40], dtype=np.uint8)
        purple_upper = np.array([160, 255, 255], dtype=np.uint8)
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        purple_mask = cv2.dilate(purple_mask, dilate_kernel, iterations=2)

        contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"found": False}

        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area < 5000:
            return {"found": False}

        x, y, w, h = cv2.boundingRect(hull)
        roi_hsv = hsv[y : y + h, x : x + w]
        if roi_hsv.size == 0:
            return {"found": False}

        hull_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
        inner_mask = hull_mask[y : y + h, x : x + w]
        inner_mask = cv2.erode(inner_mask, inner_kernel, iterations=1)

        inner_area = cv2.countNonZero(inner_mask)
        if inner_area == 0:
            return {"found": False}

        team_mask = self._build_team_mask(roi_hsv, team_color)
        team_mask = cv2.bitwise_and(team_mask, inner_mask)
        team_area = cv2.countNonZero(team_mask)
        if team_area <= inner_area * 0.1:
            return {"found": False}

        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        return {
            "found": True,
            "center_x": center_x,
            "center_y": center_y,
            "area": hull_area,
            "bbox": (int(x), int(y), int(w), int(h)),
        }

    def _detect_safe_zone(self, frame, team_color):
        zone = self.find_safety_zone(frame, team_color)
        if not zone.get("found"):
            return None

        return {
            "label": f"{team_color}_safe_zone",
            "bbox": zone["bbox"],
            "center_x": int(zone["center_x"]),
            "center_y": int(zone["center_y"]),
            "area": float(zone["area"]),
            "fill_ratio": 1.0,
        }

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
            print(f"[Detector] YOLO 推理失败，保留上一帧结果: {exc}")

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
