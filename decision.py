class DecisionMaker:
    """
    控制决策模块。

    1. 优先跟踪本队普通目标和核心目标
    2. 黄色危险目标只在没有更高优先级目标时处理
    3. 靠近目标时给出夹取建议
    """

    def __init__(self, center_tolerance=50, stop_area=25000, grab_area=18000):
        self.center_tolerance = center_tolerance
        self.stop_area = stop_area
        self.grab_area = grab_area

    def decide(self, detection_result, path_result, quality_result, frame_width):
        result = {
            "command": "STOP",
            "reason": "init",
            "gripper_side": None,
            "should_grab": False,
            "target_type": None,
            "target_label": None,
        }

        if not quality_result["is_good"]:
            result["reason"] = "image_quality_bad"
            return result

        target = detection_result["target"]
        if target is not None:
            result["target_type"] = target.get("target_type")
            result["target_label"] = target.get("label")

            frame_center_x = frame_width // 2
            offset_x = target["center_x"] - frame_center_x

            result["gripper_side"] = "LEFT" if target["center_x"] < frame_center_x else "RIGHT"

            if target["area"] >= self.stop_area:
                result["command"] = "STOP"
                result["reason"] = "target_close"
                result["should_grab"] = True
                return result

            if offset_x < -self.center_tolerance:
                result["command"] = "LEFT"
                result["reason"] = "target_on_left"
                return result

            if offset_x > self.center_tolerance:
                result["command"] = "RIGHT"
                result["reason"] = "target_on_right"
                return result

            result["command"] = "FORWARD"
            result["reason"] = "target_ahead"
            if target["area"] >= self.grab_area:
                result["should_grab"] = True
            return result

        if not path_result["path_clear"]:
            result["reason"] = "path_blocked"
            return result

        result["command"] = path_result["best_direction"]
        result["reason"] = "follow_open_path"
        return result
