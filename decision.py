class DecisionMaker:
    """Direction-only controller for search and delivery phases."""

    def __init__(self, center_tolerance=50, stop_area=25000, safe_zone_area=22000):
        self.center_tolerance = int(center_tolerance)
        self.stop_area = int(stop_area)
        self.safe_zone_area = int(safe_zone_area)

    def decide_search(self, target, path_result, frame_width):
        result = {
            "command": "STOP",
            "reason": "search_idle",
        }

        if target is not None:
            frame_center_x = frame_width // 2
            offset_x = target["center_x"] - frame_center_x

            if target.get("source") == "color" and target["area"] >= self.stop_area:
                result["command"] = "STOP"
                result["reason"] = "target_close"
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
            return result

        if not path_result["path_clear"]:
            result["reason"] = "path_blocked"
            return result

        result["command"] = path_result["best_direction"]
        result["reason"] = "follow_open_path"
        return result

    def decide_delivery(self, safe_zone, path_result, frame_width):
        result = {
            "command": "STOP",
            "reason": "deliver_idle",
        }

        if safe_zone is None:
            if path_result["path_clear"]:
                result["command"] = "FORWARD"
                result["reason"] = "search_safe_zone"
            else:
                result["command"] = path_result["best_direction"]
                result["reason"] = "follow_path_to_zone"
            return result

        offset_x = safe_zone["center_x"] - frame_width // 2
        if safe_zone["area"] >= self.safe_zone_area:
            result["command"] = "STOP"
            result["reason"] = "safe_zone_reached"
            return result

        if offset_x < -self.center_tolerance:
            result["command"] = "LEFT"
            result["reason"] = "safe_zone_left"
            return result

        if offset_x > self.center_tolerance:
            result["command"] = "RIGHT"
            result["reason"] = "safe_zone_right"
            return result

        result["command"] = "FORWARD"
        result["reason"] = "approach_safe_zone"
        return result
