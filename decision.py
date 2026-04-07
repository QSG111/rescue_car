class DecisionMaker:
    """搜索与投递阶段的运动决策。"""

    SCAN_FRAMES = 50
    RELOCATE_FRAMES = 20

    def __init__(self, center_tolerance=50, stop_area=25000, safe_zone_area=22000):
        self.center_tolerance = int(center_tolerance)
        self.stop_area = int(stop_area)
        self.safe_zone_area = int(safe_zone_area)
        self.blind_search_timer = 0

    def decide_search(self, target, path_result, frame_width):
        if target is None:
            return self._decide_blind_search(path_result)

        self.reset_blind_search()
        offset_x = self._offset_x(target["center_x"], frame_width)

        if target.get("source") == "color" and target["area"] >= self.stop_area:
            return self._result("STOP", "target_close")
        if offset_x < -self.center_tolerance:
            return self._result("LEFT", "target_on_left")
        if offset_x > self.center_tolerance:
            return self._result("RIGHT", "target_on_right")
        return self._result("FORWARD", "target_ahead")

    def decide_delivery(self, safe_zone, path_result, frame_width, opponent_safe_zone=None):
        if safe_zone is None:
            if opponent_safe_zone is not None:
                offset_x = self._offset_x(opponent_safe_zone["center_x"], frame_width)
                command = "RIGHT" if offset_x <= 0 else "LEFT"
                return self._result(command, "avoid_opponent_safe_zone")
            return self._decide_blind_search(path_result)

        self.reset_blind_search()
        offset_x = self._offset_x(safe_zone["center_x"], frame_width)

        if safe_zone["area"] >= self.safe_zone_area:
            return self._result("STOP", "safe_zone_reached")
        if offset_x < -self.center_tolerance:
            return self._result("LEFT", "safe_zone_left")
        if offset_x > self.center_tolerance:
            return self._result("RIGHT", "safe_zone_right")
        return self._result("FORWARD", "approach_safe_zone")

    def _decide_blind_search(self, path_result):
        self.blind_search_timer += 1
        cycle_frames = self.SCAN_FRAMES + self.RELOCATE_FRAMES
        cycle_pos = self.blind_search_timer % cycle_frames

        if cycle_pos < self.SCAN_FRAMES:
            return self._result("LEFT", "radar_scan")

        if path_result["path_clear"]:
            return self._result(path_result["best_direction"], "relocate")
        return self._result("LEFT", "relocate")

    def reset_blind_search(self):
        self.blind_search_timer = 0

    def _offset_x(self, center_x, frame_width):
        return center_x - frame_width // 2

    def _result(self, command, reason):
        return {
            "command": command,
            "reason": reason,
        }
