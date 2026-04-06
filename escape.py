class EscapeController:
    """
    自动脱困模块。

    简化规则：
    1. 连续多帧停住，认为可能卡住
    2. 触发后依次尝试 LEFT -> RIGHT -> BACK_SLOW -> FORWARD
    """

    def __init__(self, stop_threshold=12):
        self.stop_threshold = stop_threshold
        self.stop_count = 0
        self.escape_index = 0
        self.escape_actions = ["LEFT", "RIGHT", "BACK_SLOW", "FORWARD"]

    def check_and_override(self, current_command, path_result):
        if current_command == "STOP" or not path_result["path_clear"]:
            self.stop_count += 1
        else:
            self.stop_count = 0

        if self.stop_count >= self.stop_threshold:
            action = self.escape_actions[self.escape_index % len(self.escape_actions)]
            self.escape_index += 1
            self.stop_count = 0
            return action, True

        return current_command, False
