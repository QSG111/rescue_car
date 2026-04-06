import time


class ActionExecutor:
    """Execute timed grab and release sequences only."""

    def __init__(self, serial_controller):
        self.serial_controller = serial_controller
        self.sequence = []
        self.current_index = 0
        self.step_start_time = 0.0
        self.active = False
        self.last_trigger_time = 0.0
        self.cooldown = 2.0
        self.last_sequence_name = None
        self.just_finished = None

    def trigger_grab(self, side, target_type):
        now = time.time()
        if self.active or side not in ("LEFT", "RIGHT"):
            return False
        if now - self.last_trigger_time < self.cooldown:
            return False

        gripper_channel = "left_gripper" if side == "LEFT" else "right_gripper"
        retreat_action = "BACK_SLOW"
        retreat_time = 0.50
        if target_type == "danger":
            retreat_action = "BACKWARD"
            retreat_time = 0.35

        self.sequence = [
            ("chassis", "STOP", 0.20),
            ("sync", "DOWN", 0.50),
            (gripper_channel, "CLOSE", 0.50),
            ("sync", "MID", 0.40),
            ("chassis", retreat_action, retreat_time),
            ("chassis", "STOP", 0.20),
        ]
        self._start_sequence("grab", now)
        return True

    def trigger_release(self, sides):
        now = time.time()
        if self.active:
            return False
        if now - self.last_trigger_time < 0.8:
            return False

        if isinstance(sides, str):
            sides = [sides]

        release_channels = []
        for side in sides:
            if side == "LEFT":
                release_channels.append("left_gripper")
            elif side == "RIGHT":
                release_channels.append("right_gripper")

        if not release_channels:
            return False

        self.sequence = [
            ("chassis", "STOP", 0.20),
            ("sync", "DOWN", 0.45),
        ]
        for channel in release_channels:
            self.sequence.append((channel, "OPEN", 0.35))
        self.sequence.extend(
            [
                ("sync", "MID", 0.35),
                ("chassis", "BACK_SLOW", 0.35),
                ("chassis", "STOP", 0.20),
            ]
        )
        self._start_sequence("release", now)
        return True

    def update(self):
        self.just_finished = None
        if not self.active:
            return False

        now = time.time()
        if self.current_index >= len(self.sequence):
            self.active = False
            self.just_finished = self.last_sequence_name
            return False

        channel, action, duration = self.sequence[self.current_index]
        if self.step_start_time == 0.0:
            self._execute(channel, action)
            self.step_start_time = now
            return True

        if now - self.step_start_time >= duration:
            self.current_index += 1
            self.step_start_time = 0.0
            if self.current_index >= len(self.sequence):
                self.active = False
                self.just_finished = self.last_sequence_name
                return False
            next_channel, next_action, _ = self.sequence[self.current_index]
            self._execute(next_channel, next_action)
            self.step_start_time = now

        return True

    def _start_sequence(self, name, now):
        self.current_index = 0
        self.step_start_time = 0.0
        self.active = True
        self.last_trigger_time = now
        self.last_sequence_name = name
        self.just_finished = None

    def _execute(self, channel, action):
        if channel == "chassis":
            self.serial_controller.send_chassis(action)
        elif channel == "sync":
            self.serial_controller.send_sync(action)
        elif channel == "left_gripper":
            self.serial_controller.send_left_gripper(action)
        elif channel == "right_gripper":
            self.serial_controller.send_right_gripper(action)
