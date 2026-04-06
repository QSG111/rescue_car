import time

import cv2

from camera import CameraReader
from decision import DecisionMaker
from detect import Detector
from escape import EscapeController
from path import PathAnalyzer
from quality import QualityJudge
from serial import SerialController


class RescueExecutor:
    """Execute timed grab and release sequences."""

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

    def _start_sequence(self, name, now):
        self.current_index = 0
        self.step_start_time = 0.0
        self.active = True
        self.last_trigger_time = now
        self.last_sequence_name = name
        self.just_finished = None

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

    def _execute(self, channel, action):
        if channel == "chassis":
            self.serial_controller.send_chassis(action)
        elif channel == "sync":
            self.serial_controller.send_sync(action)
        elif channel == "left_gripper":
            self.serial_controller.send_left_gripper(action)
        elif channel == "right_gripper":
            self.serial_controller.send_right_gripper(action)


class CarryManager:
    """Track gripper occupancy and enforce transport rules."""

    def __init__(self, max_carry_count=2):
        self.max_carry_count = max(1, int(max_carry_count))
        self.slots = {"LEFT": None, "RIGHT": None}

    def total_count(self):
        return sum(1 for item in self.slots.values() if item is not None)

    def occupied_sides(self):
        return [side for side, item in self.slots.items() if item is not None]

    def free_sides(self):
        return [side for side, item in self.slots.items() if item is None]

    def has_danger(self):
        return any(item == "danger" for item in self.slots.values() if item is not None)

    def can_accept(self, target_type):
        if target_type is None:
            return False
        if self.total_count() >= self.max_carry_count:
            return False
        if not self.free_sides():
            return False
        if target_type == "danger":
            return self.total_count() == 0
        return not self.has_danger()

    def choose_side(self, preferred_side):
        if preferred_side in self.free_sides():
            return preferred_side
        fallback_side = "RIGHT" if preferred_side == "LEFT" else "LEFT"
        if fallback_side in self.free_sides():
            return fallback_side
        return None

    def register_grab(self, side, target_type):
        if side not in self.slots or self.slots[side] is not None:
            return False
        self.slots[side] = target_type
        return True

    def should_deliver_now(self):
        if self.total_count() == 0:
            return False
        if self.has_danger():
            return True
        return self.total_count() >= self.max_carry_count or not self.free_sides()

    def clear(self):
        self.slots = {"LEFT": None, "RIGHT": None}


def initialize_servos(serial_controller):
    serial_controller.send_camera("SEARCH")
    serial_controller.send_sync("MID")
    serial_controller.send_left_gripper("OPEN")
    serial_controller.send_right_gripper("OPEN")


def empty_detection_result():
    return {
        "color_targets": [],
        "safe_zone": None,
        "yolo": [],
        "target": None,
    }


def decide_delivery(safe_zone, path_result, frame_width):
    if safe_zone is None:
        if path_result["path_clear"]:
            return "FORWARD", "search_safe_zone"
        return path_result["best_direction"], "follow_path_to_zone"

    offset_x = safe_zone["center_x"] - frame_width // 2
    if safe_zone["area"] >= 22000:
        return "STOP", "safe_zone_reached"
    if offset_x < -50:
        return "LEFT", "safe_zone_left"
    if offset_x > 50:
        return "RIGHT", "safe_zone_right"
    return "FORWARD", "approach_safe_zone"


def confirm_grab_success(detection_result, carrying_side):
    target = detection_result.get("target")
    if target is None:
        return True

    frame_center_x = 320
    near_center = abs(target["center_x"] - frame_center_x) < 90
    large_target = target["area"] > 12000
    same_side = (carrying_side == "LEFT" and target["center_x"] < frame_center_x) or (
        carrying_side == "RIGHT" and target["center_x"] >= frame_center_x
    )
    return not (near_center and large_target and same_side)


def confirm_release_success(detection_result):
    return detection_result.get("safe_zone") is not None


def draw_debug(
    frame,
    detection_result,
    path_result,
    quality_result,
    decision_result,
    escaped,
    rescue_active,
    team_color,
    phase,
    carrying_count,
    occupied_sides,
):
    h, w = frame.shape[:2]

    target = detection_result["target"]
    if target is not None:
        x, y, bw, bh = target["bbox"]
        color = (0, 255, 0) if target["source"] == "color" else (255, 255, 0)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(
            frame,
            f'{target["source"]}:{target["label"]}',
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    safe_zone = detection_result.get("safe_zone")
    if safe_zone is not None:
        x, y, bw, bh = safe_zone["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
        cv2.putText(
            frame,
            safe_zone["label"],
            (x, min(h - 10, y + bh + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    path_mask_small = cv2.resize(path_result["mask"], (w // 4, h // 4))
    path_mask_bgr = cv2.cvtColor(path_mask_small, cv2.COLOR_GRAY2BGR)
    frame[0 : h // 4, 0 : w // 4] = path_mask_bgr

    target_area = 0.0 if target is None else target["area"]
    zone_area = 0.0 if safe_zone is None else safe_zone["area"]
    status_lines = [
        f"Phase: {phase}",
        f"CMD: {decision_result['command']}",
        f"Reason: {decision_result['reason']}",
        f"Path: {path_result['best_direction']}",
        f"Target area: {target_area:.0f}",
        f"Zone area: {zone_area:.0f}",
        f"Carry count: {carrying_count}",
        f"Carry sides: {','.join(occupied_sides) if occupied_sides else '-'}",
        f"Team: {team_color}",
        f"Target: {decision_result['target_label']}/{decision_result['target_type']}",
        f"Grip side: {decision_result['gripper_side']}",
        f"Need grab: {decision_result['should_grab']}",
        f"Quality: {'GOOD' if quality_result['is_good'] else 'BAD'}",
        f"Escape: {'ON' if escaped else 'OFF'}",
        f"ActionSeq: {'RUN' if rescue_active else 'IDLE'}",
    ]

    y = 25
    for line in status_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
        y += 24


def main():
    team_color = "red"
    max_carry_count = 2
    confirm_frames_required = 2

    camera = CameraReader(camera_id=0, width=640, height=480)
    detector = Detector(
        yolo_model_path=None,
        yolo_stride=5,
        yolo_classes=["person", "sports ball", "bottle"],
    )
    path_analyzer = PathAnalyzer()
    quality_judge = QualityJudge()
    decision_maker = DecisionMaker(center_tolerance=50, stop_area=25000, grab_area=18000)
    escape_controller = EscapeController(stop_threshold=12)
    serial_controller = SerialController(port="COM3", baudrate=115200, enable_serial=False)
    rescue_executor = RescueExecutor(serial_controller)
    carry_manager = CarryManager(max_carry_count=max_carry_count)

    phase = "SEARCH"
    action_side = None
    action_target_type = None
    grab_confirm_count = 0
    release_confirm_count = 0

    if not camera.open():
        print("[Main] Failed to open camera.")
        return

    serial_controller.open()
    initialize_servos(serial_controller)
    print("[Main] System started. Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("[Main] Failed to read frame.")
                break

            quality_result = quality_judge.assess(frame)
            path_result = path_analyzer.analyze(frame)

            if quality_result["is_good"]:
                detection_result = detector.detect(frame, team_color=team_color)
            else:
                detection_result = empty_detection_result()

            if phase == "SEARCH":
                decision_result = decision_maker.decide(
                    detection_result=detection_result,
                    path_result=path_result,
                    quality_result=quality_result,
                    frame_width=frame.shape[1],
                )
            elif phase == "GRAB_CONFIRM":
                decision_result = {
                    "command": "STOP",
                    "reason": "confirm_grab",
                    "gripper_side": action_side,
                    "should_grab": False,
                    "target_type": action_target_type,
                    "target_label": "confirm_grab",
                }
            elif phase == "DELIVER":
                command, reason = decide_delivery(
                    detection_result.get("safe_zone"),
                    path_result,
                    frame.shape[1],
                )
                decision_result = {
                    "command": command,
                    "reason": reason,
                    "gripper_side": action_side,
                    "should_grab": False,
                    "target_type": action_target_type,
                    "target_label": "carry_to_safe_zone",
                }
            else:
                decision_result = {
                    "command": "STOP",
                    "reason": "confirm_release",
                    "gripper_side": action_side,
                    "should_grab": False,
                    "target_type": action_target_type,
                    "target_label": "confirm_release",
                }

            if phase == "SEARCH":
                target_type = decision_result["target_type"]
                preferred_side = decision_result["gripper_side"]
                actual_side = carry_manager.choose_side(preferred_side)

                if target_type is not None and not carry_manager.can_accept(target_type):
                    decision_result["should_grab"] = False
                    decision_result["reason"] = "carry_rule_blocked"

                if decision_result["should_grab"]:
                    decision_result["gripper_side"] = actual_side
                    if actual_side is None:
                        decision_result["should_grab"] = False
                        decision_result["reason"] = "gripper_busy"

                if carry_manager.total_count() > 0 and carry_manager.should_deliver_now():
                    decision_result["command"] = "STOP"
                    decision_result["reason"] = "switch_to_deliver"
                    decision_result["should_grab"] = False
                    phase = "DELIVER"

            command = decision_result["command"]
            if phase == "SEARCH":
                command, escaped = escape_controller.check_and_override(command, path_result)
            else:
                escaped = False
            decision_result["command"] = command

            if not rescue_executor.active:
                serial_controller.send_chassis(command)

            if (
                phase == "SEARCH"
                and decision_result["should_grab"]
                and decision_result["gripper_side"] is not None
            ):
                if rescue_executor.trigger_grab(decision_result["gripper_side"], decision_result["target_type"]):
                    action_side = decision_result["gripper_side"]
                    action_target_type = decision_result["target_type"]

            if phase == "DELIVER" and carry_manager.total_count() > 0:
                if not rescue_executor.active and decision_result["reason"] == "safe_zone_reached":
                    rescue_executor.trigger_release(carry_manager.occupied_sides())

            rescue_active = rescue_executor.update()

            if rescue_executor.just_finished == "grab":
                phase = "GRAB_CONFIRM"
                grab_confirm_count = 0
                rescue_executor.just_finished = None
            elif rescue_executor.just_finished == "release":
                phase = "RELEASE_CONFIRM"
                release_confirm_count = 0
                rescue_executor.just_finished = None

            if phase == "GRAB_CONFIRM" and quality_result["is_good"]:
                if confirm_grab_success(detection_result, action_side):
                    grab_confirm_count += 1
                else:
                    grab_confirm_count = 0

                if grab_confirm_count >= confirm_frames_required:
                    if carry_manager.register_grab(action_side, action_target_type):
                        if carry_manager.should_deliver_now():
                            phase = "DELIVER"
                        else:
                            phase = "SEARCH"
                    else:
                        phase = "SEARCH"
                    action_side = None
                    action_target_type = None

            if phase == "RELEASE_CONFIRM" and quality_result["is_good"]:
                if confirm_release_success(detection_result):
                    release_confirm_count += 1
                else:
                    release_confirm_count = 0

                if release_confirm_count >= confirm_frames_required:
                    phase = "SEARCH"
                    carry_manager.clear()
                    action_side = None
                    action_target_type = None

            draw_debug(
                frame=frame,
                detection_result=detection_result,
                path_result=path_result,
                quality_result=quality_result,
                decision_result=decision_result,
                escaped=escaped,
                rescue_active=rescue_active,
                team_color=team_color,
                phase=phase,
                carrying_count=carry_manager.total_count(),
                occupied_sides=carry_manager.occupied_sides(),
            )
            cv2.imshow("SRP Simplified", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        camera.release()
        serial_controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
