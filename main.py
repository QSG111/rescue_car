import time

import cv2

from camera import CameraReader
from decision import DecisionMaker
from detect import Detector
from escape import EscapeController
from executor import ActionExecutor
from path import PathAnalyzer
from quality import QualityJudge
from serial import SerialController

SEARCH = "SEARCH"
GRAB = "GRAB"
GRAB_CONFIRM = "GRAB_CONFIRM"
DELIVER = "DELIVER"


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


def pick_grab_candidate(detection_result):
    target = detection_result.get("color_target")
    if target is None:
        return None
    if target.get("target_type") not in {"normal", "core", "danger"}:
        return None
    return target


def choose_gripper_side(target, frame_width):
    return "LEFT" if target["center_x"] < frame_width // 2 else "RIGHT"


def is_grab_ready(target, frame_width, center_tolerance, grab_area):
    if target is None:
        return False
    offset_x = abs(target["center_x"] - frame_width // 2)
    return offset_x <= center_tolerance and target["area"] >= grab_area


def make_action_context(target, side):
    return {
        "side": side,
        "target_label": target["label"],
        "target_type": target["target_type"],
        "target_center_x": target["center_x"],
        "target_area": target["area"],
    }


def confirm_grab_success(detection_result, action_context, frame_width):
    frame_center_x = frame_width // 2
    expected_side = action_context["side"]
    expected_label = action_context["target_label"]
    expected_type = action_context["target_type"]
    expected_center_x = action_context["target_center_x"]
    expected_area = action_context["target_area"]

    for target in detection_result.get("color_targets", []):
        same_side = (
            expected_side == "LEFT" and target["center_x"] < frame_center_x
        ) or (
            expected_side == "RIGHT" and target["center_x"] >= frame_center_x
        )
        if not same_side:
            continue

        similar_label = target["label"] == expected_label
        similar_type = target["target_type"] == expected_type
        near_previous_position = abs(target["center_x"] - expected_center_x) <= 140
        still_large = target["area"] >= expected_area * 0.35

        if (similar_label or similar_type) and (near_previous_position or still_large):
            return False

    return True


def draw_debug(
    frame,
    detection_result,
    path_result,
    quality_result,
    decision_result,
    escaped,
    executor_active,
    team_color,
    phase,
    carrying_count,
    occupied_sides,
    grab_confirm_count,
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
        f"Grab confirm: {grab_confirm_count}",
        f"Quality: {'GOOD' if quality_result['is_good'] else 'BAD'}",
        f"YOLO: {'ON' if detection_result['yolo_enabled'] else 'OFF'}",
        f"Escape: {'ON' if escaped else 'OFF'}",
        f"Executor: {'RUN' if executor_active else 'IDLE'}",
    ]

    y = 25
    for line in status_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
        y += 24


def main():
    team_color = "red"
    max_carry_count = 2
    grab_center_tolerance = 50
    grab_area = 18000
    confirm_frames_required = 3
    confirm_timeout_seconds = 1.2

    camera = CameraReader(camera_id=0, width=640, height=480)
    detector = Detector(
        yolo_model_path=None,
        yolo_stride=5,
        yolo_classes=["person", "sports ball", "bottle"],
    )
    path_analyzer = PathAnalyzer()
    quality_judge = QualityJudge()
    decision_maker = DecisionMaker(center_tolerance=50, stop_area=25000, safe_zone_area=22000)
    escape_controller = EscapeController(stop_threshold=12)
    serial_controller = SerialController(port="COM3", baudrate=115200, enable_serial=False)
    executor = ActionExecutor(serial_controller)
    carry_manager = CarryManager(max_carry_count=max_carry_count)

    phase = SEARCH
    action_context = None
    grab_confirm_count = 0
    grab_confirm_start_time = 0.0

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
            detection_result = detector.detect(
                frame,
                team_color=team_color,
                allow_yolo=quality_result["is_good"],
            )

            frame_width = frame.shape[1]
            escaped = False
            should_release = False
            decision_result = {
                "command": "STOP",
                "reason": "idle",
                "gripper_side": action_context["side"] if action_context else None,
                "target_type": action_context["target_type"] if action_context else None,
                "target_label": action_context["target_label"] if action_context else None,
            }

            if phase == SEARCH:
                if carry_manager.total_count() > 0 and carry_manager.should_deliver_now():
                    phase = DELIVER
                    decision_result["reason"] = "switch_to_deliver"
                else:
                    target = detection_result.get("target")
                    direction = decision_maker.decide_search(target, path_result, frame_width)
                    decision_result.update(direction)
                    if target is not None:
                        decision_result["target_type"] = target.get("target_type")
                        decision_result["target_label"] = target.get("label")
                        decision_result["gripper_side"] = choose_gripper_side(target, frame_width)

                    grab_target = pick_grab_candidate(detection_result)
                    if grab_target is not None:
                        preferred_side = choose_gripper_side(grab_target, frame_width)
                        actual_side = carry_manager.choose_side(preferred_side)
                        if (
                            actual_side is not None
                            and carry_manager.can_accept(grab_target["target_type"])
                            and is_grab_ready(
                                grab_target,
                                frame_width,
                                center_tolerance=grab_center_tolerance,
                                grab_area=grab_area,
                            )
                        ):
                            action_context = make_action_context(grab_target, actual_side)
                            phase = GRAB
                            decision_result["command"] = "STOP"
                            decision_result["reason"] = "switch_to_grab"
                            decision_result["target_type"] = grab_target["target_type"]
                            decision_result["target_label"] = grab_target["label"]
                            decision_result["gripper_side"] = actual_side

            if phase == GRAB:
                decision_result = {
                    "command": "STOP",
                    "reason": "execute_grab",
                    "gripper_side": action_context["side"] if action_context else None,
                    "target_type": action_context["target_type"] if action_context else None,
                    "target_label": action_context["target_label"] if action_context else None,
                }

            if phase == GRAB_CONFIRM:
                decision_result = {
                    "command": "STOP",
                    "reason": "confirm_grab",
                    "gripper_side": action_context["side"] if action_context else None,
                    "target_type": action_context["target_type"] if action_context else None,
                    "target_label": action_context["target_label"] if action_context else None,
                }

            if phase == DELIVER:
                safe_zone = detection_result.get("safe_zone")
                direction = decision_maker.decide_delivery(safe_zone, path_result, frame_width)
                decision_result = {
                    "command": direction["command"],
                    "reason": direction["reason"],
                    "gripper_side": action_context["side"] if action_context else None,
                    "target_type": action_context["target_type"] if action_context else None,
                    "target_label": "carry_to_safe_zone",
                }
                if carry_manager.total_count() == 0:
                    phase = SEARCH
                    decision_result["command"] = "STOP"
                    decision_result["reason"] = "empty_delivery"
                elif direction["reason"] == "safe_zone_reached":
                    should_release = True

            command = decision_result["command"]
            if phase == SEARCH:
                command, escaped = escape_controller.check_and_override(command, path_result)
                decision_result["command"] = command

            if not executor.active:
                serial_controller.send_chassis(command)

            if phase == GRAB and action_context is not None and not executor.active:
                if executor.trigger_grab(action_context["side"], action_context["target_type"]):
                    decision_result["reason"] = "grab_started"

            if phase == DELIVER and should_release and not executor.active:
                if executor.trigger_release(carry_manager.occupied_sides()):
                    decision_result["reason"] = "release_started"

            executor_active = executor.update()

            if executor.just_finished == "grab":
                phase = GRAB_CONFIRM
                grab_confirm_count = 0
                grab_confirm_start_time = time.time()
                executor.just_finished = None
            elif executor.just_finished == "release":
                carry_manager.clear()
                action_context = None
                phase = SEARCH
                grab_confirm_count = 0
                executor.just_finished = None

            if phase == GRAB_CONFIRM and action_context is not None:
                if confirm_grab_success(detection_result, action_context, frame_width):
                    grab_confirm_count += 1
                else:
                    grab_confirm_count = 0

                if grab_confirm_count >= confirm_frames_required:
                    if carry_manager.register_grab(action_context["side"], action_context["target_type"]):
                        phase = DELIVER if carry_manager.should_deliver_now() else SEARCH
                    else:
                        phase = SEARCH
                    action_context = None
                    grab_confirm_count = 0
                elif time.time() - grab_confirm_start_time >= confirm_timeout_seconds:
                    phase = SEARCH
                    action_context = None
                    grab_confirm_count = 0

            draw_debug(
                frame=frame,
                detection_result=detection_result,
                path_result=path_result,
                quality_result=quality_result,
                decision_result=decision_result,
                escaped=escaped,
                executor_active=executor_active,
                team_color=team_color,
                phase=phase,
                carrying_count=carry_manager.total_count(),
                occupied_sides=carry_manager.occupied_sides(),
                grab_confirm_count=grab_confirm_count,
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
