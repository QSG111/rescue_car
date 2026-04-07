import time

import cv2
import numpy as np

from decision import DecisionMaker
from escape import EscapeController
from executor import ActionExecutor
from main import (
    DELIVER,
    GRAB,
    GRAB_CONFIRM,
    SEARCH,
    CarryManager,
    choose_gripper_side,
    confirm_grab_success,
    draw_debug,
    initialize_servos,
    is_grab_ready,
    make_action_context,
    pick_grab_candidate,
)
from serial import SerialController


class ScriptedArena:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

    def build_frame(self, detection_result, path_result, phase, team_color, phase_frame):
        frame = np.full((self.height, self.width, 3), (38, 82, 42), dtype=np.uint8)
        cv2.rectangle(frame, (0, int(self.height * 0.55)), (self.width, self.height), (80, 120, 80), -1)
        cv2.line(frame, (self.width // 3, int(self.height * 0.55)), (self.width // 3, self.height), (120, 150, 120), 2)
        cv2.line(frame, (self.width * 2 // 3, int(self.height * 0.55)), (self.width * 2 // 3, self.height), (120, 150, 120), 2)

        target = detection_result.get("color_target")
        if target is not None:
            radius = max(12, int((target["area"] / np.pi) ** 0.5 * 0.25))
            color = (0, 0, 255) if target["label"] == "red_ball" else (255, 0, 0)
            cv2.circle(frame, (target["center_x"], target["center_y"]), radius, color, -1)

        for zone_key, inner_color in (("safe_zone", team_color), ("opponent_safe_zone", "blue" if team_color == "red" else "red")):
            zone = detection_result.get(zone_key)
            if zone is None:
                continue
            x, y, w, h = zone["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 4)
            fill_color = (0, 0, 255) if inner_color == "red" else (255, 0, 0)
            cv2.rectangle(frame, (x + 12, y + 12), (x + w - 12, y + h - 12), fill_color, -1)

        cv2.putText(
            frame,
            f"Scripted simulation frame {phase_frame}",
            (10, self.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return frame

    def build_path_result(self, phase_frame):
        best_direction = "FORWARD" if phase_frame % 90 < 65 else "LEFT"
        mask = np.full((120, self.width), 255, dtype=np.uint8)
        return {
            "path_clear": True,
            "best_direction": best_direction,
            "ratios": {
                "LEFT": 0.25,
                "FORWARD": 0.40,
                "RIGHT": 0.20,
            },
            "mask": mask,
        }

    def build_detection_result(self, phase, phase_frame, team_color):
        result = {
            "color_targets": [],
            "safe_zone": None,
            "opponent_safe_zone": None,
            "yolo": [],
            "target": None,
            "color_target": None,
            "yolo_target": None,
            "yolo_enabled": False,
        }

        own_label = f"{team_color}_ball"

        if phase == SEARCH:
            if phase_frame >= 55:
                target = self._make_ball_target(phase_frame, own_label)
                result["color_targets"] = [target]
                result["color_target"] = target
                result["target"] = target
            return result

        if phase == DELIVER:
            if 35 <= phase_frame < 70:
                result["opponent_safe_zone"] = self._make_zone_target(
                    center_x=160,
                    center_y=190,
                    width=170,
                    height=120,
                    area=16000,
                    label=f'{"blue" if team_color == "red" else "red"}_safe_zone',
                )
            elif phase_frame >= 70:
                result["safe_zone"] = self._make_delivery_zone(phase_frame, team_color)
            return result

        return result

    def _make_ball_target(self, phase_frame, label):
        progress = min(max(phase_frame - 55, 0), 55)
        center_x = int(120 + progress * 3.6)
        center_y = int(300 - progress * 1.4)
        area = float(4500 + progress * 500)
        radius = max(18, int((area / np.pi) ** 0.5))
        return {
            "source": "color",
            "label": label,
            "target_type": "normal",
            "priority": 100,
            "bbox": (center_x - radius, center_y - radius, radius * 2, radius * 2),
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
            "confidence": 1.0,
        }

    def _make_delivery_zone(self, phase_frame, team_color):
        progress = min(max(phase_frame - 70, 0), 45)
        center_x = int(420 - progress * 2.2)
        center_y = int(210 + progress * 0.8)
        width = int(150 + progress * 1.8)
        height = int(100 + progress * 1.2)
        area = float(min(12000 + progress * 280, 25000))
        return self._make_zone_target(
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            area=area,
            label=f"{team_color}_safe_zone",
        )

    def _make_zone_target(self, center_x, center_y, width, height, area, label):
        return {
            "label": label,
            "bbox": (
                int(center_x - width / 2),
                int(center_y - height / 2),
                int(width),
                int(height),
            ),
            "center_x": int(center_x),
            "center_y": int(center_y),
            "area": float(area),
            "fill_ratio": 1.0,
        }


def main():
    team_color = "red"
    max_carry_count = 3
    grab_center_tolerance = 50
    grab_area = 18000
    confirm_frames_required = 3
    confirm_timeout_seconds = 1.2
    fps = 15.0

    arena = ScriptedArena()
    decision_maker = DecisionMaker(center_tolerance=50, stop_area=25000, safe_zone_area=22000)
    escape_controller = EscapeController(stop_threshold=12)
    serial_controller = SerialController(port="COM3", baudrate=115200, enable_serial=False)
    executor = ActionExecutor(serial_controller)
    carry_manager = CarryManager(max_carry_count=max_carry_count)

    phase = SEARCH
    last_phase = phase
    phase_frame = 0
    action_context = None
    grab_confirm_count = 0
    grab_confirm_start_time = 0.0
    completed_deliveries = 0

    serial_controller.open()
    initialize_servos(serial_controller)
    print("[Sim] Scripted match started. Press Q to quit.")

    try:
        while True:
            if phase != last_phase:
                phase_frame = 0
                last_phase = phase
            else:
                phase_frame += 1

            path_result = arena.build_path_result(phase_frame)
            detection_result = arena.build_detection_result(phase, phase_frame, team_color)
            quality_result = {
                "is_good": True,
                "blur_score": 100.0,
                "brightness": 128.0,
                "is_blurry": False,
                "is_too_dark": False,
                "is_too_bright": False,
            }

            frame_width = arena.width
            escaped = False
            should_release = False
            decision_result = {
                "command": "STOP",
                "reason": "idle",
                "gripper_side": action_context["side"] if action_context else None,
                "target_type": action_context["target_type"] if action_context else None,
                "target_label": action_context["target_label"] if action_context else None,
            }
            force_deliver = completed_deliveries == 0

            if phase == SEARCH:
                if carry_manager.total_count() > 0 and carry_manager.should_deliver_now(force=force_deliver):
                    phase = DELIVER
                    decision_maker.reset_blind_search()
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
                opponent_safe_zone = detection_result.get("opponent_safe_zone")
                direction = decision_maker.decide_delivery(
                    safe_zone,
                    path_result,
                    frame_width,
                    opponent_safe_zone=opponent_safe_zone,
                )
                decision_result = {
                    "command": direction["command"],
                    "reason": direction["reason"],
                    "gripper_side": action_context["side"] if action_context else None,
                    "target_type": action_context["target_type"] if action_context else None,
                    "target_label": "carry_to_safe_zone",
                }
                if carry_manager.total_count() == 0:
                    phase = SEARCH
                    decision_maker.reset_blind_search()
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
                decision_maker.reset_blind_search()
                grab_confirm_count = 0
                completed_deliveries += 1
                executor.just_finished = None

            if phase == GRAB_CONFIRM and action_context is not None:
                if confirm_grab_success(detection_result, action_context, frame_width):
                    grab_confirm_count += 1
                else:
                    grab_confirm_count = 0

                if grab_confirm_count >= confirm_frames_required:
                    if carry_manager.register_grab(action_context["side"], action_context["target_type"]):
                        phase = DELIVER if carry_manager.should_deliver_now(force=completed_deliveries == 0) else SEARCH
                        decision_maker.reset_blind_search()
                    else:
                        phase = SEARCH
                        decision_maker.reset_blind_search()
                    action_context = None
                    grab_confirm_count = 0
                elif time.time() - grab_confirm_start_time >= confirm_timeout_seconds:
                    phase = SEARCH
                    decision_maker.reset_blind_search()
                    action_context = None
                    grab_confirm_count = 0

            frame = arena.build_frame(detection_result, path_result, phase, team_color, phase_frame)
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
            cv2.imshow("SRP Scripted Simulation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if completed_deliveries >= 1 and phase == SEARCH and carry_manager.total_count() == 0:
                print("[Sim] One full grab-deliver cycle completed.")
                break

            time.sleep(1.0 / fps)
    finally:
        serial_controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
