import importlib.machinery
import importlib.util
import os
import sys
import time


def _load_pyserial_module():
    """
    当前文件名必须叫 serial.py，会和 pyserial 同名。
    这里手动从其他搜索路径加载真正的 pyserial，避免导入冲突。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = []
    for path in sys.path:
        if not path:
            continue
        if os.path.abspath(path) == current_dir:
            continue
        search_paths.append(path)

    try:
        spec = importlib.machinery.PathFinder.find_spec("serial", search_paths)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        print(f"[Serial] 加载 pyserial 失败: {exc}")
        return None


class SerialController:
    """
    串口通信模块。

    按协议拆成四类：
    1. 底盘运动
    2. 左夹爪
    3. 右夹爪
    4. 其他舵机和握手
    """

    CHASSIS_MAP = {
        "STOP": 0x00,
        "FORWARD": 0x01,
        "BACKWARD": 0x85,
        "LEFT": 0x03,
        "RIGHT": 0x07,
        "BACK_SLOW": 0x24,
        "SPIN_LEFT": 0x21,
        "FAST_FORWARD": 0x22,
        "FAST_BACKWARD": 0x23,
    }

    SYNC_MAP = {
        "UP": 0x11,
        "DOWN": 0x13,
        "MID": 0x15,
    }

    LEFT_GRIPPER_MAP = {
        "OPEN": 0x31,
        "CLOSE": 0x33,
        "MID": 0x35,
    }

    RIGHT_GRIPPER_MAP = {
        "OPEN": 0x32,
        "CLOSE": 0x34,
        "MID": 0x36,
    }

    CAMERA_MAP = {
        "SEARCH": 0x09,
        "AIM": 0x10,
    }

    HANDSHAKE_MAP = {
        "ACK0": 0x51,
        "ACK1": 0x52,
        "ACK2": 0x53,
    }

    def __init__(self, port="COM3", baudrate=115200, enable_serial=False):
        self.port = port
        self.baudrate = baudrate
        self.enable_serial = enable_serial
        self.serial_conn = None
        self.serial_module = None

    def open(self):
        if not self.enable_serial:
            print("[Serial] 当前为调试模式，不打开真实串口。")
            return False

        if self.serial_module is None:
            self.serial_module = _load_pyserial_module()

        if self.serial_module is None:
            print("[Serial] 未找到 pyserial，切换到调试模式。")
            self.enable_serial = False
            return False

        try:
            self.serial_conn = self.serial_module.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(0.2)
            print(f"[Serial] 已连接 {self.port} @ {self.baudrate}")
            return True
        except Exception as exc:
            print(f"[Serial] 串口打开失败，切换到调试模式: {exc}")
            self.serial_conn = None
            self.enable_serial = False
            return False

    def send_command(self, command):
        self.send_chassis(command)

    def _send_byte(self, value, label):
        data = bytes([value & 0xFF])
        if self.enable_serial and self.serial_conn is not None:
            try:
                self.serial_conn.write(data)
            except Exception as exc:
                print(f"[Serial] 发送失败: {exc}")
        print(f"[Serial] send: {label} -> {hex(value)}")

    def send_chassis(self, command):
        value = self.CHASSIS_MAP.get(command, self.CHASSIS_MAP["STOP"])
        self._send_byte(value, f"chassis:{command}")

    def send_left_gripper(self, action):
        value = self.LEFT_GRIPPER_MAP.get(action)
        if value is None:
            print(f"[Serial] 未知左夹爪动作: {action}")
            return
        self._send_byte(value, f"left_gripper:{action}")

    def send_right_gripper(self, action):
        value = self.RIGHT_GRIPPER_MAP.get(action)
        if value is None:
            print(f"[Serial] 未知右夹爪动作: {action}")
            return
        self._send_byte(value, f"right_gripper:{action}")

    def send_sync(self, action):
        value = self.SYNC_MAP.get(action)
        if value is None:
            print(f"[Serial] 未知同步机构动作: {action}")
            return
        self._send_byte(value, f"sync:{action}")

    def send_camera(self, action):
        value = self.CAMERA_MAP.get(action)
        if value is None:
            print(f"[Serial] 未知云台动作: {action}")
            return
        self._send_byte(value, f"camera:{action}")

    def send_handshake(self, action):
        value = self.HANDSHAKE_MAP.get(action)
        if value is None:
            print(f"[Serial] 未知握手动作: {action}")
            return
        self._send_byte(value, f"handshake:{action}")

    def close(self):
        if self.serial_conn is not None:
            self.serial_conn.close()
            self.serial_conn = None
