# 智能救援小车系统（SRP简化版）

本项目是一个面向竞赛演示和 SRP 答辩的简化版智能救援小车视觉控制系统。

设计目标：

- 结构简单，便于讲解和维护
- 功能完整，覆盖摄像头、检测、路径、决策、串口、画质、脱困
- 实时优先，不使用复杂状态机和复杂调度
- 兼容比赛场景中的目标抓取与安全区回送

## 1. 项目结构

```text
camera.py    摄像头读取
detect.py    目标检测 + 安全区检测 + YOLO辅助
path.py      可通行区域判断
decision.py  控制决策
escape.py    自动脱困
quality.py   画面质量判断
serial.py    串口通信
main.py      主程序
README.md    项目说明
```

## 2. 主流程

`main.py` 中循环执行：

1. 获取图像
2. 检测目标
3. 判断路径
4. 判断画面质量
5. 做控制决策
6. 检查自动脱困
7. 发送底盘控制指令
8. 接近目标后执行抓取动作
9. 抓到目标后寻找本方安全区
10. 到达安全区后释放目标

系统中只保留两个简单阶段：

- `SEARCH`：搜索并抓取目标
- `DELIVER`：寻找安全区并回送释放

## 3. 赛规对应思路

根据你提供的赛规图片，当前代码中已经考虑以下内容：

- 普通目标：红色 / 蓝色
- 核心目标：黑色
- 危险目标：黄色
- 本方安全区：按本队颜色识别
- 左右夹爪独立控制
- 抓取后回送到本方安全区

当前优先级大致为：

1. 黑色核心目标
2. 本队颜色普通目标
3. 黄色危险目标
4. 对方颜色普通目标

说明：

- 这个优先级是为了让系统更符合比赛任务逻辑。
- 如果你们现场策略不同，可以在 `detect.py` 里调整 `priority`。

## 4. 检测模块说明

### 4.1 颜色检测

颜色检测是主检测，使用 HSV 阈值实现：

- 红色目标
- 蓝色目标
- 黑色目标
- 黄色目标
- 红/蓝安全区

优点：

- 实时性高
- 代码简单
- 适合答辩说明

### 4.2 YOLO 辅助检测

YOLO 只作为辅助检测，满足你的要求：

- 每 5 帧运行一次
- 非 YOLO 帧复用上次结果
- 当颜色检测失败时，可用 YOLO 结果兜底

如果要启用 YOLO，请在 `main.py` 中设置本地权重路径：

```python
detector = Detector(
    yolo_model_path=r"./best.pt",
    yolo_stride=5,
    yolo_classes=["person", "sports ball", "bottle"],
)
```

## 5. 路径判断说明

`path.py` 中采用最简策略：

- 只看画面下半部分
- 通过灰度阈值提取亮区域
- 将区域分成左、中、右三部分
- 选择更通畅的方向

输出固定为：

- `LEFT`
- `RIGHT`
- `FORWARD`
- `STOP`

## 6. 决策说明

`decision.py` 负责生成控制建议：

- 画面质量差时停车
- 有目标时优先跟踪目标
- 无目标时按通路走
- 目标足够接近时进入抓取流程

返回内容包括：

- `command`
- `reason`
- `gripper_side`
- `should_grab`
- `target_type`
- `target_label`

## 7. 自动脱困说明

`escape.py` 中采用简化版脱困策略：

- 连续多帧停住，视为可能卡住
- 按顺序尝试：
  - `LEFT`
  - `RIGHT`
  - `BACK_SLOW`
  - `FORWARD`

这个策略简单但足够答辩和基础演示。

## 8. 串口通信说明

### 8.1 已写入的协议映射

底盘：

- `STOP -> 0x00`
- `FORWARD -> 0x01`
- `BACKWARD -> 0x85`
- `LEFT -> 0x03`
- `RIGHT -> 0x07`
- `BACK_SLOW -> 0x24`
- `SPIN_LEFT -> 0x21`
- `FAST_FORWARD -> 0x22`
- `FAST_BACKWARD -> 0x23`

同步机构：

- `UP -> 0x11`
- `DOWN -> 0x13`
- `MID -> 0x15`

左夹爪：

- `OPEN -> 0x31`
- `CLOSE -> 0x33`
- `MID -> 0x35`

右夹爪：

- `OPEN -> 0x32`
- `CLOSE -> 0x34`
- `MID -> 0x36`

云台：

- `SEARCH -> 0x09`
- `AIM -> 0x10`

握手：

- `ACK0 -> 0x51`
- `ACK1 -> 0x52`
- `ACK2 -> 0x53`

### 8.2 左右夹爪分开控制

当前代码已经分成独立接口：

```python
serial_controller.send_left_gripper("OPEN")
serial_controller.send_left_gripper("CLOSE")

serial_controller.send_right_gripper("OPEN")
serial_controller.send_right_gripper("CLOSE")
```

这部分适合答辩时强调：

- 左右夹爪不是同一个动作复用
- 左右夹爪能根据目标所在侧独立选择

## 9. 抓取与回送流程

`main.py` 中通过 `RescueExecutor` 执行简单动作序列。

### 9.1 抓取流程

1. 停车
2. 同步机构下降
3. 按目标左右位置选择左夹爪或右夹爪闭合
4. 同步机构回中位
5. 后退
6. 停车

### 9.2 回送流程

1. 进入 `DELIVER` 阶段
2. 检测本方安全区
3. 朝安全区移动
4. 到达安全区后执行释放
5. 释放完成后回到 `SEARCH`

## 10. 如何运行

进入项目目录后运行：

```powershell
python main.py
```

按 `Q` 退出。

## 11. 上车前需要改的参数

### 11.1 本队颜色

在 `main.py` 中修改：

```python
team_color = "red"
```

如果抽到蓝队，改成：

```python
team_color = "blue"
```

### 11.2 串口开关

默认是调试模式：

```python
serial_controller = SerialController(port="COM3", baudrate=115200, enable_serial=False)
```

上车时改成：

```python
serial_controller = SerialController(port="COM3", baudrate=115200, enable_serial=True)
```

并将 `COM3` 改为实际串口号。

### 11.3 YOLO 权重

默认未加载权重，比赛需要时再改：

```python
yolo_model_path=r"./best.pt"
```

### 11.4 HSV 阈值

需要根据你们现场灯光微调，位置在：

- `detect.py`：目标颜色阈值
- `detect.py`：安全区颜色阈值

## 12. 建议的现场调试顺序

1. 先关闭串口，只看画面显示和识别框是否稳定
2. 调颜色阈值，确认红蓝黑黄都能分出来
3. 调安全区阈值，确认本方安全区能被识别
4. 再打开串口，检查底盘指令是否正确
5. 最后调夹爪动作时序

## 13. 当前版本的特点

优点：

- 结构清晰
- 模块齐全
- 便于答辩
- 适合继续加功能

限制：

- 路径规划还是最简版本
- 安全区识别依赖颜色和矩形外观
- 没有做更复杂的坐标定位
- 没有做多目标管理和精细任务调度

这些限制符合“SRP 简化版”的目标。
