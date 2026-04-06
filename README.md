# rescue_car

救援小车视觉控制项目，当前实现按以下原则组织：

- 颜色检测为主，YOLO 为辅助
- 图像质量差时关闭 YOLO，但保留颜色检测
- 状态机固定为 `SEARCH -> GRAB -> GRAB_CONFIRM -> DELIVER`
- `Decision` 只负责方向控制
- `Executor` 只负责动作执行
- `main` 负责状态机、抓取确认和整体流程

## 代码结构

```text
camera.py    相机读取
detect.py    颜色主检测、安全区检测、低频 YOLO 辅助检测
decision.py  方向控制决策（SEARCH / DELIVER）
executor.py  抓取与释放动作序列执行
path.py      地面可通行方向分析
quality.py   图像质量判断
escape.py    SEARCH 阶段的简单脱困覆盖
serial.py    串口命令映射与发送
main.py      状态机、抓取确认、运载流程、模块调度
```

## 当前流程

### 1. SEARCH

- 始终运行颜色检测
- 仅在画质良好时低频运行 YOLO
- `DecisionMaker.decide_search()` 只输出底盘方向
- `main.py` 根据颜色目标是否居中、是否接近、夹爪是否空闲来决定是否切到 `GRAB`

### 2. GRAB

- `main.py` 进入 `GRAB` 状态后，只触发 `ActionExecutor.trigger_grab()`
- `executor.py` 按固定时序执行停车、下放、闭合、抬起、后退

### 3. GRAB_CONFIRM

- 抓取动作完成后进入确认状态
- 通过“目标是否在原侧消失”做连续多帧确认
- 确认成功才登记夹爪占用并继续任务
- 超时未确认则回到 `SEARCH`

### 4. DELIVER

- `DecisionMaker.decide_delivery()` 只负责朝安全区行驶
- 到达安全区后，`main.py` 触发 `ActionExecutor.trigger_release()`
- 释放完成后清空载物状态并回到 `SEARCH`

## 设计约束对应

### 识别策略

- `detect.py` 中 `color_target` 永远优先于 `yolo_target`
- 抓取候选只从颜色目标里选，不依赖 YOLO 直接触发抓取

### 图像质量策略

- `main.py` 中调用 `detector.detect(..., allow_yolo=quality_result["is_good"])`
- 画质差时仅禁用 YOLO，颜色目标和安全区检测继续运行

### 职责边界

- `decision.py` 不再返回 `should_grab`、`gripper_side` 等流程字段
- `executor.py` 不参与状态切换，只执行动作序列
- `main.py` 统一负责状态迁移、抓取确认、运载切换

## 运行

安装基础依赖：

```powershell
pip install opencv-python numpy
```

如需 YOLO：

```powershell
pip install ultralytics
```

运行：

```powershell
python main.py
```

默认是串口调试模式：

```python
SerialController(port="COM3", baudrate=115200, enable_serial=False)
```

上车前按实际串口修改 `port`，并把 `enable_serial` 改为 `True`。
