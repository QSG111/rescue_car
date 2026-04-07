# rescue_car

救援小车视觉控制项目。

当前版本在不重写系统的前提下，对现有代码做最小修改以满足比赛规则。状态机结构、方向决策模块、动作执行模块均保持原有职责边界不变。

## 状态机

主流程保持为：

```text
SEARCH -> GRAB -> GRAB_CONFIRM -> DELIVER
```

## 比赛物体数量

- `red_ball`: 4
- `blue_ball`: 4
- `black_ball`: 3
- `yellow_ball`: 2

## 已实现规则

### 1. 队伍限制

- 红队只能抓 `red_ball`
- 蓝队只能抓 `blue_ball`
- `black_ball` 两队都可以抓
- 对方颜色球在检测阶段直接过滤

### 2. 开局限制

- 开局必须先抓一个本队普通球并送到本方安全区
- 在首次成功投放前，只允许抓本队球
- 首次成功投放后，才开放 `black_ball` 和 `yellow_ball`

### 3. 携带数量限制

- 单次最多携带 `3` 个球

### 4. 黄球规则

- `yellow_ball` 不能作为第一个抓取目标
- 总共最多抓 `2` 个黄球
- 同一趟最多只能抓 `1` 个黄球
- 黄球必须单独转运
- 一旦当前趟已经抓到黄球，立即强制进入 `DELIVER`

### 5. 普通球规则

- 普通球包括：`red_ball`、`blue_ball`、`black_ball`
- 普通球可以组合抓取
- 单趟最多总共携带 `3` 个

### 6. 安全区约束

- 投放阶段优先寻找本方安全区
- 若未检测到本方安全区、却检测到对方安全区，则先执行横向避让
- 这是当前代码里对“不得进入对方安全区”的最小保护实现

## 规则状态变量

规则状态保存在 `Detector` 中：

```python
self.start_delivered = False
self.yellow_picked_count = 0
self.normal_picked_count = 0

self.current_load = 0
self.current_has_yellow = False
```

说明：

- `start_delivered`：是否已完成开局第一次成功投放
- `yellow_picked_count`：累计成功抓取黄球数量
- `normal_picked_count`：累计成功抓取普通球数量
- `current_load`：当前这一趟已经抓到的球数
- `current_has_yellow`：当前这一趟是否已经带有黄球

## 规则落点

### `detect.py`

负责：

- 颜色目标检测
- 本方安全区检测
- 对方安全区检测
- 比赛规则过滤
- 规则状态维护

核心接口：

```python
def can_target_label(self, label, team_color):
```

用于决定某类球是否允许进入候选目标。

```python
def should_force_deliver(self):
```

用于集中判断当前是否必须立即进入投放流程。当前规则为：

- 尚未完成首次投放
- 或当前这趟已经抓到黄球

### `main.py`

负责：

- 状态机切换
- 抓取触发
- 抓取确认
- 投放触发
- 抓取成功与投放成功后的状态回写

抓取确认成功后：

```python
detector.register_pick_result(action_context["target_label"])
```

释放完成后：

```python
detector.register_delivery_complete()
```

### `decision.py`

仅负责投放和搜索阶段的方向决策。

当前投放接口已扩展为：

```python
decide_delivery(safe_zone, path_result, frame_width, opponent_safe_zone=None)
```

当只看到对方安全区时，会先返回避让指令而不是直接前进。

### `executor.py`

只执行抓取和释放动作序列，不参与规则判断和状态机切换。

## 关键实现说明

### 1. 目标过滤

目标过滤集中在 `Detector.can_target_label()`，核心约束包括：

- 当前负载已满时拒绝继续抓取
- 对方颜色球直接拒绝
- 首次投放前只允许本队球
- 黄球必须满足：
  - 当前空载
  - 黄球累计抓取数小于 `2`
  - 已经至少抓过一个普通球

### 2. 黄球单独转运

当前趟一旦抓到黄球：

- `current_has_yellow = True`
- 后续不再允许任何球进入候选目标
- `main.py` 会强制切换到 `DELIVER`

这样可以稳定满足“黄球必须单独转运”的规则。

### 3. 开局先送本队球

当前若还未完成首次投放，只要已经抓到球，就立即进入 `DELIVER`，避免继续叠加其它球，确保开局流程稳定且不违规。

### 4. 三球上限的最小工程实现

当前没有重写机械抓取模型，而是将 `CarryManager` 改为“按数量统计当前载荷”：

- 最大载荷为 `3`
- 左右侧字段仍保留
- 内部记录的是各侧累计载荷数量

这是为了在不改状态机结构的前提下实现比赛规则，属于工程上的最小兼容方案。

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

上车前按实际串口修改 `port`，并将 `enable_serial` 改为 `True`。

## 2026-04 同步说明

### 安全区识别

当前 `detect.py` 中的安全区检测使用“紫色外框 + 本队颜色内区”的双重判定：

- 先在 HSV 空间中提取紫色线框，阈值为 `lower=[120,40,40]`、`upper=[160,255,255]`。
- 对紫色 mask 使用 `cv2.dilate(kernel=5x5, iterations=2)`，用于加粗空心线框并连接断裂边。
- 对最大轮廓使用 `cv2.convexHull` 计算凸包。
- 若凸包面积小于 `5000`，直接判定为未找到安全区。
- 在凸包外接矩形 ROI 内继续检查本队颜色。
- `red` 使用两个 HSV 区间合并：`0-10` 和 `170-180`。
- `blue` 使用 HSV 区间：`100-140`。
- 只有当 ROI 内本队颜色像素占比超过 `10%` 时，才认定该安全区有效。

这套逻辑的目的是避免仅因看到紫色框就误把球投进对手安全区。

### 搜索与投递决策

当前 `decision.py` 已将搜索球体与搜索安全区时的“看不见目标”逻辑统一：

- 一旦重新看到球或安全区，会将 `blind_search_timer` 清零。
- 看不见目标时，先执行 `50` 帧原地左转扫描，返回 `LEFT`，`reason = "radar_scan"`。
- 之后执行 `20` 帧换位移动，`reason = "relocate"`。
- 换位阶段如果路径可通行，则按 `path_result["best_direction"]` 行驶。
- 若路径不通，则继续返回 `LEFT`，避免停在死角。

因此现在 `SEARCH` 和 `DELIVER` 两个阶段在“无目标”时的行为一致：先转圈扫描，再换位置继续找。
