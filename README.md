# Auto-Drawing-Script-for-Steam-Game-Draw-Guess-
Auto-Drawing Script for Steam Game “Draw &amp; Guess”
# Edge Draw Script Generator

将**任意图片**（PNG/JPG/AVIF…）转换为一个可执行的 Python “自动绘图脚本”：脚本会在你切换到画图软件后，自动移动鼠标并按顺序描绘图像的边缘线条（基于 Canny 边缘检测）。

> ⚠️ 注意：自动化脚本会控制鼠标。运行前请保存工作并确保你能随时中断（Ctrl+C 或 PyAutoGUI 的 FAILSAFE）。

---

## 功能概览

- **OpenCV Canny** 边缘检测（可调阈值）
- 可选 **dilation**（膨胀）让线条更粗更易画
- 将边缘像素转成**连续折线（polylines）**，减少 `mouseDown/mouseUp` 次数
- 可选 **RDP（Ramer–Douglas–Peucker）** 折线简化（`--smooth-polylines` + `--rdp-epsilon`）
- 可选 **DirectInput** 后端（某些游戏/软件中 PyAutoGUI 无效时使用）
- 输出脚本更“数据驱动”，文件更小、更易读（将点列表写入 `STROKES`，运行时循环绘制）

---

## 依赖与环境

### 生成脚本所需（运行生成器）

- Python 3.10+（推荐 3.11/3.12）
- `numpy`
- `opencv-python`
- `Pillow`

安装示例：

```bash
pip install numpy opencv-python Pillow
```

### 运行生成的绘图脚本所需（执行 output_draw_script.py）

二选一即可：

- `pyautogui`（默认）
- 或 `pydirectinput` / `pydirectinput_rgx`（加 `--directinput` 时）

安装示例：

```bash
pip install pyautogui
# 或
pip install pydirectinput
# 或
pip install pydirectinput-rgx
```

### 关于 AVIF

你示例里使用了 `input_image.avif`。Pillow 是否能读取 AVIF 取决于你本地是否安装了 AVIF 解码插件/后端。常见方案：

```bash
pip install pillow-avif-plugin
```

---

## 快速开始

### 1) 生成绘图脚本

**示例 1（与你提供的命令一致）：**

```bash
python edge_draw_script_generator_optimized_v2_improved.py \
    --colour "#00FF00" \
    --low-threshold 50 \
    --high-threshold 150 \
    --dilation 0 \
    --top-left 250 110 \
    --pixel-size 1 \
    --pause 0.001 \
    --duration 0.001 \
    --directinput \
    --smooth-polylines \
    --rdp-epsilon 2.0 \
    input_image.avif \
    output_draw_script.py
```

**示例 2（Windows 路径）：**

```bash
python "V:\python\edge_draw_script_generator_refactored.py" ^
  --colour "#00FF00FF" ^
  --low-threshold 50 ^
  --high-threshold 150 ^
  --dilation 0 ^
  --top-left 250 110 ^
  --pixel-size 1 ^
  --pause 0.004 ^
  --duration 0.004 ^
  --directinput ^
  --smooth-polylines ^
  --rdp-epsilon 0.1 ^
  "V:\6386658381162275534602095.png" ^
  "V:\python\out_draw_script.py"
```

> ✅ `--colour` 支持 `#RRGGBB`、`#RRGGBBAA`、`#AARRGGBB`。如果你输入 8 位并存在歧义，程序会优先选择 **alpha 不为 0** 的解释；否则默认按 `#RRGGBBAA`（这能让 `#00FF00FF` 按“绿色+不透明”理解）。

---

### 2) 执行生成的脚本开始绘图

```bash
python output_draw_script.py
```

脚本会打印提示并 **等待 5 秒**，你需要在这 5 秒内切换到你的绘图软件，并确保画布可绘制、鼠标在合适区域。

---

## 参数说明

### 位置参数

- `input_image`：输入图片路径（任意 Pillow/OpenCV 可读取的格式）
- `output_draw_script`：输出 Python 绘图脚本路径（`.py`）

### 边缘检测参数

- `--colour HEX`
  - 用于“边缘着色”的颜色，同时也是后续扫描/提取路径的目标颜色
  - 推荐：`"#00FF00"` 或 `"#00FF00FF"`（绿色）
- `--low-threshold FLOAT`
  - Canny 低阈值（越低越容易检出弱边缘，噪点也更多）
- `--high-threshold FLOAT`
  - Canny 高阈值（越高越“严格”）
- `--dilation INT`
  - 0 表示不膨胀；>0 表示膨胀次数，让线条更粗

### 绘制映射参数（屏幕坐标）

- `--top-left X Y`
  - 画布左上角在屏幕上的像素坐标（非常关键）
- `--pixel-size INT`
  - 图片像素到屏幕像素的缩放倍率
  - 例如：图片 1px 对应画布 2px，则 `--pixel-size 2`

### 自动化动作参数

- `--pause FLOAT`
  - 生成脚本中每次动作之间的暂停（秒）
  - 太小可能导致某些软件丢事件；太大会很慢
- `--duration FLOAT`
  - 生成脚本里每次 `moveTo` 的持续时间（秒）
  - 一些软件需要非 0 才能画得更稳定
- `--directinput`
  - 使用 DirectInput 后端（如果 PyAutoGUI 在目标软件中无效）

### 路径优化/简化参数

- `--smooth-polylines`
  - 开启折线简化（RDP）
- `--rdp-epsilon FLOAT`
  - 简化强度（越大点越少、越“硬”）
  - 经验：`0.1~0.5` 保真，`1~3` 明显简化但可能丢细节

---

## 调参建议（稳健性/效果/速度）

### 你想要更像“描线”的效果（少噪点）

- 提高 `--high-threshold`
- 增大 `--rdp-epsilon`
- 适当提高 `--low-threshold`（但不要超过 high）

### 你想要线条更明显、更粗

- 增大 `--dilation`
- 或增加 `--pixel-size`（让线条映射到更大的画布尺度）

### 你想要更稳定（少丢线/少断笔）

- `--duration` 设为 `0.002~0.02`
- `--pause` 设为 `0.001~0.01`
- 如果目标软件不吃 PyAutoGUI：加 `--directinput`

### 你想要更快（更少点、更少动作）

- 开启 `--smooth-polylines`
- 适当增大 `--rdp-epsilon`
- 降低输入图片分辨率（从源头减少边缘像素）

---

## 输出脚本说明

生成的 `output_draw_script.py` 主要包含：

- `STROKES = [...]`：点列表（每个 stroke 是一个点序列）
- `draw()`：循环 strokes，执行 `mouseDown -> moveTo... -> mouseUp`
- 安全机制：
  - 可 Ctrl+C 中断
  - PyAutoGUI 环境通常支持把鼠标移到屏幕左上角触发 FAILSAFE（如果后端支持）

---

## 常见问题（FAQ）

### 1) 画偏了 / 位置不对

- 重新测量 `--top-left X Y`
- 检查系统 DPI 缩放（Windows 125%/150% 会影响实际像素坐标）
- 检查画布是否滚动/缩放了（画布要固定在同一倍率）

### 2) 画不出来 / 只移动鼠标不按下

- 目标软件可能屏蔽 PyAutoGUI，尝试 `--directinput`
- 以管理员运行（某些环境需要更高权限）
- 确认生成脚本运行时，窗口确实获得焦点

### 3) 脚本太慢 / 文件太大

- 降低输入图片分辨率
- 开启 `--smooth-polylines` 并提高 `--rdp-epsilon`
- 调整阈值减少边缘数量（减少噪点）

---

## 未来可能的改进（未实现，但很有价值）

- **自适应采样**：曲率小区域减少点，曲率大区域保留点，兼顾速度与保真
- **反向/双向最短路排序**：允许 stroke 反向绘制，以进一步减少空移动
- **样条曲线拟合**：用 Bézier / B-spline 代替折线，点数显著降低
- **并行化**：大图的路径提取与简化可并行
- **单元测试 + fuzz 测试**：对 mask/path/RDP/脚本生成做更系统的回归保护

---

## 安全提示

自动化脚本具有“控制鼠标”的能力，建议：

- 运行前保存文件
- 不要在输入密码/支付等场景运行
- 出问题立即 Ctrl+C；PyAutoGUI 用户可尝试触发 FAILSAFE
