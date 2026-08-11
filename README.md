# PanoGUI_用于三维重建的全景视频预处理工具

一个面向三维重建流程的桌面端预处理工具,提供 **FPS 抽帧** 与 **全景图(Equirectangular)转 CubeMap 切分** 两大核心功能,帮助将 360° 全景视频快速转换为可用于 NeRF / 3D Gaussian Splatting 等三维重建算法的图像数据集。

专为研究人员与开发者设计,零配置开箱即用,内置 FFmpeg,打包后单文件即可运行。

工具采用深色主题,主界面包含 Step 1 抽帧、Step 2 CubeMap 切分、实时日志与底部状态栏。

---

## 功能特性

### Step 1 · FPS 抽帧
- 调用内置 FFmpeg 按指定帧率从全景视频中抽取图像帧
- 通过 FFprobe 自动读取视频时长与帧率,精确预估总帧数
- 实时解析 `frame=` 输出,进度条精确到帧
- 支持自定义输出目录与抽帧 FPS(建议 2–5)

### Step 2 · CubeMap 切分
- 将 Equirectangular 全景图转换为 6 个立方体面:`front / back / left / right / up / down`
- 基于球面坐标(经纬度)映射,使用 OpenCV `remap` + `BORDER_WRAP` 实现高质量重采样
- 输出尺寸 1024×1024,适合主流重建算法输入
- **多进程并行**(`multiprocessing.Pool`),自动利用全部 CPU 核心

### 交互体验
- 深色主题 UI(午夜蓝 + 玫红强调色)
- 实时处理日志(线程安全写入)
- 双进度条独立显示两步处理状态
- 悬浮 ToolTip 提示
- 底部状态栏实时反馈

---

## 应用场景

| 场景 | 说明 |
|------|------|
| 室内外全景重建 | 将 360° 相机拍摄的视频转为序列帧,送入 NeRF / 3DGS 训练 |
| CubeMap 数据制备 | 为 Skybox、环境贴图、6-DoF VR 场景生成立方体贴图 |
| 视频摘要 | 按低帧率抽取关键帧,快速浏览长视频内容 |
| 数据集预处理 | 批量将全景帧拆分为 6 面,降低单图分辨率压力 |

---

## 技术栈

- **Python 3** — 主语言
- **Tkinter / ttk** — 桌面 GUI
- **OpenCV (cv2)** — 图像重采样与映射
- **NumPy** — 球面坐标向量化计算
- **FFmpeg / FFprobe** — 视频抽帧与元信息解析
- **multiprocessing** — 多进程并行加速
- **PyInstaller** — 打包为 Windows 单文件可执行程序

---

## 项目结构

```
pano_gui/
├── pano_gui.py              # 主程序(GUI + 抽帧 + CubeMap 算法)
├── pano_gui.spec            # PyInstaller 打包配置
├── pano_gui_old.py          # 历史版本备份
├── ffmpeg/
│   ├── bin/
│   │   ├── ffmpeg.exe       # 视频处理
│   │   ├── ffprobe.exe      # 视频信息探测
│   │   └── ffplay.exe
│   └── presets/             # 编码预设
```

---

## 快速开始

## 从源码运行(推荐开发者)

```bash
# 1. 克隆仓库
git clone https://github.com/<your-username>/pano_gui.git
cd pano_gui

# 2. 安装依赖
pip install opencv-python numpy

# 3. 安装FFmpeg

# 4. 运行
python pano_gui.py

```

---

## 使用说明

1. **Step 1 抽帧**
   - 选择全景视频文件 → 选择输出目录 → 填写抽帧 FPS(默认 2)→ 点击「执行抽帧」
   - 进度条与日志会实时显示处理状态

2. **Step 2 CubeMap 切分**
   - 将 Step 1 的输出目录作为输入 → 选择新的输出目录 → 点击「执行 CubeMap」
   - 每张全景图将被切分为 6 个文件,命名格式:`frame_00001_front.jpg`、`frame_00001_back.jpg` …


---


## 算法原理简述

### Equirectangular → CubeMap 投影

全景图采用经纬度展开(Equirectangular)的方式存储 360° 图像,CubeMap 则将其映射到立方体的 6 个面上。本工具的转换流程:

1. **生成面方向向量**:对每个面(front/back/left/right/up/down),在 [-1, 1] 范围内生成网格坐标,根据面对应的方向构造 3D 方向向量并归一化。
2. **球面坐标映射**:将方向向量 `(x, y, z)` 转为经纬度 `(lon, lat)`,再映射回原图的像素坐标:
   ```
   map_x = (lon / π + 1.0) × 0.5 × width
   map_y = (0.5 - lat / π) × height
   ```
3. **重采样**:使用 `cv2.remap` 进行双线性插值,`BORDER_WRAP` 处理水平边界以保证全景图无缝衔接。

### 多进程并行

`cubemap_parallel` 使用 `multiprocessing.Pool` 与 `cpu_count()` 创建进程池,每帧由独立进程处理;通过 `Manager.Queue` 回报进度,主线程零阻塞更新 UI。

---

## 打包说明

使用 PyInstaller 打包为单文件可执行程序:

```bash
pyinstaller pano_gui.spec
```

打包配置已包含:
- FFmpeg / FFprobe / FFplay 二进制文件
- 隐藏导入(cv2、numpy、PIL、multiprocessing 等)
- UPX 压缩
- 无控制台窗口(`console=False`)

---

## 开发笔记

- GUI 主线程保持响应:所有耗时任务(FFmpeg 调用、CubeMap 计算)均放入独立线程/进程
- 日志写入通过 `root.after(0, ...)` 调度回主线程,避免 Tkinter 跨线程报错
- FFmpeg 输出通过 PIPE 实时读取,使用正则解析 `frame=` 字段更新进度

---

## 许可证
`Copyright © 2022 NianBroken. All rights reserved.`

本项目采用 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0 "Apache-2.0") 许可证。简而言之，你可以自由使用、修改和分享本项目的代码，但前提是在其衍生作品中必须保留原始许可证和版权信息，并且必须以相同的许可证发布所有修改过的代码。

