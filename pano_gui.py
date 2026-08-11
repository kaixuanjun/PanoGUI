import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
import numpy as np
import subprocess
import threading
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager

BASE = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = os.path.join(BASE, "ffmpeg", "bin", "ffmpeg.exe")


# =========================
# 文件选择
# =========================
def browse_file(e):
    p = filedialog.askopenfilename()
    if p:
        e.delete(0, tk.END)
        e.insert(0, p)


def browse_dir(e):
    p = filedialog.askdirectory()
    if p:
        e.delete(0, tk.END)
        e.insert(0, p)


def log(msg):
    """线程安全的日志写入"""
    if hasattr(log, 'text_widget') and log.text_widget:
        log.text_widget.insert(tk.END, msg + "\n")
        log.text_widget.see(tk.END)


def get_video_info(video_path):
    """使用ffprobe获取视频时长和帧率"""
    ffprobe_path = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    if not os.path.exists(ffprobe_path):
        log(f"[抽帧] ffprobe不存在: {ffprobe_path}")
        return None, None

    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        log(f"[抽帧] ffprobe返回: {result.returncode}, stdout: {result.stdout[:200] if result.stdout else '空'}")
        if result.returncode == 0 and result.stdout.strip():
            import re
            lines = result.stdout.strip().split('\n')
            fps = None
            duration = None
            for line in lines:
                line = line.strip()
                if '/' in line:
                    # 可能是 30000/1001 格式
                    try:
                        parts = line.split('/')
                        if len(parts) == 2:
                            fps = float(parts[0]) / float(parts[1])
                    except:
                        pass
                elif line.replace('.', '').isdigit():
                    # 可能是 37.771100 格式
                    try:
                        duration = float(line)
                    except:
                        pass

            return duration, fps
        else:
            log(f"[抽帧] ffprobe失败: returncode={result.returncode}, stderr: {result.stderr[:100] if result.stderr else '空'}")
    except subprocess.TimeoutExpired:
        log("[抽帧] ffprobe超时")
    except Exception as e:
        log(f"[抽帧] ffprobe异常: {e}")
    return None, None


def extract_fps(video, out, fps, progress_cb=None):
    os.makedirs(out, exist_ok=True)
    log(f"[抽帧] 开始处理: {video}")
    log(f"[抽帧] 输出目录: {out}, FPS: {fps}")

    # 获取视频时长计算预估总帧数
    duration, video_fps = get_video_info(video)
    if duration and video_fps:
        total_frames = int(duration * fps)
        log(f"[抽帧] 预估抽取: {total_frames} 帧 (时长{duration:.1f}秒)")
        if progress_cb:
            progress_cb('total', total_frames)
    else:
        log("[抽帧] 无法获取视频信息，进度可能不准确")
        if progress_cb:
            progress_cb('total', None)

    if progress_cb:
        progress_cb('start')

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", video,
        "-vf", f"fps={fps}",
        os.path.join(out, "frame_%05d.jpg")
    ]

    # 使用PIPE实时获取ffmpeg输出
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1
    )

    import re
    done_count = 0
    buffer = b''

    def read_stderr():
        nonlocal done_count, buffer
        while True:
            chunk = process.stderr.read(1024)
            if not chunk:
                # stderr关闭，可能进程已结束
                if process.poll() is not None:
                    break
                continue
            buffer += chunk

            # 解析所有frame=xxx
            while True:
                match = re.search(rb'frame=\s*(\d+)', buffer)
                if match:
                    done_count = int(match.group(1))
                    buffer = buffer[match.end():]
                    if progress_cb and done_count % 5 == 0:  # 每5帧更新一次
                        progress_cb('progress', done_count)
                else:
                    break

            # 检查进程是否结束
            if process.poll() is not None:
                # 读取剩余内容
                remaining = process.stderr.read()
                if remaining:
                    buffer += remaining
                break

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    stderr_thread.join()
    process.wait()  # 确保进程完全结束

    if process.returncode == 0 or done_count > 0:
        log(f"[抽帧] 完成! 共 {done_count} 帧")
        if progress_cb:
            progress_cb('done', done_count)
    else:
        log(f"[抽帧] 错误 (code={process.returncode})")
        if progress_cb:
            progress_cb('error')


def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def face_dirs(size, face):
    u = np.linspace(-1, 1, size)
    v = np.linspace(-1, 1, size)
    uu, vv = np.meshgrid(u, v)

    if face == "front":
        x, y, z = uu, -vv, np.ones_like(uu)
    elif face == "back":
        x, y, z = -uu, -vv, -np.ones_like(uu)
    elif face == "left":
        x, y, z = -np.ones_like(uu), -vv, uu
    elif face == "right":
        x, y, z = np.ones_like(uu), -vv, -uu
    elif face == "up":
        x, y, z = uu, np.ones_like(uu), vv
    elif face == "down":
        x, y, z = uu, -np.ones_like(uu), -vv

    dirs = np.stack([x, y, z], axis=-1)
    return normalize(dirs)


def project(img, dirs):
    h, w = img.shape[:2]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)
    map_x = (lon / np.pi + 1.0) * 0.5 * w
    map_y = (0.5 - lat / np.pi) * h
    return map_x.astype(np.float32), map_y.astype(np.float32)


def process_frame(args):
    path, out, q = args
    img = cv2.imread(path)
    if img is None:
        q.put(0)
        return

    size = 1024
    faces = ["front", "back", "left", "right", "up", "down"]

    for f in faces:
        dirs = face_dirs(size, f)
        map_x, map_y = project(img, dirs)
        out_img = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )
        cv2.imwrite(
            os.path.join(out, f"{os.path.basename(path)}_{f}.jpg"),
            out_img
        )
    q.put(1)


def cubemap_parallel(inp, out, progress_cb, finish_cb=None):
    os.makedirs(out, exist_ok=True)
    log(f"[CubeMap] 开始处理: {inp}")
    log(f"[CubeMap] 输出目录: {out}")

    files = [
        os.path.join(inp, f)
        for f in os.listdir(inp)
        if f.lower().endswith((".jpg", ".png"))
    ]

    if len(files) == 0:
        log("[CubeMap] 未找到图片文件!")
        if finish_cb:
            finish_cb()
        return

    log(f"[CubeMap] 找到 {len(files)} 个文件")
    manager = Manager()
    q = manager.Queue()
    args = [(f, out, q) for f in files]

    def worker():
        with Pool(cpu_count()) as pool:
            pool.map(process_frame, args)

    def monitor():
        done = 0
        total = len(files)
        while done < total:
            q.get()
            done += 1
            progress_cb(done, total)
        progress_cb(total, total)
        log("[CubeMap] 完成!")
        if finish_cb:
            finish_cb()

    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=monitor, daemon=True).start()


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tw = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tw, text=self.text, justify=tk.LEFT,
            background="#2d2d2d", foreground="#ffffff",
            relief=tk.SOLID, borderwidth=1,
            font=("微软雅黑", 9)
        )
        label.pack()

    def hide(self, event=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None


def main():
    global e_video, e_fps, e_out1, e_cube_in, e_cube_out
    global pbar, status_bar

    root = tk.Tk()
    root.title("用于三维重建的全景视频预处理工具")
    root.geometry("720x750")
    root.resizable(False, False)

    # 设置样式
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except:
        pass

    # 自定义颜色
    BG_MAIN = "#1a1a2e"
    BG_SECOND = "#16213e"
    BG_INPUT = "#0f3460"
    ACCENT = "#e94560"
    TEXT_MAIN = "#ffffff"
    TEXT_SECOND = "#a0a0a0"

    root.configure(bg=BG_MAIN)

    # 标题栏
    title_frame = tk.Frame(root, bg=BG_SECOND, height=50)
    title_frame.pack(fill=tk.X)
    title_frame.pack_propagate(False)

    tk.Label(
        title_frame, text="用于三维重建的全景视频预处理工具",
        font=("微软雅黑", 16, "bold"),
        bg=BG_SECOND, fg=ACCENT
    ).pack(side=tk.LEFT, padx=20, pady=10)

    # 主内容区
    main_frame = tk.Frame(root, bg=BG_MAIN)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

    # ===== Step1 抽帧 =====
    step1_frame = tk.LabelFrame(
        main_frame, text="Step 1：FPS抽帧",
        font=("微软雅黑", 11, "bold"),
        bg=BG_MAIN, fg=ACCENT,
        bd=0, labelanchor="n"
    )
    step1_frame.pack(fill=tk.X, pady=(0, 15))

    input_frame = tk.Frame(step1_frame, bg=BG_MAIN)
    input_frame.pack(fill=tk.X, pady=8)

    tk.Label(input_frame, text="视频文件:", bg=BG_MAIN, fg=TEXT_MAIN,
             font=("微软雅黑", 10)).grid(row=0, column=0, sticky="w", padx=5)
    e_video = ttk.Entry(input_frame, width=55, font=("微软雅黑", 10))
    e_video.grid(row=0, column=1, padx=5)
    ttk.Button(input_frame, text="选择", width=8,
               command=lambda: browse_file(e_video)).grid(row=0, column=2, padx=5)

    output_frame = tk.Frame(step1_frame, bg=BG_MAIN)
    output_frame.pack(fill=tk.X, pady=8)

    tk.Label(output_frame, text="输出目录:", bg=BG_MAIN, fg=TEXT_MAIN,
             font=("微软雅黑", 10)).grid(row=0, column=0, sticky="w", padx=5)
    e_out1 = ttk.Entry(output_frame, width=55, font=("微软雅黑", 10))
    e_out1.grid(row=0, column=1, padx=5)
    ttk.Button(output_frame, text="选择", width=8,
               command=lambda: browse_dir(e_out1)).grid(row=0, column=2, padx=5)

    fps_frame = tk.Frame(step1_frame, bg=BG_MAIN)
    fps_frame.pack(fill=tk.X, pady=8)

    tk.Label(fps_frame, text="抽帧FPS:", bg=BG_MAIN, fg=TEXT_MAIN,
             font=("微软雅黑", 10)).grid(row=0, column=0, sticky="w", padx=5)
    e_fps = ttk.Entry(fps_frame, width=10, font=("微软雅黑", 10))
    e_fps.insert(0, "2")
    e_fps.grid(row=0, column=1, sticky="w", padx=5)
    ToolTip(e_fps, "每秒抽取的帧数，建议2-5")

    # 抽帧进度条
    step1_progress_frame = tk.Frame(step1_frame, bg=BG_MAIN)
    step1_progress_frame.pack(fill=tk.X, pady=8)

    tk.Label(step1_progress_frame, text="处理进度:", bg=BG_MAIN, fg=TEXT_MAIN,
             font=("微软雅黑", 10)).grid(row=0, column=0, sticky="w", padx=5)

    pbar_step1 = ttk.Progressbar(
        step1_progress_frame, length=350, mode='determinate'
    )
    pbar_step1.grid(row=0, column=1, padx=(5, 2), sticky="ew")

    step1_progress_label = tk.Label(
        step1_progress_frame, text="", bg=BG_MAIN, fg=TEXT_MAIN,
        font=("微软雅黑", 10), width=15, anchor="w"
    )
    step1_progress_label.grid(row=0, column=2, padx=(0, 5), sticky="w")

    def start_step1():
        total = None
        step1_progress_label.config(text="")

        def task():
            extract_fps(
                e_video.get(), e_out1.get(), int(e_fps.get()),
                progress_cb=lambda state, val=None: root.after(0, lambda: _update_step1(state, val))
            )

        def _update_step1(state, val):
            nonlocal total
            if state == 'total':
                total = val
                pbar_step1['maximum'] = val if val else 100
                pbar_step1['value'] = 0
                if val:
                    step1_progress_label.config(text=f"0/{val}")
                    log(f"[进度] 预估总帧数: {val}")
            elif state == 'start':
                status_bar.config(text="状态: 抽帧中...", fg=ACCENT)
            elif state == 'progress':
                pbar_step1['value'] = val
                step1_progress_label.config(text=f"{val}/{total if total else '?'}")
                status_bar.config(text=f"状态: 抽帧中...", fg=ACCENT)
            elif state == 'done':
                pbar_step1['value'] = pbar_step1['maximum']
                step1_progress_label.config(text=f"{int(pbar_step1['maximum'])}/{int(pbar_step1['maximum'])}")
                status_bar.config(text="状态: 抽帧完成", fg="#4ade80")
            elif state == 'error':
                status_bar.config(text="状态: 抽帧失败", fg="#f87171")

        threading.Thread(target=task, daemon=True).start()

    ttk.Button(
        step1_frame, text="执行抽帧",
        style="Accent.TButton",
        command=start_step1
    ).pack(pady=10)

    # 样式按钮
    style.configure("Accent.TButton",
                   background=ACCENT, foreground=TEXT_MAIN,
                   font=("微软雅黑", 10, "bold"),
                   padding=(20, 5))

    # ===== Step2 CubeMap =====
    step2_frame = tk.LabelFrame(
        main_frame, text="Step 2：CubeMap切分",
        font=("微软雅黑", 11, "bold"),
        bg=BG_MAIN, fg=ACCENT,
        bd=0, labelanchor="n"
    )
    step2_frame.pack(fill=tk.X, pady=(0, 15))

    cube_input_frame = tk.Frame(step2_frame, bg=BG_MAIN)
    cube_input_frame.pack(fill=tk.X, pady=8)

    tk.Label(cube_input_frame, text="输入目录:", bg=BG_MAIN, fg=TEXT_MAIN,
             font=("微软雅黑", 10)).grid(row=0, column=0, sticky="w", padx=5)
    e_cube_in = ttk.Entry(cube_input_frame, width=55, font=("微软雅黑", 10))
    e_cube_in.grid(row=0, column=1, padx=5)
    ttk.Button(cube_input_frame, text="选择", width=8,
               command=lambda: browse_dir(e_cube_in)).grid(row=0, column=2, padx=5)

    cube_output_frame = tk.Frame(step2_frame, bg=BG_MAIN)
    cube_output_frame.pack(fill=tk.X, pady=8)

    tk.Label(cube_output_frame, text="输出目录:", bg=BG_MAIN, fg=TEXT_MAIN,
             font=("微软雅黑", 10)).grid(row=0, column=0, sticky="w", padx=5)
    e_cube_out = ttk.Entry(cube_output_frame, width=55, font=("微软雅黑", 10))
    e_cube_out.grid(row=0, column=1, padx=5)
    ttk.Button(cube_output_frame, text="选择", width=8,
               command=lambda: browse_dir(e_cube_out)).grid(row=0, column=2, padx=5)

    # 进度条
    progress_frame = tk.Frame(step2_frame, bg=BG_MAIN)
    progress_frame.pack(fill=tk.X, pady=8)

    tk.Label(progress_frame, text="处理进度:", bg=BG_MAIN, fg=TEXT_MAIN,
             font=("微软雅黑", 10)).grid(row=0, column=0, sticky="w", padx=5)

    pbar = ttk.Progressbar(
        progress_frame, length=350, mode='determinate',
        maximum=100
    )
    pbar.grid(row=0, column=1, padx=(5, 2), sticky="ew")

    step2_progress_label = tk.Label(
        progress_frame, text="", bg=BG_MAIN, fg=TEXT_MAIN,
        font=("微软雅黑", 10), width=15, anchor="w"
    )
    step2_progress_label.grid(row=0, column=2, padx=(0, 5), sticky="w")

    def update_progress(done, total):
        pct = int(done / total * 100)
        pbar['value'] = pct
        step2_progress_label.config(text=f"{done}/{total}")

    def start_step2():
        step2_progress_label.config(text="")

        def task():
            cubemap_parallel(
                e_cube_in.get(), e_cube_out.get(),
                update_progress,
                finish_cb=lambda: root.after(
                    0, lambda: (
                        status_bar.config(text="状态: Step2完成", fg="#4ade80"),
                        step2_progress_label.config(text="")
                    )
                )
            )

        status_bar.config(text="状态: 处理中...", fg=ACCENT)
        threading.Thread(target=task, daemon=True).start()

    ttk.Button(
        step2_frame, text="执行CubeMap",
        style="Accent.TButton",
        command=start_step2
    ).pack(pady=10)

    # ===== 日志区域 =====
    log_frame = tk.LabelFrame(
        main_frame, text="处理日志",
        font=("微软雅黑", 11, "bold"),
        bg=BG_MAIN, fg=ACCENT,
        bd=0, labelanchor="n"
    )
    log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    log.text_widget = tk.Text(
        log_frame, height=12, width=80,
        font=("Consolas", 9),
        bg="#0d1117", fg="#c9d1d9",
        insertbackground="#c9d1d9",
        relief=tk.FLAT, bd=5
    )
    log.text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    scrollbar = tk.Scrollbar(log_frame, command=log.text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    log.text_widget.config(yscrollcommand=scrollbar.set)

    # 底部状态栏
    status_bar = tk.Label(
        root, text="状态: 等待操作",
        font=("微软雅黑", 9),
        bg=BG_SECOND, fg=TEXT_SECOND,
        anchor="w", padx=15, pady=8
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # 设置ttk主题颜色
    style.configure("TFrame", background=BG_MAIN)
    style.configure("TLabelframe", background=BG_MAIN)
    style.configure("TLabelframe.Label", background=BG_MAIN, foreground=ACCENT)

    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
