import tkinter as tk
from tkinter import filedialog
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


def extract_fps(video, out, fps):
    os.makedirs(out, exist_ok=True)

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", video,
        "-vf", f"fps={fps}",
        os.path.join(out, "frame_%05d.jpg")
    ]

    subprocess.run(cmd)


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
        return

    size = 1024
    faces = ["front", "back", "left", "right", "up", "down"]

    for f in faces:
        dirs = face_dirs(size, f)
        map_x, map_y = project(img, dirs)

        out_img = cv2.remap(
            img,
            map_x,
            map_y,
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

    files = [
        os.path.join(inp, f)
        for f in os.listdir(inp)
        if f.lower().endswith((".jpg", ".png"))
    ]

    if len(files) == 0:
        if finish_cb:
            finish_cb()
        return

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

        if finish_cb:
            finish_cb()

    threading.Thread(target=worker).start()
    threading.Thread(target=monitor).start()


def main():
    global e_video, e_fps, e_out1
    global e_cube_in, e_cube_out
    global status, progress

    root = tk.Tk()
    root.title("全景视频预处理")
    root.geometry("640x360")

    tk.Label(root, text="Step1：FPS抽帧", font=("Arial", 12, "bold")).pack()

    f1 = tk.Frame(root)
    f1.pack()

    e_video = tk.Entry(f1, width=60)
    e_video.pack(side=tk.LEFT)
    tk.Button(f1, text="输入目录", command=lambda: browse_file(e_video)).pack(side=tk.LEFT)

    f2 = tk.Frame(root)
    f2.pack()

    e_out1 = tk.Entry(f2, width=60)
    e_out1.pack(side=tk.LEFT)
    tk.Button(f2, text="输出目录", command=lambda: browse_dir(e_out1)).pack(side=tk.LEFT)

    f3 = tk.Frame(root)
    f3.pack()

    e_fps = tk.Entry(f3, width=10)
    e_fps.insert(0, "2")
    e_fps.pack(side=tk.LEFT)

    tk.Label(f3, text="FPS").pack(side=tk.LEFT)

    tk.Button(root, text="   执 行   ",
              command=lambda: threading.Thread(
                  target=lambda: extract_fps(
                      e_video.get(),
                      e_out1.get(),
                      int(e_fps.get())
                  )
              ).start()).pack(pady=5)

    tk.Label(root, text="Step2：CubeMap切分",
             font=("Arial", 12, "bold")).pack()

    f4 = tk.Frame(root)
    f4.pack()

    e_cube_in = tk.Entry(f4, width=60)
    e_cube_in.pack(side=tk.LEFT)
    tk.Button(f4, text="输入目录", command=lambda: browse_dir(e_cube_in)).pack(side=tk.LEFT)

    f5 = tk.Frame(root)
    f5.pack()

    e_cube_out = tk.Entry(f5, width=60)
    e_cube_out.pack(side=tk.LEFT)
    tk.Button(f5, text="输出目录", command=lambda: browse_dir(e_cube_out)).pack(side=tk.LEFT)

    def update_progress(done, total):
        pct = int(done / total * 100)
        root.after(0, lambda: progress.config(
            text=f"Step2进度：{pct}% ({done}/{total})"
        ))

    def start_step2():
        status.config(text="Step2运行中...")

        def task():
            cubemap_parallel(
                e_cube_in.get(),
                e_cube_out.get(),
                update_progress,
                finish_cb=lambda: root.after(
                    0, lambda: status.config(text="Step2完成")
                )
            )

        threading.Thread(target=task).start()

    tk.Button(root, text="   执 行   ",
              command=start_step2).pack(pady=5)

    status = tk.Label(root, text="状态：等待")
    status.pack()

    progress = tk.Label(root, text="Step2进度：0%")
    progress.pack()

    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
