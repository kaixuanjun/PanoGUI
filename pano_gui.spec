# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    ['pano_gui copy.py'],
    pathex=[current_dir],
    binaries=[
        # 打包ffmpeg相关程序
        ('ffmpeg/bin/ffmpeg.exe', 'ffmpeg/bin/'),
        ('ffmpeg/bin/ffprobe.exe', 'ffmpeg/bin/'),
        ('ffmpeg/bin/ffplay.exe', 'ffmpeg/bin/'),
    ],
    datas=[],
    hiddenimports=[
        'cv2',
        'numpy',
        'PIL',
        'multiprocessing',
        'tkinter',
        'threading',
        'subprocess',
        'glob',
        're',
    ],
    hookspath=[],
    hooksconfig={},
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='用于三维重建的全景视频预处理工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='用于三维重建的全景视频预处理工具',
)
