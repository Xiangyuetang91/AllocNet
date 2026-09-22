#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把抓到的整屏帧裁成交付资产（见 README 6.4）。

为什么不自动定位轨迹：点云用 AxisColor(Z) 渲染成密集彩虹，
轨迹那点纯蓝像素会被淹没，按颜色找包围盒会误检到点云。
因此用固定窗口（由观察得出），也顺带保证动图镜头稳定。

输入 /tmp/shots/s_*.png（capture_teleop_demo.py 产出）
输出 /tmp/rviz_planning.png + /tmp/teleop_demo.gif
"""
import glob
import os
import sys

from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SRC = "/tmp/shots"
VIEW = (340, 20, 1280, 800)         # 3D 视口
# 在视口坐标系里的聚焦窗口。
# 视口尺寸 940x780。由 s_0040 观察：轨迹与红色路径落在约 (560~790, 420~570)，
# 取一个居中的 4:3 窗口把它框住，且不越出视口右界(940)。
FOCUS = (400, 300, 900, 675)        # -> 500 x 375（4:3）
OUT_PNG = "/tmp/rviz_planning.png"
OUT_GIF = "/tmp/teleop_demo.gif"


def main():
    frames = sorted(glob.glob(os.path.join(SRC, "s_*.png")))
    print("源帧 %d" % len(frames))
    if not frames:
        sys.exit("无源帧")

    def load(p, focus=True):
        im = Image.open(p).convert("RGB").crop(VIEW)
        return im.crop(FOCUS) if focus else im

    # 静态图：取中后段（轨迹已出、飞行球在动）
    idx = int(len(frames) * 0.55)
    shot = load(frames[idx])
    shot.save(OUT_PNG)
    print("静态图 %s %s <- %s" % (OUT_PNG, shot.size,
                                  os.path.basename(frames[idx])))

    # 动图：同一窗口，逐 2 帧取（保留更多过程细节），缩放
    W = 640
    gif = []
    for g in frames[::2]:
        im = load(g)
        r = W / im.width
        im = im.resize((W, int(im.height * r)), Image.LANCZOS)
        gif.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    if gif:
        gif[0].save(OUT_GIF, save_all=True, append_images=gif[1:],
                    duration=200, loop=0, optimize=True)
        print("动图 %s %d 帧 %.2f MB"
              % (OUT_GIF, len(gif), os.path.getsize(OUT_GIF) / 1e6))


if __name__ == "__main__":
    main()
