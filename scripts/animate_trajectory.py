#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
animate_trajectory.py -- 把轨迹 CSV 渲染成动态 GIF（不依赖 ROS）

用于在没有 RViz 的环境下产出"轨迹随时间推进"的可视化，
或者在 RViz 抓帧失败时作为兜底素材。

产出的是**算法层重放动画**，不是 RViz 仿真录屏 ——
README 中对其来源有明确标注。

用法
----
    python animate_trajectory.py --csv demo.csv --obstacles obstacles.csv \
                                 --out ../docs/assets/trajectory_anim.gif
"""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.patches import Circle      # noqa: E402
from PIL import Image                      # noqa: E402


def read_csv(path):
    rows = []
    if not os.path.isfile(path):
        sys.exit("找不到 CSV: %s" % path)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["time"]), float(r["x"]), float(r["y"]),
                             float(r["z"]), float(r["v"]), r["source"]))
            except (ValueError, KeyError):
                continue
    return rows


def read_obstacles(path):
    if not path or not os.path.isfile(path):
        return []
    obs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = [v.strip() for v in line.split(",")]
            if len(p) >= 3:
                try:
                    obs.append(tuple(float(v) for v in p[:3]))
                except ValueError:
                    pass
    return obs


def read_cloud(path, z_slice=None):
    """读取真实地图点云 (x,y,z)，可选高度切片。"""
    if not path or not os.path.isfile(path):
        return None
    xs, ys = [], []
    with open(path) as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                if line.lower().startswith("x"):
                    continue
            p = line.split(",")
            if len(p) < 3:
                continue
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
            except ValueError:
                continue
            if z_slice and not (z_slice[0] <= z <= z_slice[1]):
                continue
            xs.append(x)
            ys.append(y)
    return (xs, ys) if xs else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--obstacles", default=None,
                    help="障碍圆清单 (cx,cy,r)，离线演示用")
    ap.add_argument("--cloud", default=None,
                    help="真实地图点云 (x,y,z)，由 dump_cloud.py 导出")
    ap.add_argument("--cloud-z", default=None,
                    help="点云高度切片 lo,hi（米）")
    ap.add_argument("--out", default="trajectory_anim.gif")
    ap.add_argument("--frames", type=int, default=100, help="动画帧数")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=720)
    args = ap.parse_args()

    rows = read_csv(args.csv)
    if not rows:
        sys.exit("CSV 为空")
    obstacles = read_obstacles(args.obstacles)
    zs = None
    if args.cloud_z:
        lo, hi = [float(v) for v in args.cloud_z.split(",")]
        zs = (lo, hi)
    cloud = read_cloud(args.cloud, zs)
    if cloud:
        print("[anim] 载入点云 %d 点%s" % (len(cloud[0]),
              "（切片 %s）" % (zs,) if zs else ""))

    planned = [r for r in rows if r[5] == "planned"]
    flight = [r for r in rows if r[5] == "flight"]
    teleop = [r for r in rows if r[5] == "teleop"]
    trace = planned or flight
    if len(trace) < 2:
        sys.exit("没有足够的规划/实飞点用于动画")

    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    pad = 2.0
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)

    # 按累计弧长建索引，让动画匀速推进而不是跳变
    arc = [0.0]
    for i in range(1, len(trace)):
        d = math.dist(trace[i - 1][1:4], trace[i][1:4])
        arc.append(arc[-1] + d)
    total = arc[-1] or 1.0

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                           "_anim_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    n = args.frames
    for k in range(n):
        upto = arc[0] + (arc[-1] - arc[0]) * k / max(1, n - 1)
        idx = 0
        while idx < len(arc) - 1 and arc[idx] < upto:
            idx += 1
        head = trace[idx]

        fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
        if cloud:
            # 真实地图：用点云画障碍（比规则圆准确）
            ax.scatter(cloud[0], cloud[1], s=1.2, c="#666666", alpha=0.4,
                       marker="s", zorder=1)
        else:
            for (cx, cy, r) in obstacles:
                ax.add_patch(Circle((cx, cy), r, color="#555555", alpha=0.35,
                                    zorder=1))
                ax.add_patch(Circle((cx, cy), r, fill=False, color="#222222",
                                    lw=1.0, zorder=2))

        if teleop:
            ax.plot([r[1] for r in teleop], [r[2] for r in teleop],
                    color="#2ca02c", lw=1.4, ls="--", label="Keyboard cursor")
        # 已走过的部分
        ax.plot([r[1] for r in trace[:idx + 1]], [r[2] for r in trace[:idx + 1]],
                color="#1f77b4", lw=2.4, label="AllocNet trajectory")
        # 剩余部分用淡色
        ax.plot([r[1] for r in trace[idx:]], [r[2] for r in trace[idx:]],
                color="#1f77b4", lw=1.0, alpha=0.25)

        ax.scatter(trace[0][1], trace[0][2], c="#8c564b", s=110,
                   edgecolors="k", zorder=6, label="START")
        ax.scatter(trace[-1][1], trace[-1][2], c="#e377c2", s=110,
                   edgecolors="k", zorder=6, label="GOAL")
        # 当前位置
        ax.scatter(head[1], head[2], c="#d62728", s=150, marker="o",
                   edgecolors="k", zorder=7, label="Quadrotor")
        ax.annotate("v=%.2f m/s" % head[4], (head[1], head[2]),
                    textcoords="offset points", xytext=(12, 12), fontsize=9)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("AllocNet Trajectory Replay — t=%.2fs" %
                     (head[0] - trace[0][0]))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_dir, "f_%05d.png" % k))
        plt.close(fig)

    # 合成 GIF
    files = sorted(os.listdir(tmp_dir))
    imgs = []
    for f in files:
        im = Image.open(os.path.join(tmp_dir, f)).convert("RGB")
        h = int(im.height * args.width / im.width)
        im = im.resize((args.width, h), Image.LANCZOS)
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / args.fps), loop=0, optimize=True)
    mb = os.path.getsize(args.out) / 1024.0 / 1024.0
    print("[anim] 写出 %s (%.2f MB, %d 帧)" % (args.out, mb, len(imgs)))

    for f in files:
        try:
            os.remove(os.path.join(tmp_dir, f))
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
