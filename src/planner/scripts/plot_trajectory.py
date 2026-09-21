#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_trajectory.py -- 由 record_trajectory.py 的 CSV 生成作业交付图表

产出（默认写入 docs/assets/）
------------------------------
    trajectory_3d.png      三维轨迹效果图（规划轨迹 + 实飞轨迹 + 航点）
    speed_profile.png      速度曲线（随时间）
    trajectory_topdown.png 俯视图（x-y 平面，便于观察避障绕行）

用法
----
    python3 plot_trajectory.py --csv ~/allocnet_trajectory.csv \\
                               --out-dir docs/assets
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # 无显示环境（VM/headless）必须
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401  注册 3d 投影


SOURCE_STYLE = {
    "planned": dict(color="#1f77b4", lw=2.2, ls="-",
                    label="AllocNet planned trajectory"),
    "flight":  dict(color="#d62728", lw=1.6, ls="-",
                    label="Executed flight"),
    "teleop":  dict(color="#2ca02c", lw=1.4, ls="--",
                    label="Keyboard cursor path"),
}


def read_csv(path):
    """读回 CSV，返回按 source 分组的 {source: {field: [values]}}。"""
    if not os.path.isfile(path):
        sys.exit("找不到 CSV: %s" % path)

    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            src = row["source"]
            bucket = data.setdefault(
                src, {"time": [], "x": [], "y": [], "z": [], "v": []})
            try:
                for k in ("time", "x", "y", "z", "v"):
                    bucket[k].append(float(row[k]))
            except (ValueError, KeyError):
                continue

    if not data:
        sys.exit("CSV 为空，请先运行仿真并记录数据: %s" % path)
    return data


def pick(data, *names):
    """按优先级返回第一个存在的 source，便于 flight/planned 回退。"""
    for n in names:
        if n in data and len(data[n]["x"]) > 1:
            return data[n]
    return None


def split_runs(times, gap=1.0):
    """按时间间隔切分连续段，避免多段轨迹被连成一条斜线。"""
    runs, cur = [], [0]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > gap:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    return runs


def load_obstacles(path):
    """读取离线仿真导出的障碍物清单 (可选)。

    文件格式: cx, cy, radius 每行一个。没有该文件时返回空列表，
    图上就不画障碍物（真实 ROS 运行时不依赖它，因为地图来自点云）。
    """
    if not path or not os.path.isfile(path):
        return []
    obs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    obs.append(tuple(float(v) for v in parts[:3]))
                except ValueError:
                    continue
    return obs


def draw_obstacles_2d(ax, obstacles):
    """俯视图里的障碍圆。"""
    for (cx, cy, r) in obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color="#444444",
                                alpha=0.35, zorder=1))
        ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                color="#222222", lw=1.0, zorder=2))
    if obstacles:
        # 只为第一个障碍加图例条目，避免重复
        cx, cy, r = obstacles[0]
        ax.add_patch(plt.Circle((cx, cy), r, color="#444444",
                                alpha=0.35, label="Obstacle (cylinder)",
                                zorder=1))


def clear_topdown_axes(ax, obstacles, data):
    """按内容自适应俯视图范围，避免轨迹被压成一条细带。"""
    xs = [0.0]
    ys = [0.0]
    for d in data.values():
        xs.extend(d["x"])
        ys.extend(d["y"])
    for (cx, cy, r) in obstacles:
        xs.extend([cx - r, cx + r])
        ys.extend([cy - r, cy + r])
    if not xs:
        return
    pad = 1.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


# ----------------------------------------------------------------------
def draw_obstacles_3d(ax, obstacles, z0=0.0, z1=3.0):
    """在 3D 图里把圆柱障碍画成半透明柱面线框。"""
    import numpy as np
    theta = np.linspace(0, 2 * np.pi, 24)
    zz = np.array([z0, z1])
    tt, zgrid = np.meshgrid(theta, zz)
    for (cx, cy, r) in obstacles:
        xx = cx + r * np.cos(tt)
        yy = cy + r * np.sin(tt)
        ax.plot_surface(xx, yy, zgrid, color="#888888", alpha=0.22,
                        linewidth=0, shade=False)


def plot_3d(data, out_path, obstacles=None):
    obstacles = obstacles or []
    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    draw_obstacles_3d(ax, obstacles)

    for src, style in SOURCE_STYLE.items():
        d = data.get(src)
        if not d or len(d["x"]) < 2:
            continue
        for run in split_runs(d["time"]):
            ax.plot([d["x"][i] for i in run],
                    [d["y"][i] for i in run],
                    [d["z"][i] for i in run], **style)

    # 起终点标注
    for src, color, tag in (("start", "#8c564b", "START"),
                            ("goal", "#e377c2", "GOAL")):
        d = data.get(src)
        if not d or not d["x"]:
            continue
        ax.scatter(d["x"][0], d["y"][0], d["z"][0],
                   c=color, s=90, marker="o", edgecolors="k",
                   label="%s (%.1f, %.1f, %.1f)"
                         % (tag, d["x"][0], d["y"][0], d["z"][0]),
                   depthshade=False)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("AllocNet + Keyboard Teleoperation — 3D Trajectory")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    try:
        ax.set_box_aspect((1, 1, 0.45))
    except AttributeError:
        pass
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("[plot] 写出 %s" % out_path)


def plot_topdown(data, out_path, obstacles=None):
    obstacles = obstacles or []
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    draw_obstacles_2d(ax, obstacles)

    for src, style in SOURCE_STYLE.items():
        d = data.get(src)
        if not d or len(d["x"]) < 2:
            continue
        first = True
        for run in split_runs(d["time"]):
            # 注意：必须显式传 ls，否则实线会覆盖 SOURCE_STYLE 里的虚线约定
            ax.plot([d["x"][i] for i in run],
                    [d["y"][i] for i in run],
                    color=style["color"], lw=style["lw"], ls=style["ls"],
                    label=style["label"] if first else None, zorder=3)
            first = False

    for src, color, tag in (("start", "#8c564b", "START"),
                            ("goal", "#e377c2", "GOAL")):
        d = data.get(src)
        if not d or not d["x"]:
            continue
        ax.scatter(d["x"][0], d["y"][0], c=color, s=110,
                   marker="o", edgecolors="k", zorder=5,
                   label="%s (%.1f, %.1f)" % (tag, d["x"][0], d["y"][0]))

    clear_topdown_axes(ax, obstacles, data)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top-down View (X–Y) — Obstacle Avoidance")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("[plot] 写出 %s" % out_path)


def plot_speed(data, out_path):
    """速度曲线。优先用实飞段的速度，缺失时用规划轨迹的时间序列估算。"""
    flight = pick(data, "flight", "planned", "teleop")
    if not flight:
        print("[plot] 无足够数据绘制速度曲线，跳过")
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    plotted = False
    for src in ("flight", "planned", "teleop"):
        d = data.get(src)
        if not d or len(d["time"]) < 2:
            continue
        # 只保留速度非零的样本，避免把静止游标也画成一条贴地线
        ts = [t for t, v in zip(d["time"], d["v"]) if v > 1e-6]
        vs = [v for v in d["v"] if v > 1e-6]
        if len(ts) < 2:
            continue
        style = SOURCE_STYLE.get(src, {})
        ax.plot(ts, vs, color=style.get("color", "C0"),
                lw=1.8, ls=style.get("ls", "-"),
                label=style.get("label", src))
        plotted = True

    if not plotted:
        print("[plot] 速度样本全为 0（可能 /visualizer/speed 未发布），跳过")
        plt.close(fig)
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed Profile — AllocNet Trajectory Execution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("[plot] 写出 %s" % out_path)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="record_trajectory.py 输出的 CSV")
    ap.add_argument("--out-dir", default="docs/assets", help="图片输出目录")
    ap.add_argument("--obstacles", default=None,
                    help="障碍物清单 (cx,cy,r 每行一个)，仅离线演示时需要；"
                         "ROS 运行时地图来自点云，可不传")
    args = ap.parse_args()

    data = read_csv(args.csv)
    print("[plot] 读到 source: %s" %
          ", ".join("%s(%d)" % (k, len(v["x"])) for k, v in data.items()))

    obstacles = load_obstacles(args.obstacles)
    if obstacles:
        print("[plot] 载入 %d 个障碍物" % len(obstacles))

    os.makedirs(args.out_dir, exist_ok=True)
    plot_3d(data, os.path.join(args.out_dir, "trajectory_3d.png"), obstacles)
    plot_topdown(data, os.path.join(args.out_dir, "trajectory_topdown.png"),
                 obstacles)
    plot_speed(data, os.path.join(args.out_dir, "speed_profile.png"))


if __name__ == "__main__":
    main()
