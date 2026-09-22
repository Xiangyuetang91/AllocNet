#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫航点构型并抓帧，用于采集 demo 素材（见 README 6.4）。

为什么需要"扫"：AllocNet 按固定 5 段推理时间分配，某些起终点几何会让
某段时长推理为 0，规划器打印 "time and seg does not fit" 后放弃本次规划
（不报错、不出轨迹），实测成功率约 1/5。因此这里遍历候选构型，
取第一组成功的。

前置：Xvfb 虚屏 + RViz（抓图配置），见 README 6.4。
按键经 /teleop/key 注入，无需 TTY。
"""
import os
import subprocess
import sys
import time

DISP = ":99"
os.environ["DISPLAY"] = DISP

import rospy                                    # noqa: E402
from std_msgs.msg import String                 # noqa: E402
from visualization_msgs.msg import Marker       # noqa: E402

FRAMES = "/tmp/shots"
KEY, TRAJ = "/teleop/key", "/visualizer/trajectory"
STEP = 0.5
traj = []


def on_traj(msg):
    if len(msg.points):
        traj.append(len(msg.points))


def main():
    os.makedirs(FRAMES, exist_ok=True)
    for f in os.listdir(FRAMES):
        if f.endswith(".png"):
            os.remove(os.path.join(FRAMES, f))

    rospy.init_node("best_shot", anonymous=True)
    pub = rospy.Publisher(KEY, String, queue_size=100)
    rospy.Subscriber(TRAJ, Marker, on_traj, queue_size=20)
    time.sleep(2.0)
    print("订阅者=%d" % pub.get_num_connections())

    n = [0]

    def send(s, d=0.06):
        for c in s:
            pub.publish(String(data=c))
            time.sleep(d)

    def grab():
        subprocess.call(["import", "-display", DISP, "-window", "root",
                         os.path.join(FRAMES, "s_%04d.png" % n[0])],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        n[0] += 1

    def move_to(tx, ty):
        send("r"); time.sleep(0.4)
        dx, dy = int(round(tx / STEP)), int(round(ty / STEP))
        if dx:
            send(("w" if dx > 0 else "s") * abs(dx))
        if dy:
            send(("a" if dy > 0 else "d") * abs(dy))
        time.sleep(0.35)

    # 多试几种构型（方向、距离各异）
    cands = [
        ((-3.0, 0.0), (2.0, 2.0)),
        ((-2.0, -1.0), (3.0, 1.0)),
        ((-3.0, 1.0), (1.0, 2.0)),
        ((-2.0, 0.0), (2.0, 1.0)),
        ((0.0, -1.0), (3.0, 2.0)),
        ((-4.0, -2.0), (1.0, 1.0)),
        ((-1.0, 2.0), (3.0, -1.0)),
        ((-3.0, -3.0), (2.0, 2.0)),
        ((0.0, 0.0), (2.0, 2.0)),
        ((-2.0, 2.0), (2.0, -2.0)),
    ]

    for i, (s, g) in enumerate(cands):
        before = len(traj)
        print("=== %d: %s -> %s ===" % (i + 1, s, g))
        move_to(*s)
        send("g"); time.sleep(0.7); grab()      # 起点标记
        grab()
        move_to(*g)
        grab()

        t0 = time.time()
        send("g")
        while time.time() - t0 < 3.5:           # 规划瞬间高频抓
            grab(); time.sleep(0.14)

        if len(traj) > before:
            print("  ✔ 成功，记录飞行 16s")
            t1 = time.time()
            while time.time() - t1 < 16:
                grab(); time.sleep(0.28)
            print("  帧数=%d  START=%s GOAL=%s" % (n[0], s, g))
            return 0
        print("  未成功")

    print("✘ 全失败 帧数=%d" % n[0])
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(2)
