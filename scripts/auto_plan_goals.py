#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correct_e2e.py -- 用**正确的方式**唤醒规划器并触发规划

背景（两个独立问题，此前互相掩盖）
==================================
问题 1：规划器的 `mapInitialized` 初始为 false，目标点被静默丢弃
        实测：不重发地图时，连发 27 个目标点，规划器毫无输出 ——
        既没有 "New Try"，也没有 "Infeasible"，因为 `targetCallBack`
        第一句 `if (mapInitialized)` 直接跳过了整个函数体。

问题 2：`/structure_map/change_map` 不是"重发地图"，而是"**换一张新地图**"
        structure_map.cpp:
            void genMapCallback(const std_msgs::Bool& msg) {
              _seed += 1.0;
              _struct_map_gen.change_ratios(_seed, false, dt);  // ← 重新随机化比例
              _num += 1;
              pubSensedPoints();
            }
        实测障碍比例在 cylinders 3.42%~14.16%、circles 0.57%~13.42% 之间
        大幅跳变，地图被填满，任何航点都会被判 Infeasible。

正确解法：用 `/structure_map/change_res`（Float32，传当前分辨率 0.1）
        resCallback -> changeRes(不变) + resetMap + pubSensedPoints
        即：**重新发布同一张地图**，不改变障碍内容。
"""

import math
import sys
import time

import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker

MAP = "/structure_map/global_gridmap"
RES = "/structure_map/change_res"
GOAL = "/move_base_simple/goal"

FLIGHT_Z = 1.5
RATIO = (FLIGHT_Z - 0.0 - 0.2) / (5.0 - 0.4)


def log(m):
    print("[e2e] %s" % m)
    sys.stdout.flush()


def main():
    rospy.init_node("correct_e2e", anonymous=True, disable_signals=True)

    store = {}
    rospy.Subscriber(MAP, PointCloud2, lambda m: store.setdefault("m", m))
    traj = {"n": 0}

    def on_traj(mk):
        if mk.points:
            traj["n"] = len(mk.points)

    rospy.Subscriber("/visualizer/trajectory", Marker, on_traj)
    pub_res = rospy.Publisher(RES, Float32, queue_size=1, latch=True)
    pub_goal = rospy.Publisher(GOAL, PoseStamped, queue_size=1, latch=True)

    time.sleep(3)

    # --- 1. 用 change_res 重发同一张地图（唤醒 mapInitialized）-----------
    log("用 change_res 重发同一张地图（不换图）...")
    pub_res.publish(Float32(data=0.1))

    t0 = time.time()
    while "m" not in store and time.time() - t0 < 25:
        time.sleep(0.2)

    if "m" not in store:
        log("未收到点云")
        return 2

    pts = list(pc2.read_points(store["m"], field_names=("x", "y", "z"),
                               skip_nans=True))
    log("地图点数: %d" % len(pts))

    # --- 2. 挑候选航点 ----------------------------------------------------
    tol = 0.3
    threat = [(px, py) for (px, py, pz) in pts if abs(pz - FLIGHT_Z) <= tol]
    log("z=%.1f±%.1f 切片内障碍点: %d" % (FLIGHT_Z, tol, len(threat)))

    def clr(x, y):
        if not threat:
            return 99.0
        return min(math.hypot(px - x, py - y) for (px, py) in threat)

    cands = [(x * 1.0, y * 1.0) for x in range(-9, 10) for y in range(-9, 10)]
    ranked = sorted(cands, key=lambda p: -clr(*p))
    log("最空旷 5 点: %s" % ", ".join("(%.0f,%.0f|c=%.2f)" % (x, y, clr(x, y))
                                      for (x, y) in ranked[:5]))

    # --- 3. 逐组尝试 ------------------------------------------------------
    def send(x, y):
        m = PoseStamped()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = "odom"
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = FLIGHT_Z
        m.pose.orientation.z = RATIO
        m.pose.orientation.w = 1.0
        pub_goal.publish(m)

    pairs = []
    for i, s in enumerate(ranked[:6]):
        for g in sorted(ranked[i + 1:], key=lambda p: -math.dist(p, s))[:2]:
            if math.dist(s, g) > 5.0:
                pairs.append((s, g))

    for idx, (s, g) in enumerate(pairs):
        traj["n"] = 0
        log("[%d/%d] (%.0f,%.0f) -> (%.0f,%.0f)  %.1fm"
            % (idx + 1, len(pairs), s[0], s[1], g[0], g[1], math.dist(s, g)))
        send(*s)
        time.sleep(2.5)
        send(*g)

        t0 = time.time()
        while time.time() - t0 < 9.0:
            if traj["n"] > 0:
                break
            time.sleep(0.25)

        if traj["n"] > 0:
            log("  ✔ 规划成功！轨迹点数 %d" % traj["n"])
            log("等待飞行数据 (25s)...")
            time.sleep(25)
            log("完成：起点(%.0f,%.0f) 终点(%.0f,%.0f)"
                % (s[0], s[1], g[0], g[1]))
            return 0
        log("  ✘ 不可行")

    log("全部候选不可行")
    return 5


if __name__ == "__main__":
    sys.exit(main())
