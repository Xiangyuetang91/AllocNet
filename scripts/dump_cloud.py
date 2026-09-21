#!/usr/bin/env python3
"""导出当前地图点云到 CSV，供宿主机绘图时叠加障碍物。"""
import sys
import time

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32

MAP = "/structure_map/global_gridmap"
RES = "/structure_map/change_res"


def main():
    rospy.init_node("dump_cloud", anonymous=True, disable_signals=True)
    store = {}
    rospy.Subscriber(MAP, PointCloud2, lambda m: store.setdefault("m", m))
    pub_res = rospy.Publisher(RES, Float32, queue_size=1, latch=True)

    time.sleep(3)
    pub_res.publish(Float32(data=0.1))     # 重发同一张图（不换图）

    t0 = time.time()
    while "m" not in store and time.time() - t0 < 25:
        time.sleep(0.2)
    if "m" not in store:
        print("NO_CLOUD")
        return 2

    pts = list(pc2.read_points(store["m"], field_names=("x", "y", "z"),
                               skip_nans=True))
    print("点数: %d" % len(pts))

    # 稀疏化：0.1m 体素太密，按 0.2m 抽样，保留足够画图的点
    with open("/tmp/cloud.csv", "w") as f:
        f.write("x,y,z\n")
        step = max(1, len(pts) // 40000)      # 最多约 4 万点
        for i in range(0, len(pts), step):
            x, y, z = pts[i]
            f.write("%.3f,%.3f,%.3f\n" % (x, y, z))
    print("已写出 /tmp/cloud.csv (step=%d)" % step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
