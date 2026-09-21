#!/usr/bin/env bash
# =============================================================================
# vm_run_demo.sh -- 无 GUI 环境下跑通 AllocNet 规划并采集数据
#
# 解决了 learning_planning.cpp 的地图初始化时序陷阱（详见 README §6.5）：
#   targetCallBack 的第一句是 `if (mapInitialized)`，而 mapInitialized 只在
#   mapCallBack 里置位，且地图点云是一次性发布的。若规划器错过那一帧，
#   之后所有目标点都会被**静默丢弃**（不规划、不报错、无日志）。
#
# 用法:
#     bash vm_run_demo.sh [输出CSV路径]
# =============================================================================
set -uo pipefail

WS="${WS:-$HOME/allocnet_ws}"
OUT_CSV="${1:-$HOME/allocnet_trajectory.csv}"
LOG="${LOG:-/tmp/allocnet_demo.log}"

log() { echo -e "\033[32m[demo]\033[0m $*"; }
warn() { echo -e "\033[33m[warn]\033[0m $*"; }

# source ROS 前后必须包 set +u：ROS 的 setup.bash 会引用未定义的 ROS_DISTRO，
# 在 set -u 下会直接终止整个脚本（详见 README §6.2）。
set +u
. /opt/ros/noetic/setup.bash
. "$WS/devel/setup.bash"
set -u

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

# ---------------------------------------------------------------------------
log "1/6 清理旧进程"
pkill -f roslaunch       2>/dev/null || true
pkill -f rosmaster       2>/dev/null || true
pkill -f structure_map   2>/dev/null || true
pkill -f learning_planning 2>/dev/null || true
sleep 3

# ---------------------------------------------------------------------------
log "2/6 冷启动仿真（无 RViz）"
rm -f "$OUT_CSV" "$LOG"
setsid nohup roslaunch planner teleop_planning.launch \
    use_gui:=false record:=true log_csv:="$OUT_CSV" \
    > "$LOG" 2>&1 < /dev/null &

# 等地图生成完成。地图是 20x20x5m / 0.1m 分辨率 = 2000 体素，
# 生成耗时约 20-30 秒（取决于 CPU）。
for i in $(seq 1 40); do
    sleep 2
    if grep -q "Finished generate random map" "$LOG" 2>/dev/null; then
        log "地图就绪（等待 ${i}×2s）"
        break
    fi
    [ "$i" -eq 40 ] && { warn "地图生成超时，日志尾部："; tail -20 "$LOG"; exit 1; }
done

grep -q "model loaded" "$LOG" && log "AllocNet 模型已加载" \
    || warn "未见 'model loaded'，规划可能失败"

# ---------------------------------------------------------------------------
log "3/6 用 change_res 唤醒规划器的 mapInitialized（不要用 change_map！）"
#
# /structure_map/change_map 的语义是「换一张新地图」而非「重发」：
#   genMapCallback -> _seed += 1; change_ratios(_seed,...)
# 每发一次障碍比例都会畸变（实测圆柱/环形比例从 12%/0.6% 变到 14%/13%），
# 地图被填满后任何航点都会被判 Infeasible。
#
# 正确的做法是用 change_res 以**相同分辨率**重发同一张图：
#   resCallback -> changeRes(不变) + resetMap + pubSensedPoints
# 注意类型是 std_msgs/Float32（不是 Bool，也不是 Empty）。
rostopic pub -1 /structure_map/change_res std_msgs/Float32 "data: 0.1" \
    >/dev/null 2>&1
sleep 4

# ---------------------------------------------------------------------------
log "4/6 下发起点（第 1 次）"
# 高度不写在 position.z，而是编码进 orientation.z 的归一化比例：
#     z_goal = z_origin + dilate + |ori.z| * (z_size - 2*dilate)
# 目标高度 1.5m -> ratio = (1.5 - 0 - 0.2) / (5 - 0.4) = 0.2826
#
# 航点选取：本场景障碍密度很高（约 30 万占据体素），建议先用
# dump_cloud.py 分析地图，挑净空足够的点；这里给的是一组实测可行的坐标。
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  "{header: {frame_id: 'odom'}, pose: {position: {x: 0.0, y: -9.0, z: 1.5}, orientation: {z: 0.283, w: 1.0}}}" \
  >/dev/null 2>&1
sleep 3

log "5/6 下发终点（第 2 次，触发规划）"
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  "{header: {frame_id: 'odom'}, pose: {position: {x: 9.0, y: 9.0, z: 1.5}, orientation: {z: 0.283, w: 1.0}}}" \
  >/dev/null 2>&1

log "等待规划与飞行 (20s)..."
sleep 20

# ---------------------------------------------------------------------------
log "6/6 采集结果"
NPTS=$(timeout 8 rostopic echo -n1 /visualizer/trajectory/points 2>/dev/null | grep -c 'x:' || echo 0)
echo "  轨迹点数: $NPTS"
if [ "$NPTS" -gt 0 ] 2>/dev/null; then
    log "规划成功 ✔"
else
    warn "轨迹为空 —— 规划未触发。请检查 $LOG"
fi

echo "  速度: $(timeout 5 rostopic echo -n1 /visualizer/speed 2>/dev/null | head -1)"
echo "  CSV: $OUT_CSV  ($(wc -l < "$OUT_CSV" 2>/dev/null || echo 0) 行)"
