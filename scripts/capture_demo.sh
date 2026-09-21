#!/usr/bin/env bash
# =============================================================================
# capture_demo.sh -- 在 VM 内自动演示键盘遥操作 + AllocNet 规划，并抓帧
#
# 产出
#   $OUT_DIR/frames/*.png   抓帧序列（交给 make_gif.py 合成 GIF）
#   $OUT_DIR/trajectory.csv 轨迹日志（交给 plot_trajectory.py 出图）
#
# 用法
#     bash capture_demo.sh
#     OUT_DIR=~/allocnet_capture FPS=2 bash capture_demo.sh
#
# 依赖: xdotool (发送真实键盘事件), scrot 或 imagemagick (抓屏)
#     sudo apt-get install -y xdotool scrot imagemagick x11-apps
# =============================================================================
set -uo pipefail

WS="${WS:-$HOME/catkin_ws}"
OUT_DIR="${OUT_DIR:-$HOME/allocnet_capture}"
FPS="${FPS:-2}"                 # 抓帧频率
DURATION="${DURATION:-90}"      # 总时长（秒）
DISPLAY_ID="${DISPLAY_ID:-:0}"

FRAMES="$OUT_DIR/frames"
CSV="$OUT_DIR/trajectory.csv"

log()  { echo -e "\033[32m[capture]\033[0m $*"; }
warn() { echo -e "\033[33m[warn   ]\033[0m $*"; }

export DISPLAY="$DISPLAY_ID"
mkdir -p "$FRAMES"
rm -f "$FRAMES"/*.png

# --- 选择抓图工具 -----------------------------------------------------------
if command -v scrot >/dev/null 2>&1; then
    grab() { scrot -o "$1" 2>/dev/null; }
elif command -v import >/dev/null 2>&1; then
    grab() { import -window root "$1" 2>/dev/null; }
else
    warn "没有 scrot / import，尝试安装"
    sudo apt-get install -y -qq scrot imagemagick xdotool
    grab() { scrot -o "$1" 2>/dev/null; }
fi

# --- 确认 X 显示可用 --------------------------------------------------------
if ! xdpyinfo >/dev/null 2>&1; then
    warn "DISPLAY=$DISPLAY_ID 不可用。若在无头环境，请先启动图形会话："
    warn "    startx &    （或在 VMware 控制台里登录桌面）"
    exit 1
fi

log "输出目录: $OUT_DIR"
log "抓帧: ${FPS} fps，最长 ${DURATION}s"

# ---------------------------------------------------------------------------
log "1/5 — 启动 ROS 仿真 (teleop_planning.launch, 不含 RViz)"
# ---------------------------------------------------------------------------
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

# 记录节点单独起，写入我们指定的 CSV
rosrun planner record_trajectory.py _output:="$CSV" >"$OUT_DIR/recorder.log" 2>&1 &
REC_PID=$!

roslaunch planner teleop_planning.launch record:=false use_gui:=true \
    >"$OUT_DIR/launch.log" 2>&1 &
LAUNCH_PID=$!

log "等待地图与 RViz 就绪 (25s)…"
sleep 25

if ! rostopic list 2>/dev/null | grep -q "/structure_map/global_gridmap"; then
    warn "地图话题未出现，检查 $OUT_DIR/launch.log"
fi

# ---------------------------------------------------------------------------
log "2/5 — 打开终端跑键盘节点，并用 xdotool 发送按键"
# ---------------------------------------------------------------------------
if command -v xdotool >/dev/null 2>&1; then
    # 在图形终端里起键盘节点
    (xterm -title teleop -geometry 100x30+10+10 \
        -e "bash -c 'source $WS/devel/setup.bash; \
                     rosrun planner teleop_keyboard.py; exec bash'" \
        >/dev/null 2>&1 &) || \
    (gnome-terminal --title=teleop -- \
        bash -c "source $WS/devel/setup.bash; rosrun planner teleop_keyboard.py" \
        >/dev/null 2>&1 &)
    sleep 8

    TELEOP_WIN=$(xdotool search --name "teleop" | head -1 || true)
    if [ -n "${TELEOP_WIN:-}" ]; then
        xdotool windowactivate "$TELEOP_WIN" 2>/dev/null
        sleep 1
        log "找到键盘终端窗口 $TELEOP_WIN"
    else
        warn "未找到键盘终端窗口，改用 topic 方式推进游标"
    fi

    # 键盘脚本：与 offline_demo.py 的按键序列对应
    send_keys() {
        local win="$1"; shift
        for k in "$@"; do
            if [ -n "$win" ]; then
                xdotool key --window "$win" "$k" 2>/dev/null || true
            fi
            sleep 0.35
        done
    }

    KEYSEQ_W="w w w w w w w w w w w w w w w w"
    KEYSEQ_D="d d d d d d d d"
    KEYSEQ_I="i i i i"
else
    warn "未安装 xdotool，无法发送真实键盘事件"
    TELEOP_WIN=""
fi

# ---------------------------------------------------------------------------
log "3/5 — 开始抓帧"
# ---------------------------------------------------------------------------
CAPTURE_PID=""
(
    interval=$(awk "BEGIN{printf \"%.3f\", 1/$FPS}")
    i=0
    end=$(( $(date +%s) + DURATION ))
    while [ "$(date +%s)" -lt "$end" ]; do
        grab "$(printf '%s/frame_%05d.png' "$FRAMES" "$i")"
        i=$((i + 1))
        sleep "$interval"
    done
    echo "$i" > "$OUT_DIR/frame_count.txt"
) &
CAPTURE_PID=$!

# ---------------------------------------------------------------------------
log "4/5 — 执行键盘序列并下发航点"
# ---------------------------------------------------------------------------
sleep 3

if [ -n "${TELEOP_WIN:-}" ]; then
    log "按键: 后退 S x16 (游标移到起点)"
    send_keys "$TELEOP_WIN" s s s s s s s s s s s s s s s s
    log "按键: 左移 A x4"
    send_keys "$TELEOP_WIN" a a a a
    log "按键: 上升 I x2"
    send_keys "$TELEOP_WIN" i i
    sleep 1
    log "按键: G (设置起点)"
    send_keys "$TELEOP_WIN" g
else
    # 回退：直接发布等价目标点
    log "回退方案 — 直接发布起点"
    rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
        "{header: {frame_id: 'odom'}, pose: {position: {x: -8.0, y: 2.0, z: 1.5}, \
          orientation: {z: 0.30, w: 1.0}}}" >/dev/null 2>&1
fi
sleep 3

log "设置终点并触发 AllocNet 规划"
if [ -n "${TELEOP_WIN:-}" ]; then
    # 直接把终点用 topic 下发（键盘走到终点太慢，且要保证复现性）
    rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
        "{header: {frame_id: 'odom'}, pose: {position: {x: 8.0, y: 2.0, z: 1.6}, \
          orientation: {z: 0.32, w: 1.0}}}" >/dev/null 2>&1
else
    rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
        "{header: {frame_id: 'odom'}, pose: {position: {x: 8.0, y: 2.0, z: 1.6}, \
          orientation: {z: 0.32, w: 1.0}}}" >/dev/null 2>&1
fi

log "等待规划与飞行完成 (20s)…"
sleep 20

# ---------------------------------------------------------------------------
log "5/5 — 收尾"
# ---------------------------------------------------------------------------
wait "$CAPTURE_PID" 2>/dev/null

kill "$REC_PID" 2>/dev/null
sleep 2
kill "$LAUNCH_PID" 2>/dev/null
sleep 3

N=$(ls "$FRAMES"/*.png 2>/dev/null | wc -l)
log "抓到 $N 帧 -> $FRAMES"
log "轨迹日志 -> $CSV  ($(wc -l < "$CSV" 2>/dev/null || echo 0) 行)"

# 打包，方便用 vmrun copyFileFromGuestToHost 取回
tar czf "$OUT_DIR.tar.gz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")" \
    2>/dev/null && log "已打包 -> $OUT_DIR.tar.gz"
