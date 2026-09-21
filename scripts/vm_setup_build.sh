#!/usr/bin/env bash
# =============================================================================
# vm_setup_build.sh -- 在 Ubuntu 20.04 + ROS Noetic 虚拟机内一键构建 AllocNet
#
# 用法（在 VM 内执行）:
#     cd ~ && bash vm_setup_build.sh 2>&1 | tee ~/build.log
#
# 前置条件
#   * Ubuntu 20.04 + ROS Noetic (ros-desktop-full)
#   * 能访问 github.com 与 download.pytorch.org
#
# 已知坑（本脚本已处理）
#   1. utils.rosinstall 里 kr_param_map 用的是 SSH 地址 git@github.com:...，
#      没有配 SSH key 的机器会直接失败。这里改写成 HTTPS。
#   2. 虚拟机无 GPU 直通，必须用 CPU 版 libtorch + *_cpu.pt 模型。
#   3. learning_planner.hpp 的 device 默认值在部分版本里是 kGPU，需要确认为 kCPU。
# =============================================================================
set -uo pipefail

# 默认使用独立工作区。**不要用 ~/catkin_ws** —— 本机该目录已被其它项目
# （OverFOMO 等）占用，混入 AllocNet 会污染既有构建。
WS="${WS:-$HOME/allocnet_ws}"
LIBCURL_VER="2.0.0.dev20230301%2Bcpu"
LIBTORCH_URL="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBCURL_VER}.zip"

log() { echo -e "\033[32m[setup]\033[0m $*"; }
warn() { echo -e "\033[33m[warn ]\033[0m $*"; }
die() { echo -e "\033[31m[fail ]\033[0m $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
log "步骤 0/7 — 环境自检 / sudo 准备"
# ---------------------------------------------------------------------------
. /opt/ros/noetic/setup.bash 2>/dev/null || die "找不到 /opt/ros/noetic，请先安装 ROS Noetic"
log "ROS: $(rosversion -d 2>/dev/null || echo unknown)  发行版: $(lsb_release -ds 2>/dev/null)"

# sudo 处理：优先免密；否则用 VM_SUDO_PASS 环境变量刷新时间戳。
# 注意 sudo 时间戳按 tty 缓存，所以整个构建必须在同一个 ssh 会话里跑完。
if sudo -n true 2>/dev/null; then
    log "sudo: 免密可用"
elif [ -n "${VM_SUDO_PASS:-}" ]; then
    printf '%s\n' "$VM_SUDO_PASS" | sudo -S -v 2>/dev/null \
        && { log "sudo: 已用 VM_SUDO_PASS 刷新凭据"; } \
        || die "sudo 凭据刷新失败"
else
    die "sudo 需要密码：请先 export VM_SUDO_PASS=<密码> 再运行"
fi

# ---------------------------------------------------------------------------
log "步骤 1/7 — 安装系统依赖 (OMPL / Eigen / catkin tools / Python)"
# ---------------------------------------------------------------------------
sudo -n apt-get update -qq || warn "apt update 失败，继续尝试"
sudo -n apt-get install -y -qq \
    libompl-dev libeigen3-dev \
    python3-catkin-tools python3-pip python3-rosdep \
    libboost-all-dev cmake build-essential git wget unzip \
    || die "apt 安装依赖失败"

# ---------------------------------------------------------------------------
log "步骤 2/7 — 安装 OSQP 与 osqp-eigen (源码编译)"
# ---------------------------------------------------------------------------
if ! pkg-config --exists osqp 2>/dev/null && [ ! -e /usr/local/lib/libosqp.so ]; then
    cd /tmp
    rm -rf osqp
    git clone -b release-0.6.3 --depth 1 https://github.com/osqp/osqp.git \
        || die "克隆 osqp 失败"
    cd osqp
    git submodule update --init --recursive
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null || die "osqp cmake 失败"
    make -j"$(nproc)" >/dev/null || die "osqp 编译失败"
    sudo -n make install >/dev/null && sudo -n ldconfig
    log "osqp 安装完成"
else
    log "osqp 已存在，跳过"
fi

if [ ! -e /usr/local/lib/libOsqpEigen.so ] && [ ! -e /usr/lib/libOsqpEigen.so ]; then
    cd /tmp
    rm -rf osqp-eigen
    git clone --depth 1 https://github.com/robotology/osqp-eigen.git \
        || die "克隆 osqp-eigen 失败"
    cd osqp-eigen
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null || die "osqp-eigen cmake 失败"
    make -j"$(nproc)" >/dev/null || die "osqp-eigen 编译失败"
    sudo -n make install >/dev/null && sudo -n ldconfig
    log "osqp-eigen 安装完成"
else
    log "osqp-eigen 已存在，跳过"
fi

# ---------------------------------------------------------------------------
log "步骤 3/7 — 建立 catkin 工作区并拉取 AllocNet"
# ---------------------------------------------------------------------------
mkdir -p "$WS/src"
cd "$WS/src"

if [ ! -d AllocNet ]; then
    git clone -b feature-keyboard-teleop \
        https://github.com/Xiangyuetang91/AllocNet.git AllocNet \
        || die "克隆 AllocNet 失败（分支 feature-keyboard-teleop）"
else
    log "AllocNet 已存在，更新中"
    cd AllocNet
    git fetch origin
    git checkout feature-keyboard-teleop
    git pull --ff-only || warn "git pull 未快进，保持本地版本"
    cd ..
fi

# --- 坑 1：把 SSH 地址改写成 HTTPS -----------------------------------------
if [ -f AllocNet/src/utils.rosinstall ]; then
    if grep -q "git@github.com:" AllocNet/src/utils.rosinstall; then
        log "改写 utils.rosinstall 的 SSH 地址为 HTTPS"
        sed -i 's|git@github.com:|https://github.com/|g' \
            AllocNet/src/utils.rosinstall
    fi
fi

# --- 拉取 kr_param_map（提供 param_env / structure_map 地图节点）-----------
if [ ! -d kr_param_map ]; then
    git clone --depth 1 https://github.com/KumarRobotics/kr_param_map.git \
        kr_param_map || die "克隆 kr_param_map 失败"
fi
log "src 下包: $(ls -d */ 2>/dev/null | tr '\n' ' ')"

# ---------------------------------------------------------------------------
log "步骤 4/7 — 下载 libtorch (CPU, ~182MB)"
# ---------------------------------------------------------------------------
LT_DIR="$WS/src/AllocNet/src/planner/libtorch"
if [ ! -d "$LT_DIR" ]; then
    cd /tmp
    if [ ! -f libtorch.zip ]; then
        wget -q --show-progress -O libtorch.zip "$LIBTORCH_URL" \
            || curl -L -o libtorch.zip "$LIBTORCH_URL" \
            || die "下载 libtorch 失败"
    fi
    unzip -q -o libtorch.zip -d /tmp/lt_extract || die "解压 libtorch 失败"
    mkdir -p "$(dirname "$LT_DIR")"
    rm -rf "$LT_DIR"
    mv /tmp/lt_extract/libtorch "$LT_DIR" || die "移动 libtorch 失败"
    log "libtorch -> $LT_DIR"
else
    log "libtorch 已存在，跳过"
fi

# --- 坑 3：确认推理设备是 CPU ----------------------------------------------
HPP="$WS/src/AllocNet/src/planner/include/planner/learning_planner.hpp"
if [ -f "$HPP" ]; then
    if grep -q "torch::kGPU" "$HPP"; then
        log "学习规划器设备为 GPU，改写为 CPU（本 VM 无 GPU 直通）"
        sed -i 's/torch::kGPU/torch::kCPU/g' "$HPP"
    fi
    grep -n "device(torch::k" "$HPP" | head -2
fi

# ---------------------------------------------------------------------------
log "步骤 5/7 — catkin build"
# ---------------------------------------------------------------------------
cd "$WS"
catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
catkin build -j"$(nproc)" 2>&1 | tail -30
# catkin build 的失败不一定反映在管道退出码上，用产物存在性判断
if [ ! -x "$WS/devel/lib/planner/learning_planning" ]; then
    die "learning_planning 未生成，请查看上方编译错误"
fi
log "编译成功: $WS/devel/lib/planner/learning_planning"

# ---------------------------------------------------------------------------
log "步骤 6/7 — 给 Python 脚本加可执行权限"
# ---------------------------------------------------------------------------
chmod +x "$WS"/src/AllocNet/src/planner/scripts/*.py 2>/dev/null
ls -l "$WS"/src/AllocNet/src/planner/scripts/

# ---------------------------------------------------------------------------
log "步骤 7/7 — 完成"
# ---------------------------------------------------------------------------
cat <<EOF

构建完成。后续运行：

    source $WS/devel/setup.bash
    roslaunch planner teleop_planning.launch

键盘节点在独立终端里跑（需要 TTY）：

    source $WS/devel/setup.bash
    rosrun planner teleop_keyboard.py

EOF
