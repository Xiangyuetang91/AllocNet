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
# 日志：直接追加写文件，**不要用 `exec > >(tee ...)`**。
# 进程替换在 nohup/ssh 分离运行的场景下会因管道无人读取而阻塞
# （表现为进程停在 pipe_read，日志不再增长）。
BUILD_LOG="${BUILD_LOG:-$HOME/allocnet_build.log}"
LIBCURL_VER="2.0.0.dev20230301%2Bcpu"
LIBTORCH_URL="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBCURL_VER}.zip"

log() { echo -e "\033[32m[setup]\033[0m $*"; }
warn() { echo -e "\033[33m[warn ]\033[0m $*"; }
die() { echo -e "\033[31m[fail ]\033[0m $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
log "步骤 0/7 — 环境自检 / sudo 准备"
# ---------------------------------------------------------------------------
# ⚠ 必须临时关闭 `set -u` 才能 source ROS 的 setup.bash。
#
# ROS Noetic 的 /opt/ros/noetic/etc/catkin/profile.d/1.ros_distro.sh
# 第 3 行会引用尚未定义的 ROS_DISTRO。在本脚本 `set -u` 生效时，
# 这会直接判定为 unbound variable 并**终止整个脚本**，
# 且因为发生在 source 内部，连 `|| die` 都不会执行 —— 表现为
# 脚本在打印完步骤标题后无声退出，极难定位。
#
# 实测：
#     bash -c 'set -u;  . /opt/ros/noetic/setup.bash'  -> ROS_DISTRO: unbound variable
#     bash -c 'set +u; . /opt/ros/noetic/setup.bash'  -> 正常，catkin 模块可用
#
# source 之后必须恢复 -u，并借此机会把 ROS 的 PATH / PYTHONPATH 真正导入，
# 否则 catkin_make 会因 `No module named 'catkin'` 而失败。
set +u
if [ ! -f /opt/ros/noetic/setup.bash ]; then
    set -u
    die "找不到 /opt/ros/noetic/setup.bash，请先安装 ROS Noetic"
fi
. /opt/ros/noetic/setup.bash
set -u

log "ROS: ${ROS_DISTRO:-unknown}  发行版: $(lsb_release -ds 2>/dev/null)"
log "catkin 模块: $(python3 -c 'import catkin; print("OK")' 2>/dev/null || echo MISSING)"

# sudo 处理。
#
# 重要：sudo 的时间戳缓存是**按 tty 隔离**的。在非交互 ssh 会话里用
# `sudo -S -v` 刷新出来的凭据，后续 `sudo -n` 在另一 tty 上下文中看不到，
# 会再次索要密码并卡死。实测：
#     REFRESH_OK                         <- sudo -S -v 成功
#     after: sudo: a password is required <- 紧接着 sudo -n 就失败
# 因此这里不依赖缓存，改为每条 sudo 命令都用 sudo -S 直接喂密码。
if sudo -n true 2>/dev/null; then
    log "sudo: 免密可用"
    SUDO() { sudo -n "$@"; }
elif [ -n "${VM_SUDO_PASS:-}" ]; then
    log "sudo: 使用 VM_SUDO_PASS 逐条认证"
    # shellcheck disable=SC2317
    SUDO() { printf '%s\n' "$VM_SUDO_PASS" | sudo -S -p '' "$@"; }
else
    die "sudo 需要密码：请先 export VM_SUDO_PASS=<密码> 再运行"
fi

# ---------------------------------------------------------------------------
log "步骤 0b/7 — 修复 apt 源（ROS 密钥过期 / hosts 劫持）"
# ---------------------------------------------------------------------------
# 本机实测遇到两个会让 apt 静默失败的问题：
#   1. ROS apt 仓库的 GPG 密钥已过期：
#        uid [ expired] Open Robotics <info@osrfoundation.org>
#      apt 会报 EXPKEYSIG F42ED6FBAB17C654 并拒绝该源。
#   2. /etc/hosts 把 packages.ros.org 指向 198.18.0.31（代理/占位 IP），
#      导致下载 404。
# 这里先自愈，否则后面的依赖装不上。

if grep -qE '^\s*[0-9.]+\s+packages\.ros\.org' /etc/hosts 2>/dev/null; then
    log "检测到 /etc/hosts 劫持 packages.ros.org，移除该条目"
    SUDO cp /etc/hosts /etc/hosts.allocnet.bak
    SUDO sed -i -E '/^\s*[0-9.]+\s+packages\.ros\.org/d' /etc/hosts
fi

if apt-key list 2>/dev/null | grep -q 'expired.*Open Robotics\|Open Robotics.*expired'; then
    log "ROS 密钥已过期，重新获取"
    timeout 90 curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
        | SUDO apt-key add - 2>/dev/null || warn "获取 ros.asc 失败，继续尝试"
fi

# 非致命：apt update 可能因其它第三方源（librealsense 等）报错，
# 只要 Ubuntu 官方源可用即可继续。
SUDO apt-get update -qq 2>&1 | grep -vE '^(W:|Get:|Hit:|Ign:)' | head -5 \
    || warn "apt update 有警告，继续尝试"

# ---------------------------------------------------------------------------
log "步骤 1/7 — 安装系统依赖 (OMPL / Eigen / Python)"
# ---------------------------------------------------------------------------
# libsdl1.2-dev / libsdl-image1.2-dev：kr_param_map 的 param_env 包
# （提供 structure_map 地图节点）依赖 rosdep 键 sdl / sdl-image，
# 缺了它 param_env 编译不过，而地图节点是整套仿真的前置条件。
#
# 注意 catkin_tools 单独装且**允许失败**：某些镜像上
# python3-catkin-tools 会 404。ROS 自带的 catkin_make 已足够构建，
# 后面的 build 步骤会自动在两者间选择。
SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libompl-dev libeigen3-dev \
    python3-pip python3-rosdep \
    libboost-all-dev cmake build-essential git wget unzip \
    libsdl1.2-dev libsdl-image1.2-dev \
    || warn "部分依赖安装失败，继续尝试"

SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3-catkin-tools 2>/dev/null \
    || warn "python3-catkin-tools 装不上，将回退使用 catkin_make"

[ -d /usr/include/ompl ] || die "libompl-dev 未安装成功（OMPL 是编译必需项）"
log "OMPL: OK   Eigen: $([ -d /usr/include/eigen3 ] && echo OK || echo MISSING)"
log "构建器: $(command -v catkin >/dev/null && echo catkin_tools || echo catkin_make)"

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
    SUDO make install >/dev/null && SUDO ldconfig
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
    SUDO make install >/dev/null && SUDO ldconfig
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
THREADS="$(nproc)"

# catkin_make / catkin 都在 /opt/ros/noetic/bin 下，非交互 shell 不会自动加载，
# 必须显式加入 PATH（子 shell 里 source 会静默终止脚本，见步骤 0 的说明）。
if [ -d /opt/ros/noetic/bin ]; then
    case ":$PATH:" in
        *":/opt/ros/noetic/bin:"*) ;;
        *) PATH="/opt/ros/noetic/bin:$PATH" ;;
    esac
    export PATH
fi
log "catkin 工具: catkin=$(command -v catkin || echo none)  catkin_make=$(command -v catkin_make || echo none)"

# libtorch 是 _GLIBCXX_USE_CXX11_ABI=1 的预编译包，必须与项目 ABI 一致，
# 否则链接期会报大量 std::__cxx11 符号缺失。
CMAKE_EXTRA="-DCMAKE_BUILD_TYPE=Release -D_GLIBCXX_USE_CXX11_ABI=1"

# 优先 catkin_tools；不可用则回退 ROS 自带的 catkin_make。
# 只构建本实验需要的三个包：
#   planner    AllocNet 推理 + QP 轨迹优化（含键盘节点）
#   param_env  地图生成，发布 /structure_map/global_gridmap
#   vicon_env  kr_param_map 的姊妹包，param_env 的构建依赖
if command -v catkin >/dev/null 2>&1; then
    log "使用 catkin_tools 构建 planner / param_env / vicon_env"
    catkin config --extend /opt/ros/noetic \
        --cmake-args $CMAKE_EXTRA >/dev/null 2>&1
    catkin build planner param_env vicon_env -j"$THREADS" 2>&1 | tail -45
    if [ ! -x "$WS/devel/lib/planner/learning_planning" ]; then
        warn "指定包构建未产出目标，回退全量构建"
        catkin build -j"$THREADS" 2>&1 | tail -30
    fi
else
    log "catkin_tools 不可用，回退 catkin_make"
    # catkin_make 不支持按包构建，需要临时屏蔽无关包。
    # 用一个只含所需包的软链接视图，避免其它包的编译错误打断流程。
    mkdir -p "$WS/src_allocnet"
    for p in AllocNet kr_param_map; do
        [ -e "$WS/src_allocnet/$p" ] || ln -sfn "$WS/src/$p" "$WS/src_allocnet/$p"
    done
    ( cd "$WS" && catkin_make --source src_allocnet \
        -DCMAKE_BUILD_TYPE=Release -D_GLIBCXX_USE_CXX11_ABI=1 \
        -j"$THREADS" 2>&1 | tail -45 )
fi

if [ ! -x "$WS/devel/lib/planner/learning_planning" ]; then
    die "learning_planning 未生成，请查看上方编译错误"
fi
log "编译成功: $WS/devel/lib/planner/learning_planning"
ls -la "$WS/devel/lib/planner/" 2>/dev/null
ls -la "$WS/devel/lib/param_env/" 2>/dev/null

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
