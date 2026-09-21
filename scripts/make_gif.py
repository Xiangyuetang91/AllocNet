#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_gif.py -- 把抓帧序列合成为演示 GIF

在**宿主机（Windows）**上运行，输入是从 VM 取回的 frames/ 目录。

用法
----
    python make_gif.py --frames frames/ --out docs/assets/teleop_demo.gif
    python make_gif.py --frames frames/ --out demo.gif \
        --crop 0,1080,0,1920 --width 900 --stride 2 --fps 8

参数说明
--------
    --crop   L,T,R,B 裁剪框（像素）。VM 全屏截图通常 1920x1080，
            裁掉桌面边框只留 RViz 区域能让 GIF 小很多也清楚很多。
    --width  输出宽度，等比缩放。默认 900。
    --stride 抽帧步长。抓帧 2fps 时用 stride=2 得到约 4 倍速的播放效果。
    --fps    GIF 播放帧率。
"""

import argparse
import glob
import os
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("需要 Pillow: pip install Pillow")


def parse_crop(text):
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        sys.exit("--crop 需要 4 个值: L,T,R,B")
    try:
        return tuple(int(float(p)) for p in parts)
    except ValueError:
        sys.exit("--crop 必须是数字: L,T,R,B")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="抓帧目录")
    ap.add_argument("--out", required=True, help="输出 .gif 路径")
    ap.add_argument("--crop", default=None, help="裁剪框 L,T,R,B")
    ap.add_argument("--width", type=int, default=900, help="输出宽度")
    ap.add_argument("--stride", type=int, default=1, help="抽帧步长")
    ap.add_argument("--fps", type=int, default=8, help="播放帧率")
    ap.add_argument("--max-frames", type=int, default=180,
                    help="最多使用多少帧（控制体积）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    if not files:
        sys.exit("在 %s 里没找到 .png" % args.frames)

    files = files[::max(1, args.stride)][:args.max_frames]
    crop = parse_crop(args.crop)
    print("[gif] 使用 %d 帧 (共发现 %d)" %
          (len(files), len(glob.glob(os.path.join(args.frames, "*.png")))))

    frames = []
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        if crop:
            im = im.crop(crop)
        if args.width and im.width != args.width:
            h = int(im.height * args.width / im.width)
            im = im.resize((args.width, h), Image.LANCZOS)
        # 量化到 256 色，显著减小体积
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
        if (i + 1) % 20 == 0:
            print("[gif]   已处理 %d/%d" % (i + 1, len(files)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    duration = int(1000 / args.fps)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)

    size_mb = os.path.getsize(args.out) / 1024.0 / 1024.0
    print("[gif] 写出 %s  (%.2f MB, %d 帧, %d fps)" %
          (args.out, size_mb, len(frames), args.fps))
    if size_mb > 25:
        print("[gif] 提示: 体积偏大，可加大 --stride 或减小 --width 后重试")


if __name__ == "__main__":
    main()
