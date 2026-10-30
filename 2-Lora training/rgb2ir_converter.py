#!/usr/bin/env python3
"""
RGB → IR (白热风格) 批量转换器
将普通 RGB 图像转换为白热红外风格，用于背景 LoRA 训练数据准备。

用法:
    # 单张转换
    python rgb2ir_converter.py input.jpg -o output.png

    # 批量转换整个目录
    python rgb2ir_converter.py ./rgb_dir/ -o ./ir_dir/

    # 批量转换，指定输出格式
    python rgb2ir_converter.py ./rgb_dir/ -o ./ir_dir/ --ext .png

原理:
    RGB → 灰度 → 反转(白热) → 三通道重建(微蓝调)
    三通道有色差 → VAE 可正常编解码 → LoRA 可正常训练
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def rgb_to_whitehot(rgb_bgr: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """
    将 BGR 图像转换为白热 IR 风格 (RGB 输出)。

    Args:
        rgb_bgr: OpenCV 读取的 BGR 图像, shape (H, W, 3)
        gain: 灰度对比度增益（默认 1.0 = 不增强）。>1 时围绕中灰 128
            线性拉伸灰度，增强目标-背景可辨识度，用于补偿低照度场景
            （dusk/night）下白热转换的对比度衰减。

    Returns:
        RGB 白热 IR 图像, shape (H, W, 3)
    """
    # BGR → 灰度
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)

    # 可选对比度增益（模拟 IR 监控的 gain 调节）
    if gain != 1.0:
        gray = np.clip(
            (gray.astype(np.float32) - 128.0) * gain + 128.0, 0, 255
        ).astype(np.uint8)

    # 白热 = 反转灰度 (亮区 = 热源 = 高灰度值)
    white_hot = 255 - gray

    # 三通道重建，蓝通道略微衰减 → 微蓝调，更像真实 IR 监控画面
    ir = np.stack(
        [
            white_hot,
            white_hot,
            (white_hot * 0.85).astype(np.uint8),
        ],
        axis=-1,
    )

    return ir  # RGB 顺序


def convert_single(input_path: str, output_path: str) -> None:
    """转换单张图片"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图片: {input_path}")

    ir = rgb_to_whitehot(img)
    Image.fromarray(ir).save(output_path)
    print(f"✓ {input_path} → {output_path}")


def convert_batch(input_dir: str, output_dir: str, ext: str = ".jpg") -> int:
    """批量转换目录下所有图片"""
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    input_path = Path(input_dir)
    files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in valid_exts
    )

    if not files:
        print(f"⚠ 目录 {input_dir} 中未找到图片文件", file=sys.stderr)
        return 0

    count = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"✗ 跳过无法读取的文件: {f}", file=sys.stderr)
            continue

        ir = rgb_to_whitehot(img)
        out_name = f"{f.stem}{ext}"
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(ir).save(out_path)
        count += 1

    print(f"✓ 转换完成: {count}/{len(files)} 张 → {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="RGB → IR (白热风格) 批量转换器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="输入图片路径 或 输入目录")
    parser.add_argument("-o", "--output", required=True, help="输出路径 或 输出目录")
    parser.add_argument("--ext", default=".jpg", help="批量转换时输出格式 (默认: .jpg)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        convert_single(str(input_path), args.output)
    elif input_path.is_dir():
        convert_batch(str(input_path), args.output, args.ext)
    else:
        print(f"✗ 输入路径不存在: {args.input}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
