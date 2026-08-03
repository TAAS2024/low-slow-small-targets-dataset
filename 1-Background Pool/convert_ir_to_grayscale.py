"""
IR 灰度统一处理脚本
用途：将混合了伪彩色和灰度的 IR 帧全部统一为三通道灰度 JPG
日期：2026-07-29

问题背景：
- DroneMMset Inf01（236 帧）: 伪彩色热力图（紫-橙调色板）
- DroneMMset Inf02（266 帧）: 纯灰度
- Anti-UAV-RGBT: 接近灰度但含微量编码色差
→ 混合训练 LoRA 会导致颜色分布混乱，生成 artifact

方案：cv2.COLOR_BGR2GRAY → COLOR_GRAY2BGR（保留三通道 JPEG 兼容性）

使用：
    python convert_ir_to_grayscale.py <目标目录>

已验证的目录：
    IR_lora_training_samples/       502 帧 ✅
    IR_raw_frames/                 3,804 帧 ✅
    IR_raw_frames_antiuav/        14,844 帧 ✅（抽帧时已用 ffmpeg format=gray）
"""

import cv2
import numpy as np
import sys
from pathlib import Path


def convert_directory(target_dir: str):
    files = sorted(Path(target_dir).glob("*.jpg"))
    if not files:
        print(f"❌ 目录为空或无 JPG 文件: {target_dir}")
        return

    print(f"处理 {len(files)} 个 JPG 文件...")
    gray_count, skip_count, fail_count = 0, 0, 0

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  ⚠ 无法读取: {f.name}")
            fail_count += 1
            continue

        # 检查是否已经是灰度（B≈G≈R）
        diff_bg = np.abs(img[:, :, 0].astype(float) - img[:, :, 1].astype(float)).mean()
        diff_br = np.abs(img[:, :, 0].astype(float) - img[:, :, 2].astype(float)).mean()

        if diff_bg < 1.0 and diff_br < 1.0:
            skip_count += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(f), gray_3ch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        gray_count += 1

        if (gray_count + skip_count) % 500 == 0:
            print(f"  进度: {gray_count + skip_count}/{len(files)} (转灰度={gray_count}, 已是灰度={skip_count}, 失败={fail_count})")

    print(f"\n✅ 完成: 转灰度={gray_count}, 已是灰度={skip_count}, 失败={fail_count}")
    print(f"   目标目录: {target_dir}")

    # 随机抽 3 张验证
    import random
    verify_files = random.sample(files, min(3, len(files)))
    all_ok = True
    for vf in verify_files:
        arr = cv2.imread(str(vf))
        d_bg = np.abs(arr[:, :, 0].astype(float) - arr[:, :, 1].astype(float)).mean()
        all_ok = all_ok and (d_bg < 1.0)
    print(f"   {'✅ 全部灰度验证通过' if all_ok else '❌ 存在非灰度文件！'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python convert_ir_to_grayscale.py <目标目录>")
        sys.exit(1)

    convert_directory(sys.argv[1])
