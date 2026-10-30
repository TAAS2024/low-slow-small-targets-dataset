#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  低慢小数据集 — 审查端全链路 Demo
  CDFF Validator Pipeline: S6 → S7 → S8 → S9 (短路求值 + FailureBuffer)
═══════════════════════════════════════════════════════════════════════════════

演示 5 个测试用例覆盖所有阶段：
  用例1 ✅ 合格帧 — 全链路 PASS
  用例2 ❌ 低质量帧 — S6 截停 (BRISQUE 超标)
  用例3 ❌ 尺寸异常 — S7 截停 (bbox 与距离不匹配)
  用例4 ❌ 轨迹跳帧 — S8 截停 (位置突变)
  用例5 ❌ 检测失败 — S9 截停 (Mock YOLO 不检测)

运行:
  python3 demo_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 路径设置
ROOT = Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/6-Validator")
DEMO_DIR = ROOT / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# 添加各阶段路径
for stage in ["S6 RGB 图像质量检查", "S7 无人机与场景一致性检查", "S8 轨迹物理合理性检查", "S9 YOLO结果检查", "V5-pipeline"]:
    p = str(ROOT / stage / "code")
    if p not in sys.path:
        sys.path.insert(0, p)

from quality_validator import QualityValidator           # noqa: E402
from consistency_validator import ConsistencyValidator    # noqa: E402
from trajectory_validator import TrajectoryValidator      # noqa: E402
from detection_validator import DetectionValidator        # noqa: E402
from validator_pipeline import ValidatorPipeline, PipelineResult  # noqa: E402
from failure_buffer import FailureBuffer                  # noqa: E402


# ============================================================================
# 工具函数：生成测试图像
# ============================================================================

def make_clean_rgb(size: int = 256) -> np.ndarray:
    """生成干净的合成 RGB 图像：蓝天 + 小无人机。"""
    img = np.zeros((size, size, 3), np.uint8)
    # 蓝天渐变
    for y in range(size):
        b = int(180 + (255 - 180) * y / size)  # 上深下浅
        img[y, :, 0] = b
    img[:, :, 1] = 140  # 绿
    img[:, :, 2] = 50   # 红 → 蓝天
    # 无人机：深灰色方块
    cx, cy = size // 2, size // 2
    s = 20
    img[cy-s:cy+s, cx-s:cx+s] = [60, 60, 60]
    # 旋翼：浅灰十字
    img[cy-s-3:cy-s, cx-2:cx+2] = [180, 180, 180]
    img[cy+s:cy+s+3, cx-2:cx+2] = [180, 180, 180]
    img[cy-2:cy+2, cx-s-3:cx-s] = [180, 180, 180]
    img[cy-2:cy+2, cx+s:cx+s+3] = [180, 180, 180]
    return img


def make_blurry_rgb(size: int = 256) -> np.ndarray:
    """生成模糊/噪声 RGB 图像。"""
    img = make_clean_rgb(size)
    # 强高斯噪声 → BRISQUE 高分
    noise = np.random.randint(0, 80, (size, size, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.5, noise, 0.5, 0)
    # 再模糊
    img = cv2.GaussianBlur(img, (15, 15), 0)
    return img


def make_ir_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """模拟 IR 转换：灰度 + 轻微热特征。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 模拟热反转：暗色(无人机)变亮
    ir = 255 - gray
    ir = np.clip(ir + np.random.normal(0, 5, ir.shape), 0, 255)
    return ir.astype(np.uint8)


# ============================================================================
# Demo 主流程
# ============================================================================

def print_header(title: str):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def run_demo():
    # 初始化 FailureBuffer
    fb_path = DEMO_DIR / "failure_log.jsonl"
    if fb_path.exists():
        fb_path.unlink()
    fb = FailureBuffer(str(fb_path), threshold=50,
                       archive_dir=str(DEMO_DIR / "crops"))

    # Demo: 放宽参数展示全链路流程（真实校准值见各阶段 config）
    s0 = QualityValidator(brisque_thr=200.0)

    from lighting_consistency import LightingConsistencyChecker
    s1 = ConsistencyValidator(
        lighting_checker=LightingConsistencyChecker(kl_threshold=100.0),
    )
    s2 = TrajectoryValidator()
    # S9 用 Mock 避免依赖 YOLO 下载
    s3 = DetectionValidator(iou_thr=0.5)

    pipeline = ValidatorPipeline(
        s0=s0, s1=s1, s2=s2, s3=s3, failure_buffer=fb,
    )

    size = 256

    # ═══════════════════════════════════════════════════════════════════════
    print_header("用例1: ✅ 合格帧 — 预期全链路 PASS")
    # ═══════════════════════════════════════════════════════════════════════
    rgb1 = make_clean_rgb(size)
    sample1 = {
        "id": "case1_good",
        "rgb": rgb1,
        "rgb_bbox": np.array([0.5, 0.5, 0.156, 0.156]),  # 40/256, kite@5m exp_w=0.173
        "drone_category": "kite",
        "distance_m": 5,
    }
    res1 = pipeline.validate(sample1)
    print(res1.summary())

    # ═══════════════════════════════════════════════════════════════════════
    print_header("用例2: ❌ 低质量帧 — 预期 S6 截停 (BRISQUE 超标)")
    # ═══════════════════════════════════════════════════════════════════════
    # 用例2 专用：低 BRISQUE 阈值演示 S6 截停
    s0_strict = QualityValidator(brisque_thr=50.0)
    pipeline2 = ValidatorPipeline(
        s0=s0_strict, s1=s1, s2=s2, s3=s3, failure_buffer=fb,
    )
    rgb2 = make_blurry_rgb(size)
    sample2 = {
        "id": "case2_blurry",
        "rgb": rgb2,
        "rgb_bbox": np.array([0.5, 0.5, 0.156, 0.156]),
        "drone_category": "kite",
        "distance_m": 5,
    }
    res2 = pipeline2.validate(sample2)
    print(res2.summary())
    # 保存对比图
    cv2.imwrite(str(DEMO_DIR / "case1_clean.png"), rgb1)
    cv2.imwrite(str(DEMO_DIR / "case2_blurry.png"), rgb2)

    # ═══════════════════════════════════════════════════════════════════════
    print_header("用例3: ❌ 尺寸异常 — 预期 S7 截停 (200m 处 bbox 太大)")
    # ═══════════════════════════════════════════════════════════════════════
    rgb3 = make_clean_rgb(size)
    sample3 = {
        "id": "case3_bad_size",
        "rgb": rgb3,
        "rgb_bbox": np.array([0.5, 0.5, 0.15, 0.15]),  # 200m处期望~0.001→必FAIL
        "drone_category": "small_quadcopter",
        "distance_m": 200,
    }
    res3 = pipeline.validate(sample3)
    print(res3.summary())

    # ═══════════════════════════════════════════════════════════════════════
    print_header("用例4: ❌ 轨迹跳帧 — 预期 S8 截停 (位置突变)")
    # ═══════════════════════════════════════════════════════════════════════
    frames = []
    for i in range(5):
        fr = {
            "id": f"case4_seq/frame_{i}",
            "rgb": make_clean_rgb(size),
            "rgb_bbox": np.array([0.3 + i * 0.05, 0.5, 0.156, 0.156]),
            "drone_category": "kite",
            "distance_m": 5,
            "frame_id": i,
            "bbox": [0.3 + i * 0.05, 0.5, 0.156, 0.156],
            "speed": 0.05,
            "direction": 0.0,
        }
        frames.append(fr)

    # 注入跳帧
    frames[3]["bbox"] = [0.9, 0.9, 0.156, 0.156]
    frames[3]["rgb_bbox"] = np.array([0.9, 0.9, 0.156, 0.156])
    # 调整后续帧
    frames[4]["bbox"] = [0.95, 0.9, 0.156, 0.156]
    frames[4]["rgb_bbox"] = np.array([0.95, 0.9, 0.156, 0.156])

    res4 = pipeline.validate_sequence(frames)
    print(res4.summary())

    # ═══════════════════════════════════════════════════════════════════════
    print_header("用例5: ❌ 检测失败 — 预期 S9 截停 (Mock: YOLO 无预测)")
    # ═══════════════════════════════════════════════════════════════════════
    # 用 Mock 子类注入空预测，模拟 YOLO 未检测到目标
    class MockDetectionValidator(DetectionValidator):
        @property
        def model(self):
            return None
        def _detect(self, image):
            return np.empty((0, 4))

    s3_mock = MockDetectionValidator(iou_thr=0.5)
    pipeline_mock = ValidatorPipeline(
        s0=s0, s1=s1, s2=s2, s3=s3_mock, failure_buffer=fb,
    )

    rgb5 = make_clean_rgb(size)
    sample5 = {
        "id": "case5_undetectable",
        "rgb": rgb5,
        "rgb_bbox": np.array([0.1, 0.1, 0.156, 0.156]),
        "gt_bboxes": np.array([[0.1, 0.1, 0.156, 0.156]]),
        "drone_category": "kite",
        "distance_m": 5,
    }
    res5 = pipeline_mock.validate(sample5)
    print(res5.summary())

    # ═══════════════════════════════════════════════════════════════════════
    print_header("📊 FailureBuffer 统计")
    # ═══════════════════════════════════════════════════════════════════════
    stats = fb.stats()
    print(f"  总失败数: {stats['total']}")
    print(f"  按阶段:   {json.dumps(stats['by_stage'], ensure_ascii=False)}")
    print(f"  按失败码: {json.dumps(stats['by_code'], ensure_ascii=False)}")
    print(f"  日志文件: {fb_path}")

    # 展示日志内容
    print(f"\n  日志条目预览:")
    for entry in fb.load_all():
        print(f"    [{entry['stage']}] {entry['failure_code']} "
              f"(sample={entry['sample_id']})")

    # ═══════════════════════════════════════════════════════════════════════
    print_header("📋 汇总")
    # ═══════════════════════════════════════════════════════════════════════
    results = [
        ("用例1 合格帧", res1),
        ("用例2 低质量", res2),
        ("用例3 尺寸异常", res3),
        ("用例4 轨迹跳帧", res4),
        ("用例5 检测失败", res5),
    ]
    for name, r in results:
        status = "✅ PASS" if r.passed else f"❌ {r.failure_stage}: {r.failure_code}"
        print(f"  {name:12s} → {status}")

    print(f"\n  FailureBuffer: {fb.count()} 条失败记录 → {fb_path}")
    print(f"  Demo 图像:     {DEMO_DIR}/case1_clean.png, case2_blurry.png")
    print()


if __name__ == "__main__":
    run_demo()
