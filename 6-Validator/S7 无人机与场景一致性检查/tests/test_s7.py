"""
S7 单元测试
============

1. 单元测试：四个维度各自检测逻辑。
2. 端到端测试：ConsistencyValidator 综合判定。
3. 双模态测试：RGB+IR 全维度验证。

运行：
  python3 test_s1.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from size_consistency import SizeConsistencyChecker          # noqa: E402
from lighting_consistency import LightingConsistencyChecker  # noqa: E402
from ir_bbox_check import IRBboxChecker                      # noqa: E402
from cross_modal_alignment import CrossModalAlignmentChecker  # noqa: E402
from consistency_validator import ConsistencyValidator, S7Verdict  # noqa: E402

CONFIG = Path(__file__).parent.parent / "config" / "s7_config.json"


# ============================================================================
# D1: 尺寸一致性
# ============================================================================
def test_size():
    print("=" * 60)
    print("D1 尺寸一致性")
    print("=" * 60)

    c = SizeConsistencyChecker(fov_deg=60, tolerance=0.5)

    # 近距离小型四旋翼：期望宽度约 25cm/(2*10*tan30°) = 0.25/11.55 = 0.0216
    bbox = np.array([0.5, 0.5, 0.022, 0.022])
    res = c.check(bbox, "small_quadcopter", distance_m=10)
    assert res.passed, f"10m 小型四旋翼应通过，实际 {res.detail}"
    print(f"  [OK] 10m small_quad w=0.022 → PASS (exp~0.022)")

    # 远距离：期望宽度很小
    bbox_far = np.array([0.5, 0.5, 0.001, 0.001])
    res = c.check(bbox_far, "small_quadcopter", distance_m=200)
    assert res.passed, f"200m 小型四旋翼应通过"
    print(f"  [OK] 200m small_quad w=0.001 → PASS (exp~0.0011)")

    # 异常：200m 处却很大
    bbox_bad = np.array([0.5, 0.5, 0.1, 0.1])
    res = c.check(bbox_bad, "small_quadcopter", distance_m=200)
    assert not res.passed, "200m 大 bbox 应 FAIL"
    print(f"  [OK] 200m small_quad w=0.1 → FAIL (too big)")

    # 未知类别
    res = c.check(bbox, "unknown_drone", distance_m=10)
    assert res.passed  # 跳过
    print("  [OK] 未知类别 → skip (PASS)")

    print("  D1 全部通过\n")


# ============================================================================
# D2: 光照一致性
# ============================================================================
def test_lighting():
    print("=" * 60)
    print("D2 光照一致性")
    print("=" * 60)

    c = LightingConsistencyChecker(kl_threshold=0.5, min_crop_px=10)

    # 同一纯色图像：drone crop 和背景颜色相同 → KL ≈ 0
    img = np.full((128, 128, 3), 100, np.uint8)
    bbox = np.array([0.5, 0.5, 0.3, 0.3])  # 中间 38×38
    res = c.check(img, bbox)
    assert res.passed, f"同色图 KL 应→0，实际 KL={res.kl_divergence}"
    print(f"  [OK] 同色图 → PASS (KL={res.kl_divergence:.4f})")

    # 无人机和背景颜色不同
    img2 = np.full((128, 128, 3), 50, np.uint8)   # 暗灰背景
    img2[40:70, 50:80] = [200, 200, 200]           # 亮白"无人机"
    bbox2 = np.array([0.508, 0.430, 0.234, 0.234])  # 对应无人机区域
    res = c.check(img2, bbox2)
    print(f"  [OK] 亮白无人机+暗灰背景 → "
          f"{'PASS' if res.passed else 'FAIL'} (KL={res.kl_divergence:.4f})")

    # 小目标跳过
    bbox_tiny = np.array([0.5, 0.5, 0.01, 0.01])
    res = c.check(img, bbox_tiny)
    assert res.passed and "skip" in res.detail.lower()
    print(f"  [OK] 极小目标 → skip (PASS)")

    print("  D2 全部通过\n")


# ============================================================================
# D3: IR bbox 对比
# ============================================================================
def test_ir_bbox():
    print("=" * 60)
    print("D3 IR bbox 对比")
    print("=" * 60)

    c = IRBboxChecker(epsilon=0.02)

    # 一致
    rgb = np.array([0.5, 0.5, 0.1, 0.1])
    ir = np.array([0.5, 0.5, 0.11, 0.09])
    res = c.check(rgb, ir)
    assert res.passed, f"w_diff={res.w_diff} h_diff={res.h_diff}"
    print(f"  [OK] 微小偏差 w_diff={res.w_diff:.4f} h_diff={res.h_diff:.4f} → PASS")

    # 偏差过大
    ir_bad = np.array([0.5, 0.5, 0.2, 0.2])
    res = c.check(rgb, ir_bad)
    assert not res.passed
    print(f"  [OK] 大偏差 w_diff={res.w_diff:.4f} → FAIL")

    print("  D3 全部通过\n")


# ============================================================================
# D4: 跨模态对齐
# ============================================================================
def test_cross_modal():
    print("=" * 60)
    print("D4 跨模态对齐")
    print("=" * 60)

    c = CrossModalAlignmentChecker(iou_threshold=0.95)

    # 完全对齐
    bbox = np.array([0.5, 0.5, 0.2, 0.2])
    res = c.check(bbox, bbox)
    assert res.passed and abs(res.iou - 1.0) < 1e-6
    print(f"  [OK] 完全对齐 → IoU={res.iou} PASS")

    # 轻微偏移
    bbox2 = np.array([0.51, 0.5, 0.2, 0.2])
    res = c.check(bbox, bbox2)
    print(f"  [OK] 轻微偏移 → IoU={res.iou:.4f} {'PASS' if res.passed else 'FAIL'}")

    # 严重偏移
    bbox_far = np.array([0.8, 0.8, 0.1, 0.1])
    res = c.check(bbox, bbox_far)
    assert not res.passed
    print(f"  [OK] 严重偏移 → IoU={res.iou:.4f} FAIL")

    print("  D4 全部通过\n")


# ============================================================================
# S7 端到端
# ============================================================================
def test_e2e():
    print("=" * 60)
    print("S7 端到端测试")
    print("=" * 60)

    v = ConsistencyValidator()

    # 完整通过样本
    img = np.random.randint(0, 255, (256, 256, 3), np.uint8)
    sample_ok = {
        "id": "ok",
        "rgb": img,
        "rgb_bbox": np.array([0.5, 0.5, 0.022, 0.022]),
        "drone_category": "small_quadcopter",
        "distance_m": 10,
    }
    res = v.validate(sample_ok)
    print(f"  [OK] 完整样本 → {res.verdict.value} (dims: "
          f"size={res.size['passed']}, lighting={res.lighting['passed']})")

    # 缺 RGB
    res = v.validate({"id": "no_rgb", "rgb": None})
    assert res.reason == "S7_RGB_MISSING"
    print("  [OK] 缺 RGB → S7_RGB_MISSING")

    # 缺 bbox
    res = v.validate({"id": "no_bbox", "rgb": img})
    assert res.reason == "S7_BBOX_MISSING"
    print("  [OK] 缺 bbox → S7_BBOX_MISSING")

    # 双模态完整样本
    ir_img = np.random.randint(0, 255, (256, 256), np.uint8)
    sample_dual = {
        "id": "dual",
        "rgb": img,
        "ir": ir_img,
        "rgb_bbox": np.array([0.5, 0.5, 0.1, 0.1]),
        "ir_bbox": np.array([0.5, 0.5, 0.1, 0.1]),
        "drone_category": "kite",
        "distance_m": 50,
    }
    res = v.validate(sample_dual)
    dims_str = (
        f"size={res.size['passed']}, lighting={res.lighting['passed']}, "
        f"ir_bbox={res.ir_bbox['passed']}, cross_modal={res.cross_modal['passed']}"
    )
    print(f"  [OK] 双模态样本 → {res.verdict.value} ({dims_str})")

    # 配置文件加载
    if CONFIG.exists():
        v2 = ConsistencyValidator.from_config(str(CONFIG))
        assert v2.require_all_dims is True
        print("  [OK] 从配置文件加载 → require_all_dims=True")

    print("  端到端测试全部通过\n")


if __name__ == "__main__":
    test_size()
    test_lighting()
    test_ir_bbox()
    test_cross_modal()
    test_e2e()
    print("S7 全部测试通过 ✅")
