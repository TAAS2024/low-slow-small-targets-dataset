"""
S9 单元测试
============

1. 单元测试：IoU 计算、S9 判定逻辑分支。
2. 集成测试（需 YOLO 模型）：真实 vs 空图检测。

运行：
  python3 test_s3.py                     # 全部
  python3 test_s3.py --skip-integration  # 仅单元测试
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from detection_validator import (              # noqa: E402
    DetectionValidator,
    S9Verdict,
    _iou_xywh,
)

CODE_DIR = Path(__file__).parent.parent / "code"
CONFIG = Path(__file__).parent.parent / "config" / "s9_config.json"


# ============================================================================
# 单元测试
# ============================================================================
def unit_tests():
    print("=" * 60)
    print("S9 单元测试")
    print("=" * 60)

    # --- IoU 计算 ---
    # 完全重叠
    a = np.array([0.5, 0.5, 0.2, 0.2])
    b = np.array([0.5, 0.5, 0.2, 0.2])
    assert abs(_iou_xywh(a, b) - 1.0) < 1e-6, f"IoU 应为 1.0，实际 {_iou_xywh(a, b)}"
    print("  [OK] IoU 完全重叠 = 1.0")

    # 完全不重叠
    c = np.array([0.9, 0.9, 0.1, 0.1])
    assert _iou_xywh(a, c) == 0.0
    print("  [OK] IoU 无重叠 = 0.0")

    # 一半重叠 — [0.5,0.5,0.2,0.2] vs [0.6,0.5,0.2,0.2]
    d = np.array([0.6, 0.5, 0.2, 0.2])
    iou_ad = _iou_xywh(a, d)
    # a: corners (0.4,0.4) to (0.6,0.6)
    # d: corners (0.5,0.4) to (0.7,0.6)
    # inter: (0.5,0.4) to (0.6,0.6) = 0.1 × 0.2 = 0.02
    # union: 0.04 + 0.04 - 0.02 = 0.06
    # iou = 0.02/0.06 = 0.3333
    assert abs(iou_ad - 0.3333) < 0.01, f"IoU 半重叠应为 0.333，实际 {iou_ad}"
    print("  [OK] IoU 半重叠 ≈ 0.333")

    # --- 判定逻辑（不需要 YOLO，用 mock）---
    v = DetectionValidator(iou_thr=0.5)

    # 缺 RGB
    res = v.validate({"id": "t1", "rgb": None, "gt_bboxes": np.array([[0.5, 0.5, 0.1, 0.1]])})
    assert res.reason == "S9_RGB_MISSING"
    assert not res.passed
    print("  [OK] 缺 RGB → S9_RGB_MISSING")

    # 无 GT
    rgb = np.random.randint(0, 255, (128, 128, 3), np.uint8)
    res = v.validate({"id": "t2", "rgb": rgb, "gt_bboxes": np.empty((0, 4))})
    assert res.passed and res.reason == "S9_NO_GT (nothing to detect)"
    print("  [OK] 无 GT → S9_NO_GT (PASS)")

    # GT 形状错误
    res = v.validate({"id": "t3", "rgb": rgb, "gt_bboxes": np.array([1, 2, 3])})
    assert "S9_BAD_GT_SHAPE" in res.reason
    print("  [OK] GT 形状错误 → S9_BAD_GT_SHAPE")

    # --- 内部 _check_line 逻辑：用空 preds 模拟全 miss ---
    # 手动构造一个 DetectionValidator 子类来注入 mock predictions
    class MockValidator(DetectionValidator):
        def __init__(self, mock_preds, **kwargs):
            super().__init__(**kwargs)
            self._mock_preds = mock_preds

        @property
        def model(self):
            return None  # 不会真正加载模型

        def _detect(self, image):
            return np.array(self._mock_preds, dtype=np.float32)

    # 全检测到
    gt = np.array([[0.5, 0.5, 0.2, 0.2]])
    mock_preds = [[0.52, 0.48, 0.22, 0.18]]  # IoU > 0.5
    mv = MockValidator(mock_preds, iou_thr=0.5)
    res = mv.validate({"id": "t_full", "rgb": rgb, "gt_bboxes": gt})
    assert res.passed and res.verdict == S9Verdict.PASS
    assert res.rgb["num_detected"] == 1 and res.rgb["num_missed"] == 0
    print("  [OK] Mock 全部检测到 → PASS")

    # 未检测到
    mock_preds2 = [[0.1, 0.1, 0.05, 0.05]]  # IoU ≈ 0
    mv2 = MockValidator(mock_preds2, iou_thr=0.5)
    res = mv2.validate({"id": "t_miss", "rgb": rgb, "gt_bboxes": gt})
    assert not res.passed and res.verdict == S9Verdict.FAIL
    assert "S9_UNDETECTABLE" in res.reason
    print("  [OK] Mock 未检测到 → FAIL (S9_UNDETECTABLE)")

    # 多 GT：1 检测到 1 未检测到
    gt2 = np.array([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]])
    mock_preds3 = [[0.52, 0.48, 0.22, 0.18]]  # 只匹配第一个 GT
    mv3 = MockValidator(mock_preds3, iou_thr=0.5)
    res = mv3.validate({"id": "t_partial", "rgb": rgb, "gt_bboxes": gt2})
    assert not res.passed
    assert res.rgb["num_detected"] == 1 and res.rgb["num_missed"] == 1
    print("  [OK] Mock 部分检测到 → FAIL（1/2 missed）")

    # IR 线对照记录（不影响判定）
    ir = np.full((128, 128), 100, np.uint8)
    mv4 = MockValidator(mock_preds, iou_thr=0.5)
    res = mv4.validate({"id": "t_ir", "rgb": rgb, "ir": ir, "gt_bboxes": gt})
    assert res.passed  # IR 不判 fail
    # IR 线应包含 note（因为模型名含 yolov8n）
    if res.ir and res.ir.get("note"):
        assert "reference" in res.ir["note"].lower() or "ir" in res.ir["note"].lower()
    print("  [OK] IR 线对照记录 → 不影响 PASS 判定")

    # --- 配置文件加载 ---
    if CONFIG.exists():
        v_cfg = DetectionValidator.from_config(str(CONFIG))
        assert v_cfg.iou_thr == 0.5
        assert v_cfg.conf_thr == 0.25
        print("  [OK] 从配置文件加载 → iou_thr=0.5, conf_thr=0.25")

    print("  单元测试全部通过\n")


# ============================================================================
# 集成测试：真实 YOLO 推理
# ============================================================================
def integration_tests():
    """用真实 yolov8n.pt 对合成图跑一次检测，验证管线不崩溃。"""
    print("=" * 60)
    print("S9 集成测试：真实 YOLO 推理")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [SKIP] ultralytics 未安装")
        return

    # 先检查模型是否已下载
    model_path = "yolov8n.pt"
    if not Path(model_path).exists():
        print(f"  [SKIP] {model_path} 未下载（首次运行会自动下载）")
        return

    v = DetectionValidator(model_path=model_path, iou_thr=0.5, conf_thr=0.25)

    # 合成一张简单的测试图：灰色背景 + 一个小方块
    img = np.full((320, 320, 3), 128, np.uint8)
    img[100:140, 120:160] = [255, 255, 255]  # 40×40 白色方块
    gt = np.array([[0.4375, 0.375, 0.125, 0.125]])  # 对应 40×40 方块

    res = v.validate({
        "id": "synth_cube",
        "rgb": img,
        "gt_bboxes": gt,
    })

    print(f"  合成方块图: {res}")
    print(f"  RGB 线: detected={res.rgb['detected']}, "
          f"GT数={res.rgb['total_gt']}, "
          f"预测框数={res.rgb['pred_count']}, "
          f"IoU={res.rgb['iou_scores']}")

    # 全黑图（不应检测到任何东西）
    black = np.zeros((320, 320, 3), np.uint8)
    res2 = v.validate({
        "id": "black",
        "rgb": black,
        "gt_bboxes": np.array([[0.5, 0.5, 0.3, 0.3]]),
    })
    print(f"  全黑图: {res2}")

    # 报告落盘
    report = {
        "model": model_path,
        "iou_thr": v.iou_thr,
        "tests": [
            {"name": "synth_cube", "verdict": res.verdict.value,
             "detected": res.rgb["detected"], "iou": res.rgb["iou_scores"]},
            {"name": "black", "verdict": res2.verdict.value,
             "detected": res2.rgb["detected"], "iou": res2.rgb["iou_scores"]},
        ],
    }
    out = Path(__file__).parent.parent / "reports" / "s3_test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告写入 → {out.resolve()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-integration", action="store_true")
    args = ap.parse_args()

    unit_tests()
    if not args.skip_integration:
        integration_tests()
