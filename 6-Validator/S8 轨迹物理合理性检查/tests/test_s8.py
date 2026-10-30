"""
S8 单元测试
============

1. 规则单元测试：四条规则正确检测异常 + 放行正常过渡。
2. 端到端测试：多帧序列综合判定。
3. 真实 GT JSON 测试（如有 Agent 2 产出）+ 人工异常注入。

运行：
  python3 test_s2.py                     # 全部
  python3 test_s2.py --gt <gt_json>      # 含真实 GT 验证
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from trajectory_validator import (           # noqa: E402
    TrajectoryValidator,
    S8Verdict,
    AnomalyType,
    check_position,
    check_size,
    check_acceleration,
    check_direction,
)

CODE_DIR = Path(__file__).parent.parent / "code"
CONFIG = Path(__file__).parent.parent / "config" / "s8_config.json"


# ============================================================================
# 规则单元测试
# ============================================================================
def rule_tests():
    print("=" * 60)
    print("S8 规则单元测试")
    print("=" * 60)

    # --- 规则1：位置连续性 ---
    # 正常过渡（位移很小）
    bbox_a = np.array([0.5, 0.5, 0.1, 0.1])
    bbox_b = np.array([0.52, 0.51, 0.1, 0.1])
    assert check_position(bbox_a, bbox_b, v_max=0.3) is None
    print("  [OK] R1 正常位移 → None")

    # 异常跳帧（位移过大）
    bbox_c = np.array([0.9, 0.9, 0.1, 0.1])
    a = check_position(bbox_a, bbox_c, v_max=0.3)
    assert a is not None and a.anomaly_type == AnomalyType.POSITION_JUMP
    print(f"  [OK] R1 跳帧 → POSITION_JUMP (penalty={a.penalty:.3f})")

    # --- 规则2：尺寸连续性 ---
    bbox_same = np.array([0.5, 0.5, 0.1, 0.1])
    assert check_size(bbox_same, bbox_same, 0.3) is None
    print("  [OK] R2 尺寸不变 → None")

    bbox_big = np.array([0.5, 0.5, 0.5, 0.5])
    a = check_size(bbox_same, bbox_big, 0.3)
    assert a is not None and a.anomaly_type == AnomalyType.SIZE_MUTATION
    print(f"  [OK] R2 尺寸突变 (0.1→0.5) → SIZE_MUTATION (penalty={a.penalty:.3f})")

    # --- 规则3：加速度 ---
    assert check_acceleration(0.1, 0.12, a_max=0.15) is None
    print("  [OK] R3 缓加速 → None")

    a = check_acceleration(0.1, 0.5, a_max=0.15)
    assert a is not None and a.anomaly_type == AnomalyType.ACCELERATION
    print(f"  [OK] R3 急加速 (0.1→0.5) → ACCELERATION (penalty={a.penalty:.3f})")

    # --- 规则4：方向平滑 ---
    assert check_direction(30, 50, theta_max=45) is None
    print("  [OK] R4 30°→50° (Δ=20°) → None")

    a = check_direction(10, 200, theta_max=45)
    assert a is not None and a.anomaly_type == AnomalyType.DIRECTION_TURN
    # 10° to 200°: diff = 190°, but min angle = 170°
    assert a.penalty > 0
    print(f"  [OK] R4 10°→200° → DIRECTION_TURN (penalty={a.penalty:.3f})")

    # 角度环绕测试：350° → 10° = 20°
    assert check_direction(350, 10, theta_max=45) is None
    print("  [OK] R4 350°→10° (Δ=20° with wrap) → None")

    print("  规则单元测试全部通过\n")


# ============================================================================
# 端到端测试
# ============================================================================
def e2e_tests():
    print("=" * 60)
    print("S8 端到端测试")
    print("=" * 60)

    v = TrajectoryValidator(v_max=0.3, a_max=0.15, theta_max=45,
                            size_mutation_thr=0.3, pass_threshold=0.8)

    # --- 正常序列：缓速直线运动 ---
    normal_frames = []
    for i in range(5):
        cx = 0.3 + i * 0.05   # 每帧右移 5%
        normal_frames.append({
            "frame_id": i,
            "bbox": [cx, 0.5, 0.1, 0.1],
            "speed": 0.05,
            "direction": 0.0,
        })
    res = v.validate(normal_frames)
    assert res.passed and res.verdict == S8Verdict.PASS
    assert res.score == 1.0
    assert len(res.anomalies) == 0
    print(f"  [OK] 正常5帧直线 → PASS (score={res.score})")

    # --- 跳帧序列：第3帧突然跳到远端 ---
    jump_frames = normal_frames.copy()
    jump_frames[3]["bbox"] = [0.9, 0.9, 0.1, 0.1]  # 跳帧
    jump_frames[4]["bbox"] = [0.95, 0.9, 0.1, 0.1]
    res = v.validate(jump_frames)
    assert not res.passed
    assert any(a["anomaly_type"] == "POSITION_JUMP" for a in res.anomalies)
    print(f"  [OK] 跳帧序列 → FAIL score={res.score:.3f} "
          f"anomalies={len(res.anomalies)}")

    # --- 尺寸突变序列 ---
    size_frames = normal_frames.copy()
    size_frames[2]["bbox"] = [0.45, 0.5, 0.5, 0.5]  # 突然变大
    res = v.validate(size_frames)
    has_size = any(a["anomaly_type"] == "SIZE_MUTATION" for a in res.anomalies)
    print(f"  [OK] 尺寸突变序列 → {'FAIL' if not res.passed else 'PASS'} "
          f"score={res.score:.3f} size_anomaly={has_size}")

    # --- 急转弯序列 ---
    turn_frames = normal_frames.copy()
    turn_frames[2]["direction"] = 180.0  # 突然调头
    res = v.validate(turn_frames)
    has_turn = any(a["anomaly_type"] == "DIRECTION_TURN" for a in res.anomalies)
    print(f"  [OK] 急转弯序列 → {'FAIL' if not res.passed else 'PASS'} "
          f"score={res.score:.3f} turn_anomaly={has_turn}")

    # --- 混合异常：跳帧 + 加速 ---
    mixed_frames = normal_frames.copy()
    mixed_frames[2]["bbox"] = [0.8, 0.2, 0.1, 0.1]   # 跳帧
    mixed_frames[2]["speed"] = 0.5                      # 急加速
    res = v.validate(mixed_frames)
    assert not res.passed
    # 应该至少有位置和加速度两类异常
    types = set(a["anomaly_type"] for a in res.anomalies)
    print(f"  [OK] 混合异常 → FAIL score={res.score:.3f} "
          f"anomaly_types={types}")

    # --- 单帧序列 ---
    res = v.validate([normal_frames[0]])
    assert res.passed and res.reason.startswith("S8_TOO_FEW_FRAMES")
    print("  [OK] 单帧序列 → PASS (S8_TOO_FEW_FRAMES)")

    # --- 配置文件加载 ---
    if CONFIG.exists():
        v_cfg = TrajectoryValidator.from_config(str(CONFIG))
        assert v_cfg.v_max == 0.3
        assert v_cfg.theta_max == 45.0
        print("  [OK] 从配置文件加载 → v_max=0.3, theta_max=45")

    print("  端到端测试全部通过\n")


# ============================================================================
# 真实 GT JSON 测试
# ============================================================================
def real_gt_test(gt_json_path: str):
    """读取 Agent 2 产出的真实 GT JSON 序列，验证正常通过；再注入异常验证检测。"""
    print("=" * 60)
    print("S8 真实 GT JSON 测试")
    print("=" * 60)

    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data if isinstance(data, list) else data.get("frames", [])
    if len(frames) < 2:
        print(f"  [SKIP] GT JSON 只有 {len(frames)} 帧，需要 >= 2")
        return

    v = TrajectoryValidator.from_config(str(CONFIG)) if CONFIG.exists() \
        else TrajectoryValidator()

    # 原始 GT 应通过
    res = v.validate(frames)
    print(f"  原始 GT: {len(frames)} 帧 → {res.verdict.value} "
          f"score={res.score:.3f} anomalies={len(res.anomalies)}")

    if not res.passed:
        print("  ⚠️ 原始 GT 未通过！异常详情：")
        for a in res.anomalies:
            print(f"    {a['anomaly_type']}: {a['detail']} "
                  f"({a['from_frame']}→{a['to_frame']})")

    # 注入异常
    if len(frames) >= 3:
        injected = [dict(f) for f in frames]
        mid = len(injected) // 2
        injected[mid]["bbox"] = [0.95, 0.95, 0.05, 0.05]  # 跳帧
        res_inj = v.validate(injected)
        print(f"  注入跳帧: {res_inj.verdict.value} score={res_inj.score:.3f} "
              f"anomalies={len(res_inj.anomalies)}")
        assert not res_inj.passed, "注入异常应判 FAIL！"

    print("  真实 GT 测试通过\n")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", help="Agent 2 产出的 GT JSON 文件路径", default=None)
    args = ap.parse_args()

    rule_tests()
    e2e_tests()

    if args.gt:
        real_gt_test(args.gt)
    else:
        # 尝试自动查找
        candidates = [
            Path("/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/"
                 "4-Transformer/output/gt"),
        ]
        for cand in candidates:
            jsons = sorted(cand.glob("*.json")) if cand.exists() else []
            if jsons:
                print(f"[auto] 发现 GT JSON: {jsons[0].name}")
                real_gt_test(str(jsons[0]))
                break
        else:
            print("  [SKIP] 未找到 GT JSON，用 --gt 指定路径")
