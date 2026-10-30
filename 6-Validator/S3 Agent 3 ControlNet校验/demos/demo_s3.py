"""
═══════════════════════════════════════════════════════════════
  S3 — Agent 3 (ControlNet) 输出校验 Demo
═══════════════════════════════════════════════════════════════
演示 4 项 seg/depth/pose 条件图检查。
运行: python3 demos/demo_s3.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s3_controlnet_validator import S3ControlNetValidator

H, W = 256, 256
v = S3ControlNetValidator(config_dir=os.path.join(os.path.dirname(__file__), "..", "config"))


def demo(name, seg, depth, pose, bbox_center=None, scale=None):
    print(f"\n{'='*60}")
    print(f"  📋 {name}")
    print(f"{'='*60}")
    r = v.validate(seg, depth, pose, bbox_center, scale)
    status = "✅ PASS" if r["pass"] else f"❌ FAIL [{r['failure_code']}]"
    print(f"  结果: {status}")
    print(f"  原因: {r['reason']}")
    if "details" in r:
        for k, val in r["details"].items():
            print(f"    {k}: {val}")
    return r


# ── 辅助 ──────────────────────────────────────────────
def ok_seg():
    """正常的无人机分割图：中间一个 80×80 的方形 mask"""
    s = np.zeros((H, W), dtype=np.uint8)
    s[88:168, 88:168] = 1
    return s

def ok_depth():
    d = np.random.rand(H, W).astype(np.float32) * 0.8 + 0.1
    d[88:168, 88:168] = np.random.rand(80, 80).astype(np.float32) * 0.3 + 0.05
    return d

def ok_pose():
    return np.random.rand(H, W).astype(np.float32) * 0.5 + 0.25

# 棋盘格 seg（锯齿状边界）— 密集小块
def checker_seg():
    s = np.zeros((H, W), dtype=np.uint8)
    for i in range(64, 192, 4):
        for j in range(64, 192, 4):
            if (i + j) % 8 == 0:
                s[i:i+4, j:j+4] = 1
    return s


# 平坦 depth（无人机与背景深度相同）
def flat_depth():
    return np.full((H, W), 0.5, dtype=np.float32)


print("""
╔══════════════════════════════════════════════════════════════╗
║        S3 — Agent 3 (ControlNet) 输出校验 Demo             ║
║         4 项检查 × (PASS + FAIL) = 8 场景                   ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── 检 1: 三种条件图尺寸一致 ───────────────────────────
demo("检1-1 ✅ seg/depth/pose 均为 256×256", ok_seg(), ok_depth(), ok_pose())

depth_bad = np.random.rand(128, 128).astype(np.float32)
demo("检1-2 ❌ 尺寸 — depth=128×128 ≠ seg=256×256", ok_seg(), depth_bad, ok_pose())

# ── 检 2: seg 无人机区域位置与 bbox 对齐 ───────────────
demo("检2-1 ✅ seg 中心 (128,128) ≈ bbox 中心 (125,130)", ok_seg(), ok_depth(), ok_pose(), bbox_center=(125, 130))

demo("检2-2 ❌ 对齐 — seg 中心 (128,128) vs bbox 中心 (200,200) 偏差大",
     ok_seg(), ok_depth(), ok_pose(), bbox_center=(200, 200))

# ── 检 3: seg 边界平滑性 ───────────────────────────────
demo("检3-1 ✅ 平滑方形 mask → 膨胀增长率 < 50%", ok_seg(), ok_depth(), ok_pose())

demo("检3-2 ❌ 锯齿 — 棋盘格 seg → 膨胀增长率 > 50% (内部空隙被填满)",
     checker_seg(), ok_depth(), ok_pose())

# ── 检 4: depth 一致性 ─────────────────────────────────
demo("检4-1 ✅ depth 中心区域 (无人机) 值 < 边缘 (背景)", ok_seg(), ok_depth(), ok_pose(), scale=1.0)

demo("检4-2 ❌ depth — 全图平坦 0.5 (无人机与背景深度无差异)",
     ok_seg(), flat_depth(), ok_pose(), scale=1.0)

# ── 总结 ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("  🏁 S3 Demo 完成。共 8 个场景：4 PASS + 4 FAIL。")
print(f"{'='*60}\n")
