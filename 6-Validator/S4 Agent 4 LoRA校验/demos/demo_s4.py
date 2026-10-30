"""
═══════════════════════════════════════════════════════════════
  S4 — Agent 4 (LoRA) 输出校验 Demo
═══════════════════════════════════════════════════════════════
演示 3 项 LoRA 特有生成 artifact 检查。
运行: python3 demos/demo_s4.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s4_lora_validator import S4LoRAValidator

v = S4LoRAValidator(config_dir=os.path.join(os.path.dirname(__file__), "..", "config"))


def demo(name, frames, bboxes=None):
    print(f"\n{'='*60}")
    print(f"  📋 {name}")
    print(f"{'='*60}")
    r = v.validate(frames, bboxes)
    status = "✅ PASS" if r["pass"] else f"❌ FAIL [{r['failure_code']}]"
    print(f"  结果: {status}")
    print(f"  原因: {r['reason']}")
    if "details" in r:
        for k, val in r["details"].items():
            if isinstance(val, (list, np.ndarray)):
                print(f"    {k}: {val}")
            else:
                print(f"    {k}: {val:.4f}" if isinstance(val, float) else f"    {k}: {val}")
    return r


# ── 辅助 ──────────────────────────────────────────────
H, W = 256, 256

def distinct_frame():
    """背景和无人机区域明显不同的帧"""
    bg = np.random.rand(H, W, 3).astype(np.float32) * 0.1 + 0.4     # ~0.45
    drone = np.random.rand(30, 30, 3).astype(np.float32) * 0.05 + 0.15  # ~0.17
    f = bg.copy()
    f[50:80, 50:80] = drone
    return np.clip(f, 0, 1).astype(np.float32)

def uniform_frame():
    """全图均匀 → 无人机和背景一样 → 概念泄漏"""
    return np.random.rand(H, W, 3).astype(np.float32) * 0.05 + 0.5

def too_dark_frame():
    return np.random.rand(H, W, 3).astype(np.float32) * 0.1

def too_bright_frame():
    return np.random.rand(H, W, 3).astype(np.float32) * 0.3 + 0.7

def red_cast_frame():
    f = np.random.rand(H, W, 3).astype(np.float32) * 0.1 + 0.2
    f[:, :, 0] += 0.4  # 红色通道偏高
    return np.clip(f, 0, 1).astype(np.float32)


print("""
╔══════════════════════════════════════════════════════════════╗
║          S4 — Agent 4 (LoRA) 输出校验 Demo                 ║
║      3 项检查 × (PASS + FAIL) = 6 场景                      ║
╚══════════════════════════════════════════════════════════════╝
""")

BBOX = [(50, 50, 30, 30)]

# ── 检 1: 概念泄漏 ─────────────────────────────────────
frames_ok = [distinct_frame() for _ in range(3)]
demo("检1-1 ✅ 背景(~0.45)与无人机(~0.17)区分明显 → SSIM 低", frames_ok, BBOX * 3)

frames_bleed = [uniform_frame() for _ in range(3)]
demo("检1-2 ❌ 概念泄漏 — 全图均匀 (无人机纹理渗入背景)", frames_bleed, BBOX * 3)

# ── 检 2: 纹理重复 ─────────────────────────────────────
frames_diff = [distinct_frame() for _ in range(5)]
demo("检2-1 ✅ 5 帧各不相同 → 纹理多样", frames_diff, BBOX * 5)

same = distinct_frame()
demo("检2-2 ❌ 纹理重复 — 5 帧完全相同 (过拟合)", [same.copy() for _ in range(5)], BBOX * 5)

# ── 检 3: 全局色偏 ─────────────────────────────────────
demo("检3-1 ✅ 正常色彩 (~0.45)", [distinct_frame()])

demo("检3-2 ❌ 色偏 — 全图偏暗 (均值 ~0.05)", [too_dark_frame()])

demo("检3-3 ❌ 色偏 — 红色通道偏强", [red_cast_frame()])

# ── 总结 ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("  🏁 S4 Demo 完成。共 7 个场景：3 PASS + 4 FAIL。")
print(f"{'='*60}\n")
