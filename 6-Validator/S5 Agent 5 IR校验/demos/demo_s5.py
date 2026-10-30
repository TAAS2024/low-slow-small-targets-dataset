"""
═══════════════════════════════════════════════════════════════
  S5 — Agent 5 (IR 转换) 输出校验 Demo
═══════════════════════════════════════════════════════════════
演示 3 项防御性检查 — 只告警不截停。
运行: python3 demos/demo_s5.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s5_ir_validator import S5IRValidator

v = S5IRValidator(config_dir=os.path.join(os.path.dirname(__file__), "..", "config"))

H, W = 256, 256


def demo(name, ir, rgb=None):
    print(f"\n{'='*60}")
    print(f"  📋 {name}")
    print(f"{'='*60}")
    r = v.validate(ir, rgb)
    status = "✅ PASS" if r["pass"] else "❌ FAIL"
    n_warnings = len(r["warnings"])
    banner = "⚠️  PASS (防御性放行)" if n_warnings else "✅ PASS"
    print(f"  结果: {banner}")
    print(f"  原因: {r['reason']}")
    if r["warnings"]:
        print(f"  ⚠️  警告 ({n_warnings} 条):")
        for w in r["warnings"]:
            print(f"    [{w['code']}] {w['reason']}")
    return r


def ok_ir():
    """正常双峰 IR"""
    half = H * W // 2
    a = np.random.normal(60, 15, half)
    b = np.random.normal(180, 15, half)
    ir = np.concatenate([a, b]).clip(0, 255).astype(np.uint8).reshape(H, W)
    return ir

def ok_rgb():
    return np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)


print("""
╔══════════════════════════════════════════════════════════════╗
║         S5 — Agent 5 (IR 转换) 输出校验 Demo               ║
║    3 项检查 × (无告警 + 有告警) = 6 场景                   ║
║    🔑 核心原则：只告警不截停，pass 始终为 True               ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── 检 1: 尺寸一致 ─────────────────────────────────────
demo("检1-1 ✅ IR=256×256 ↔ RGB=256×256", ok_ir(), ok_rgb())

ir_small = ok_ir()[:128, :128]
demo("检1-2 ⚠️  尺寸 — IR=128×128 ≠ RGB=256×256", ir_small, ok_rgb())

# ── 检 2: 灰度确认 ─────────────────────────────────────
demo("检2-1 ✅ IR 单通道 (H,W)", ok_ir())

# 伪彩色三通道
pseudo = np.zeros((128, 128, 3), dtype=np.uint8)
pseudo[:, :, 0] = 50
pseudo[:, :, 1] = 100
pseudo[:, :, 2] = 200
demo("检2-2 ⚠️  非灰度 — 三通道均值各不相同 (伪彩色)", pseudo)

# 三通道但完全一致
gray_3ch = np.full((128, 128, 3), 128, dtype=np.uint8)
demo("检2-3 ✅ 三通道但一致 (R=G=B=128) → 实际是灰度", gray_3ch)

# ── 检 3: 直方图合理性 ─────────────────────────────────
demo("检3-1 ✅ 双峰 IR (天空~60, 地面~180)", ok_ir())

flat_ir = np.full((128, 128), 100, dtype=np.uint8)
demo("检3-2 ⚠️  直方图 — 单峰全灰 (全图=100)", flat_ir)

# ── 核心演示：即使全部告警也放行 ───────────────────────
print(f"\n{'='*60}")
print("  🔑 关键演示：三项全告警 → pass 仍为 True")
print(f"{'='*60}")
# 三通道伪彩色 + 不同尺寸 + 单峰
bad_ir = np.full((128, 128, 3), 100, dtype=np.uint8)
bad_ir[:, :, 0] += 20
bad_ir[:, :, 2] -= 20
r = v.validate(bad_ir, ok_rgb())
print(f"  结果: ⚠️  PASS (防御性放行)")
print(f"  pass = {r['pass']}  ← 始终 True")
print(f"  failure_code = {r['failure_code']}  ← 始终 None")
print(f"  共 {len(r['warnings'])} 条警告——但数据流不被截停")

# ── 总结 ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("  🏁 S5 Demo 完成。共 7 个场景。")
print("  📌 核心记忆：IR 线只告警不截停，pass 始终为 True。")
print(f"{'='*60}\n")
