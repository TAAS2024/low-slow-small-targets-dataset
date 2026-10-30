"""
═══════════════════════════════════════════════════════════════
  S2 — Agent 2 (Transformer) 输出校验 Demo
═══════════════════════════════════════════════════════════════
演示 6 项逐帧 bbox + 条件图检查。
运行: python3 demos/demo_s2.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s2_transformer_validator import S2TransformerValidator

F_H, F_W = 256, 256
v = S2TransformerValidator(
    config_dir=os.path.join(os.path.dirname(__file__), "..", "config"),
    frame_size=(F_H, F_W),
)


def demo(name, frames, traj_len, drone="quadcopter"):
    print(f"\n{'='*60}")
    print(f"  📋 {name}")
    print(f"{'='*60}")
    r = v.validate(frames, traj_len, drone)
    status = "✅ PASS" if r["pass"] else f"❌ FAIL [{r['failure_code']}]"
    print(f"  结果: {status}")
    print(f"  原因: {r['reason']}")
    if "details" in r:
        for k, val in r["details"].items():
            print(f"    {k}: {val}")
    return r


# ── 辅助：正常帧 ───────────────────────────────────────
def ok_frame():
    return {
        "bbox": (50, 50, 40, 40),
        "seg_map": np.ones((F_H, F_W), dtype=np.uint8) * 2,  # 有 drone 类
        "depth_map": np.random.rand(F_H, F_W).astype(np.float32),
    }


seg_no_drone = np.zeros((F_H, F_W), dtype=np.uint8)


print("""
╔══════════════════════════════════════════════════════════════╗
║        S2 — Agent 2 (Transformer) 输出校验 Demo            ║
║       6 项检查 × (PASS + FAIL) = 12 场景                   ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── 检 1: bbox 越界 ────────────────────────────────────
frames_ok = [ok_frame() for _ in range(8)]
demo("检1-1 ✅ 8 帧 bbox=(50,50,40,40) → 全部在 (256,256) 内", frames_ok, 8)

bad1 = [ok_frame() for _ in range(8)]
bad1[3]["bbox"] = (-10, 50, 40, 40)
demo("检1-2 ❌ bbox — 第3帧 x=-10 (左越界)", bad1, 8)

bad2 = [ok_frame() for _ in range(8)]
bad2[5]["bbox"] = (50, 200, 40, 100)
demo("检1-3 ❌ bbox — 第5帧 y+h=200+100=300 > 256 (下越界)", bad2, 8)

# ── 检 2: bbox 尺寸过小/过大 ───────────────────────────
demo("检2-1 ✅ bbox=40×40 → 合理尺寸", frames_ok, 8)

bad3 = [ok_frame() for _ in range(8)]
bad3[2]["bbox"] = (50, 50, 2, 2)
demo("检2-2 ❌ bbox — 2×2 过小 (低于 min_size)", bad3, 8)

bad4 = [ok_frame() for _ in range(8)]
bad4[4]["bbox"] = (0, 0, 256, 256)
demo("检2-3 ❌ bbox — 256×256 过大 (超出 max_ratio)", bad4, 8)

# ── 检 3: bbox 位置异常 ────────────────────────────────
demo("检3-1 ✅ bbox 中心 (70,70) → 不在图像边缘", frames_ok, 8)

bad5 = [ok_frame() for _ in range(8)]
bad5[1]["bbox"] = (0, 100, 40, 40)
demo("检3-2 ❌ 位置 — bbox 紧贴左边缘 x=0", bad5, 8)

# ── 检 4: seg_map 无人机类缺失 ─────────────────────────
demo("检4-1 ✅ seg_map 含无人机类 (2)", frames_ok, 8)

bad6 = [ok_frame() for _ in range(8)]
bad6[6]["seg_map"] = seg_no_drone
demo("检4-2 ❌ seg — seg_map 中无无人机类 (全0)", bad6, 8)

# ── 检 5: depth_map 值域 ───────────────────────────────
demo("检5-1 ✅ depth_map ∈ [0,1] float", frames_ok, 8)

bad7 = [ok_frame() for _ in range(8)]
bad7[3]["depth_map"] = np.full((F_H, F_W), -1.0, dtype=np.float32)
demo("检5-2 ❌ depth — 全 -1 (无效深度值)", bad7, 8)

# ── 检 6: 帧数一致性 ───────────────────────────────────
demo("检6-1 ✅ 8 帧 ↔ trajectory_len=8", frames_ok, 8)

demo("检6-2 ❌ 帧数 — 8 帧但 trajectory_len=3 (不匹配)", frames_ok, 3)

# ── 总结 ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("  🏁 S2 Demo 完成。共 12 个场景：6 PASS + 6 FAIL。")
print(f"{'='*60}\n")
