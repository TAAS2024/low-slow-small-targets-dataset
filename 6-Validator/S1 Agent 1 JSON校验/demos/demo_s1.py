"""
═══════════════════════════════════════════════════════════════
  S1 — Agent 1 (LLM JSON) 输出校验 Demo
═══════════════════════════════════════════════════════════════
演示 6 项检查的 PASS / FAIL 场景（短路求值）。
运行: python3 demos/demo_s1.py
"""
import os, sys, json, copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s1_json_validator import S1JSONValidator

v = S1JSONValidator(config_dir=os.path.join(os.path.dirname(__file__), "..", "config"))

# ── 合法 JSON（扁平结构） ──────────────────────────────
VALID = {
    "drone_type": "quadcopter",
    "trajectory": [
        {"distance": 0.0,   "action": "fly_straight",  "t": 0.0},
        {"distance": 15.0,  "action": "fly_straight",  "t": 1.0},
        {"distance": 30.0,  "action": "fly_curve",     "t": 2.0},
    ],
    "time_of_day": "morning",
    "weather": "clear",
    "scene_type": "outdoor_urban",
    "scene_description": "A quadcopter flies over an urban area in morning light",
    "modality": "RGB",
    "camera": {
        "position": "eye_level",
        "fov_deg": 90,
        "elevation_deg": 0,
    },
    "confidence_note": "high",
}


def demo(name, data):
    print(f"\n{'='*60}")
    print(f"  📋 {name}")
    print(f"{'='*60}")
    r = v.validate(data)
    status = "✅ PASS" if r["pass"] else f"❌ FAIL [{r['failure_code']}]"
    print(f"  结果: {status}")
    print(f"  原因: {r['reason']}")
    return r


print("""
╔══════════════════════════════════════════════════════════════╗
║           S1 — Agent 1 (LLM JSON 输出) 校验 Demo           ║
║        6 项检查 × (PASS + FAIL) = 12 场景                   ║
║        ⚡ 短路求值：第一项失败即返回                          ║
╚══════════════════════════════════════════════════════════════╝
""")

# ═══ 检 1: Schema 完整性 ═══════════════════════════════
demo("检1-1 ✅ 完整 9 字段 + camera 3 子字段", VALID)

bad = copy.deepcopy(VALID)
del bad["weather"]
demo("检1-2 ❌ Schema — 缺少 weather 字段", bad)

bad = copy.deepcopy(VALID)
bad["trajectory"] = []
demo("检1-3 ❌ Schema — trajectory 为空列表", bad)

# ═══ 检 2: 枚举合法性 ══════════════════════════════════
demo("检2-1 ✅ drone_type=quadcopter, weather=clear, time_of_day=morning", VALID)

bad = copy.deepcopy(VALID)
bad["weather"] = "tornado"
demo("检2-2 ❌ 枚举 — weather='tornado' (非法值)", bad)

bad = copy.deepcopy(VALID)
bad["drone_type"] = "spaceship"
demo("检2-3 ❌ 枚举 — drone_type='spaceship' (非法值)", bad)

# ═══ 检 3: 逻辑互斥 ════════════════════════════════════
demo("检3-1 ✅ morning + clear + outdoor_urban → 无互斥", VALID)

bad = copy.deepcopy(VALID)
bad["time_of_day"] = "night"
bad["weather"] = "backlight"
demo("检3-2 ❌ 逻辑 — night + backlight 互斥（逆光需要太阳，夜晚没有）", bad)

bad = copy.deepcopy(VALID)
bad["scene_type"] = "indoor"
bad["weather"] = "rainy"
demo("检3-3 ❌ 逻辑 — indoor + rainy 互斥（室内不下雨）", bad)

# ═══ 检 4: 描述关键词交叉检查 ══════════════════════════
demo("检4-1 ✅ 描述含 'quadcopter' + 'urban' + 'morning' → 与字段一致", VALID)

bad = copy.deepcopy(VALID)
bad["scene_description"] = "rainy day in the city"
# 含 "rainy" → 但 weather="clear"
demo("检4-2 ❌ 描述 — 说 'rainy' 但 weather='clear'", bad)

bad = copy.deepcopy(VALID)
bad["scene_description"] = "drone flying at night under moonlight"
bad["time_of_day"] = "morning"
demo("检4-3 ❌ 描述 — 说 'night' 但 time_of_day='morning'", bad)

# ═══ 检 5: trajectory 值域 ═════════════════════════════
demo("检5-1 ✅ 3 个轨迹点，t 严格递增，action 合法，distance ≥ 0", VALID)

bad = copy.deepcopy(VALID)
bad["trajectory"] = [
    {"distance": -5.0, "action": "fly_straight", "t": 0.0},
]
demo("检5-2 ❌ 轨迹 — distance=-5 < 0", bad)

bad = copy.deepcopy(VALID)
bad["trajectory"] = [
    {"distance": 10.0, "action": "fly_straight", "t": 2.0},
    {"distance": 20.0, "action": "fly_straight", "t": 1.0},
]
demo("检5-3 ❌ 轨迹 — t=1 < 前一帧 t=2（非严格递增）", bad)

bad = copy.deepcopy(VALID)
bad["trajectory"] = [
    {"distance": 10.0, "action": "teleport", "t": 0.0},
]
demo("检5-4 ❌ 轨迹 — action='teleport' 不在合法枚举中", bad)

# ═══ 检 6: camera 参数 ═════════════════════════════════
demo("检6-1 ✅ position=eye_level, fov=90, elevation=0 (在 [-10,30] 内)", VALID)

bad = copy.deepcopy(VALID)
bad["camera"]["fov_deg"] = 200
demo("检6-2 ❌ 相机 — fov=200 超出 [20, 120]", bad)

bad = copy.deepcopy(VALID)
bad["camera"]["position"] = "top_down"
bad["camera"]["elevation_deg"] = 10
demo("检6-3 ❌ 相机 — position='top_down' 但 elevation=10 不在 [60,90]", bad)

# ── 总结 ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("  🏁 S1 Demo 完成。共 14 个场景：6 PASS + 8 FAIL。")
print(f"{'='*60}\n")
