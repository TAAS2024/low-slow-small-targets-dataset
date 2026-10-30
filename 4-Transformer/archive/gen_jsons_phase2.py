#!/usr/bin/env python3
"""Generate Agent 1 JSON templates for Phase 2 mixed pairs (sky + ground)."""
import json, os

BASE = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer"
MIXED = f"{BASE}/output/phase2_mixed/pairs.jsonl"
OUT = f"{BASE}/output/phase2_mixed/json"
os.makedirs(OUT, exist_ok=True)

with open(MIXED) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

SKY_TEMPLATE = {
    "drone_type": "quadrotor",
    "trajectory": [{"t": 0.0, "action": "hover", "distance": 50.0, "norm_u": 0.5, "norm_v": 0.5}],
    "time_of_day": "afternoon",
    "weather": "clear",
    "scene_type": "rural",
    "scene_description": "Open sky background, drone hovering at distance.",
    "modality": "RGB",
    "camera": {"position": "bottom", "elevation_deg": 45.0, "fov_deg": 60.0},
}

GROUND_TEMPLATE = {
    "drone_type": "quadrotor",
    "trajectory": [{"t": 0.0, "action": "hover", "distance": 30.0, "norm_u": 0.5, "norm_v": 0.3}],
    "time_of_day": "afternoon",
    "weather": "clear",
    "scene_type": "mountain",
    "scene_description": "Aerial view of terrain below, drone looking down.",
    "modality": "RGB",
    "camera": {"position": "top", "elevation_deg": 0.0, "fov_deg": 60.0},
}

count = 0
for p in pairs:
    scene = p["scene_type"]
    tmpl = SKY_TEMPLATE if scene == "sky" else GROUND_TEMPLATE
    name = p["frame_name"]
    json_path = f"{OUT}/{name}.json"
    with open(json_path, "w") as f:
        json.dump(tmpl, f, indent=2)
    p["json_path"] = json_path
    count += 1

# Save updated pairs
NEW_PAIRS = f"{BASE}/output/phase2_mixed/pairs_with_json.jsonl"
with open(NEW_PAIRS, "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Generated {count} JSONs → {OUT}/")
print(f"Updated pairs → {NEW_PAIRS}")
