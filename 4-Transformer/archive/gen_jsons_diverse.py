#!/usr/bin/env python3
"""Generate diverse Agent 1 JSON templates for Phase 2 mixed pairs.
Uses filename hash for deterministic but diverse attribute assignment.
Covers all 9 fields of the Agent 1 Schema with realistic enum values."""
import json, os, hashlib

BASE = "/mnt/d/learning/ObsidianVault/Paper-低慢小数据集生成架构/4-Transformer"
PAIRS = f"{BASE}/output/phase2_mixed/pairs_with_json.jsonl"
JSON_OUT = f"{BASE}/output/phase2_mixed/json_diverse"
os.makedirs(JSON_OUT, exist_ok=True)

with open(PAIRS) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

def hash_int(name, start, end):
    """Deterministic int in [start, end) from filename hash."""
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    return start + h % (end - start)

def pick(name, lst):
    return lst[hash_int(name, 0, len(lst))]

DRONES = ["DJI Air 2S", "DJI Mavic 2 Pro", "DJI Mavic 3", "DJI Mini 2"]
ACTIONS_SKY = ["hover", "hover", "approach", "lateral_move", "circle", "ascend"]
ACTIONS_GROUND = ["hover", "hover", "hover", "lateral_move"]
SCENES_SKY = ["rural", "mountain", "coastal", "forest"]
SCENES_GROUND = ["mountain", "rural", "forest", "farmland"]
TOD = ["morning", "afternoon", "afternoon", "dusk"]
WEATHER = ["clear", "clear", "clear", "overcast"]

for p in pairs:
    name = p["frame_name"]
    scene = p["scene_type"]

    drone = pick(name, DRONES)
    dist = hash_int(name, 30 if scene == "sky" else 20, 201 if scene == "sky" else 101)
    action = pick(name, ACTIONS_SKY if scene == "sky" else ACTIONS_GROUND)

    # IR detection from filename
    modality = "IR" if ("_ir_" in name.lower() or "inf" in name.lower()) else "RGB"

    if scene == "sky":
        cam_pos, elev = "bottom", hash_int(name, 20, 61)
        scene_t = pick(name, SCENES_SKY)
        desc = f"{drone} drone hovering at {dist}m, captured against open sky."
    else:
        cam_pos, elev = "top", hash_int(name, 0, 11)
        scene_t = pick(name, SCENES_GROUND)
        desc = f"{drone} drone flying at {dist}m altitude, aerial view of {scene_t} terrain below."

    j = {
        "drone_type": "quadrotor",
        "trajectory": [{
            "t": 0.0, "action": action,
            "distance": float(dist),
            "norm_u": round(hash_int(name, 30, 70) / 100.0, 2),
            "norm_v": round(hash_int(name, 20, 80) / 100.0, 2),
        }],
        "time_of_day": pick(name, TOD),
        "weather": pick(name, WEATHER),
        "scene_type": scene_t,
        "scene_description": desc,
        "modality": modality,
        "camera": {
            "position": cam_pos,
            "elevation_deg": float(elev),
            "fov_deg": 60.0,
        },
    }

    json_path = f"{JSON_OUT}/{name}.json"
    with open(json_path, "w") as jf:
        json.dump(j, jf, indent=2)
    p["json_path"] = json_path

NEW = f"{BASE}/output/phase2_mixed/pairs_diverse.jsonl"
with open(NEW, "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Generated {len(pairs)} diverse JSONs → {JSON_OUT}/")
print(f"Updated pairs → {NEW}")
